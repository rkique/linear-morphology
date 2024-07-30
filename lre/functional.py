from dataclasses_json import DataClassJsonMixin
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Sequence, Optional

from lre.data import Relation,RelationSample
import lre.models as models
from lre.lretyping import Layer, ModelInput, ModelOutput, StrSequence
from lre.metrics import is_nontrivial_prefix, any_is_nontrivial_prefix
import lre.tokenizer_utils as tokenizer_utils
from baukit.baukit import TraceDict, parameter_names, get_parameter

import random
import transformers
import torch
import torch.nn as nn
import logging
from pathlib import Path

import numpy as np
from llra.build import layer_norm

logger = logging.getLogger(__name__)

START_LAYER, END_LAYER = 10,31
APPROX_FOLDER = f'llama_{START_LAYER}_{END_LAYER}_approx'
H_LAYER_NAME = f'model.layers.{START_LAYER}'
Z_LAYER_NAME = f'model.layers.{END_LAYER}'

DEFAULT_BATCH_SIZE = 1
DEFAULT_N_ICL = 8 
DEFAULT_N_TOP_LM = 1

#Build a prompt from a template string, target, and examples. 
#Defaults to using the first object.
def make_prompt(template: str,
                target: RelationSample,
                examples: Optional[list[RelationSample]] = None,
                sep_token: str = "\n") -> str:
    
    prompt = template.format(target.subject)
    if examples != None:
      others = [x for x in examples if x != target]
      random.shuffle(others)
      prompt = (
                  sep_token.join(
                      template.format(x.subject) + f" {x.object[0]}" for x in others
                  )
                  + sep_token
                  + prompt
              )
      #prompt = models.maybe_prefix_eos(mt, prompt) (?)
    return prompt

#Returns the subject token index, but also the list of tokens.
def find_subject_token_index(*,
                             prompt: str,
                             subject: str,
                             offset: int = -1,
                             mt: models.ModelAndTokenizer) -> tuple[int, ModelInput]:
    device = models.determine_device(mt)
    inputs = mt.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping").to(device)

    if "token_type_ids" in inputs:  # llama tokenizer has this annoying field
        inputs.pop("token_type_ids")
        
    # Find the last occurrence of the subject
    subject_i, subject_j = tokenizer_utils.find_token_range(
        prompt, subject, offset_mapping=offset_mapping[0], occurrence=-1
    )
    #convert the offset to an absolute index.
    subject_token_index = tokenizer_utils.offset_to_absolute_index(
        subject_i, subject_j, offset
    )
    # print(f"subject_token_index is {subject_token_index}")
    return subject_token_index, inputs

class ComputeHiddenStatesOutput(NamedTuple):
    """The output of `compute_hidden_states`."""

    hiddens: list[torch.Tensor]
    outputs: ModelOutput

@torch.no_grad()
def compute_hidden_states(
    *,
    mt: models.ModelAndTokenizer,
    layers: Sequence[Layer],
    prompt: str | StrSequence | None = None,
    inputs: ModelInput | None = None,
    **kwargs: Any,
) -> ComputeHiddenStatesOutput:
    device = models.determine_device(mt)
    if (prompt is None) == (inputs is None):
        raise ValueError("Must pass either `prompt` or `inputs`, not both.")

    if inputs is None:
        assert prompt is not None
        inputs = mt.tokenizer(prompt, return_tensors="pt", padding="longest").to(
            device
        )
    inputs = inputs.to(device)

    #TraceDict follows the layer paths.
    layer_paths = models.determine_layer_paths(mt, layers=layers, return_dict=True)
    with TraceDict(mt.model, layer_paths.values()) as ret:
        outputs = mt.model(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, **kwargs
        )
    #Assemble hidden state outputs from the TraceDict.
    hiddens = []
    for layer in layers:
        h = untuple(ret[layer_paths[layer]].output)
        hiddens.append(h)

    return ComputeHiddenStatesOutput(hiddens=hiddens, outputs=outputs)

"""Saves the first-order weight and bias of the LM Jacobian between `h` and `z`.
    h_layer: Layer to take h from.
    h_index: Token index for h.
    z_layer: Layer to take z from.
    z_index: Token index for z.
    inputs: Precomputed tokenized inputs, recomputed if not set.
   Returns None.
"""
@torch.no_grad()
@torch.inference_mode(mode=False)
def order_1_approx(
    *,
    mt: models.ModelAndTokenizer,
    prompt: str,
    prompt_kind: str,
    relation_name: str,
    subject: str,
    h_layer: Layer,
    h_index: int,
    h: torch.Tensor | None = None,
    z_layer: Layer | None = None,
    z_index: int | None = None,
    inputs: ModelInput | None = None,
) -> list[dict]:
    approxes = []
    device = models.determine_device(mt)
    
    def save_grads(h_layer_name, z_layer_name, mt,
                   prompt, prompt_kind, relation_name, subject, h_index, 
                   h, z_index, inputs):
        z_index = -1
        inputs = inputs or mt.tokenizer(prompt, return_tensors="pt").to(device)
        inputs = inputs.to(device)
        
        # Precompute everything up to the subject (h_index), if there is anything before it.
        input_ids = inputs.input_ids
        
        #Runs the model and sets up layer hooks with TraceDict
        with TraceDict(mt.model, layers=(h_layer_name,
                                         z_layer_name)) as ret:
            outputs =  mt.model(input_ids=input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            
        use_cache = past_key_values is not None
        
        s_j = ret[h_layer_name].output[0][0, h_index] #s_j
        o_j1 = ret[z_layer_name].output[0][0, -1] #o_j1
        
        def o_j1_from_s_j(s_j: torch.Tensor) -> torch.Tensor:
            
            def insert_s_j(output: tuple, layer: str) -> tuple:
                hs = untuple(output)
                if layer != h_layer_name:
                    logger.warn(f"[insert_s_j] layer {layer} does not match {h_layer_name}")
                    return output
                hs[0, h_index] = s_j
                
            with TraceDict(mt.model, (h_layer_name, z_layer_name), 
                                    edit_output=insert_s_j) as ret:
                mt.model(input_ids=input_ids, 
                         use_cache=use_cache,
                        past_key_values=past_key_values)
            z = untuple(ret[z_layer_name].output)[0, -1]
            return z.half().to(device)

        logging.info(f"[order_1_approx] starting weight calculation for {prompt}")
        
        s_j = s_j.half().to(device)
        o_j1 = o_j1.half().to(device)

        s_o_weight = torch.autograd.functional.jacobian(o_j1_from_s_j, s_j).half().to(device)
        s_o_bias = o_j1[None] - s_j[None].mm(s_o_weight.t())
        
        logging.info(f"""[order_1_approx] weight calculation finished \n
                        {s_j=} \n
                        {o_j1=} \n
                        s_o_weight: {s_o_weight} \n
                        {s_o_bias=} \n
                    """)
        h_layer = h_layer_name.split(".")[2]
        z_layer = z_layer_name.split(".")[2]
        torch.save(s_o_weight, f'{APPROX_FOLDER}/{relation_name}/{subject}/s_o_weight_{h_layer}_{z_layer}.pt')
        torch.save(s_o_bias, f'{APPROX_FOLDER}/{relation_name}/{subject}/s_o_bias_{h_layer}_{z_layer}.pt')
    
    save_grads(H_LAYER_NAME, Z_LAYER_NAME, mt,
               prompt, prompt_kind, relation_name, subject,
               h_index, h, z_index, inputs)
    
    # save_grads("transformer.h.1", "transformer.h.27", mt, 
    #        prompt, prompt_kind,
    #        relation_name, subject, h_index, h, z_index, inputs)

    # for j in range(h_layer, z_layer):
    #     save_grads(f"transformer.h.{j}",f"transformer.h.{j+1}", mt, 
    #     prompt, prompt_kind, relation_name, subject, h_index, h, z_index, inputs)

    # NB(evan): Something about the jacobian computation causes a lot of memory
    # fragmentation, or some kind of memory leak. This seems to help.
    torch.cuda.empty_cache()

    return None

@dataclass(frozen=True, kw_only=True)
class PredictedToken(DataClassJsonMixin):
    token: str
    prob: float

    def __str__(self) -> str:
        return f"{self.token} -> {self.prob:.3f}"

@torch.inference_mode()
def predict_next_token(
    *,
    mt: models.ModelAndTokenizer,
    prompt: str | StrSequence,
    k: int = 5,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[PredictedToken]]:
    device = models.determine_device(mt)
    if isinstance(prompt, str):
        prompt = [prompt]
        
    if isinstance(mt.model, \
                  transformers.models.gemma2.modeling_gemma2.Gemma2ForCausalLM):
        padding_side = "right"
    else:
        padding_side = "left"
        
    #pad all inputs left to the longest length.
    with models.set_padding_side(mt, padding_side=padding_side):
        inputs = mt.tokenizer(prompt, return_tensors="pt", padding="longest").to(device)
    with torch.inference_mode():
        predictions = []
        #for each batch_size group of input_ids.
        for i in range(0, len(inputs.input_ids), batch_size):

            #get model output
            batch_outputs = mt.model(
                input_ids=inputs.input_ids[i: i + batch_size],
                attention_mask=inputs.attention_mask[i : i+batch_size]
            )
            
            #get output logits->probs->topk probs
            next_token_probs = batch_outputs.logits[:,-1].float().softmax(dim=-1)
            next_token_topk = next_token_probs.topk(dim=-1, k=k)

            for token_ids, token_probs in zip(
                next_token_topk.indices, next_token_topk.values
            ):
                predictions.append(
                    [
                        PredictedToken(
                            token=mt.tokenizer.decode(token_id), prob=prob.item() #np --> py scalar
                        )
                        for token_id, prob in zip(token_ids, token_probs)
                    ]
                )
    return predictions

#Filters samples based on the prompt being known.
@torch.inference_mode()
def filter_relation_samples(
    *,
    mt: models.ModelAndTokenizer,
    test_relation: Relation,
    prompt_template: str,
    n_top_lm: int = DEFAULT_N_TOP_LM,
    batch_size: int = DEFAULT_BATCH_SIZE,
    subj_token_filter: Literal["all", "single", "multi"] = "all",
) -> Relation:

    test_prompts = [
        make_prompt(template=prompt_template, target=sample, examples=test_relation.samples) for sample in test_relation.samples
    ]
    
    predictions = predict_next_token(mt=mt, prompt=test_prompts, k=n_top_lm)

    filtered_samples = []
    for sample, prediction in zip(test_relation.samples, predictions):
        known_flag = any_is_nontrivial_prefix(
            predictions=[prediction[0].token],
            targets=sample.object
        )
    filtered = set(samples=sorted(filtered_samples, key=lambda x: x.subject))
    print(len(filtered))
    return filtered
            
def untuple(x: Any) -> Any:
    """If `x` is a tuple, return the first element."""
    if isinstance(x, tuple):
        return x[0]
    return x
