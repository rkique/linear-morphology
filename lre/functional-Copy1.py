#We take three methods from functional: make_prompt, find_subject_token_index, compute_hidden_states
from dataclasses_json import DataClassJsonMixin
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Sequence, Optional

from lre.data import Relation,RelationSample
import lre.models as models
from lre.lretyping import Layer, ModelInput, ModelOutput, StrSequence
from lre.metrics import is_nontrivial_prefix, any_is_nontrivial_prefix
import lre.tokenizer_utils as tokenizer_utils
from baukit.baukit import TraceDict

import random
import torch
import logging
from pathlib import Path

from llra.build import layer_norm

logger = logging.getLogger(__name__)

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

#(Misleading) Returns the subject token index, but also the list of tokens.
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
    subject_token_index = tokenizer_utils.offset_to_absolute_index(
        subject_i, subject_j, offset
    )

    return subject_token_index, inputs

class ComputeHiddenStatesOutput(NamedTuple):
    """The output of `compute_hidden_states`."""

    hiddens: list[torch.Tensor]
    outputs: ModelOutput

@torch.no_grad()
def compute_hidden_states(
    *,
    mt: models.ModelAndTokenizer,c thing up to the subject (h_index), if there is anything before it.
        # past_key_values = None
        input_ids = inputs.input_ids
        _h_index = h_index
        if _h_index > 0:
            outputs = mt.model(input_ids=input_ids[:, :_h_index], use_cache=True)
            past_key_values = outputs.past_key_values
            input_ids = input_ids[:, _h_index:]
            _h_index = 0
        use_cache = past_key_values is not None
        
        #SET H_LAYER_NAME TO AFTER LAYERNORM
        #[h_layer_name, z_layer_name] = models.determine_layer_paths(mt, [h_layer, z_layer])
        #h_layer_name = h_layer_name + '.ln_1'
        logging.info(f'{h_layer_name=} {z_layer_name=}')
            
        #Runs the model while tracking with TraceDict
        #We're not interested in using edit_output
        with TraceDict(mt.model, layers=(h_layer_name, z_layer_name)) as ret:
            outputs = mt.model(
                input_ids=input_ids,
                use_cache=use_cache,
                past_key_values=past_key_values
            )
            
        s_j = untuple(ret[h_layer_name].output)[0, _h_index] #s_j
        o_j = untuple(ret[h_layer_name].output)[0, -1] #o_j
        s_j1 = untuple(ret[z_layer_name].output)[0, _h_index] #s_j+1
        o_j1 = untuple(ret[z_layer_name].output)[0, -1] #o_j+1
    
        def compute_o_j1_from_s_j(s_j: torch.Tensor) -> torch.Tensor:
            def insert_s_j(output: tuple, layer: str) -> tuple:
                hs = untuple(output)
                if layer != h_layer_name:
                    logger.warn(f"[insert_s_j] layer {layer} does not match {h_layer_name}")
                    return output
                hs[0, _h_index] = s_j
                return output
                
            with TraceDict(mt.model, (h_layer_name, z_layer_name)) as ret:
                mt.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache)
            z = untuple(ret[z_layer_name].output)[0, -1] #obj position
            return z.half().to(device)
            
        def compute_s_j1_from_s_j(s_j: torch.Tensor) -> torch.Tensor:
            def insert_s_j(output: tuple, layer: str) -> tuple:
                hs = untuple(output)
                if layer != h_layer_name:
                    logger.warn(f"[insert_s_j] layer {layer} does not match {h_layer_name}")
                    return output
                hs[0, _h_index] = s_j
                return output
                
            with TraceDict(mt.model, (h_layer_name, z_layer_name), edit_output=None) as ret:
                mt.model(input_ids=input_ids,past_key_values=past_key_values, use_cache=use_cache)
            _h = untuple(ret[z_layer_name].output)[0, _h_index] #subj position
            return _h.half().to(device)
    
        logging.info(f"[order_1_approx] starting weight calculation for {prompt}")
        s_j = s_j.half().to(device)
        o_j = o_j.half().to(device)

        s_o_weight = torch.autograd.functional.jacobian(compute_o_j1_from_s_j, s_j).half().to(device)
        s_o_bias = o_j1[None] - s_j[None].mm(s_o_weight.t())

    
        torch.save(s_o_weight, f'dapprox/{relation_name}/{subject}/s_o_weight_{h_layer_name}_{z_layer_name}.pt')
        torch.save(s_o_bias, f'dapprox/{relation_name}/{subject}/s_o_bias_{h_layer_name}_{z_layer_name}.pt')
        
    save_grads("transformer.h.1", "transformer.h.27", mt, 
           prompt, prompt_kind,
           relation_name, subject, h_index, h, z_index, inputs)

    for j in range(h_layer, z_layer):
        save_grads(f"transformer.h.{j}",f"transformer.h.{j+1}", mt, 
        prompt, prompt_kind, relation_name, subject, h_index, h, z_index, inputs)
        
    torch.cuda.empty_cache()

    return None

@dataclass(frozen=True, kw_only=True)
class PredictedToken(DataClassJsonMixin):
    token: str
    prob: float

    def __str__(self) -> str:
        return f"{self.token} -> {self.prob:.3f}"

#Predicted LM token is (token, prob). k is used in probs.topk
#returns list[list[PredictedToken]]. Why?
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
    #pad all inputs left to the longest length.
    with models.set_padding_side(mt, padding_side="left"):
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
