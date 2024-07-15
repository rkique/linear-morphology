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


@dataclass(frozen=True, kw_only=True)
class Order1ApproxOutput:
    """A first-order approximation.

    Attributes:
        weight: The weight matrix.
        bias: The bias vector.
        h: The subject hidden state.
        h_layer: The layer of h.
        h_index: The token index of h.
        z: The (true) object hidden state.
        z_layer: The layer of z.
        z_index: The token index of z.
        inputs: The LM inputs used to compute the approximation.
        logits: The LM logits, shape (batch_size, length, vocab_size).
    """

    weight: torch.Tensor
    bias: torch.Tensor

    h: torch.Tensor
    h_layer: Layer
    h_index: int

    z: torch.Tensor
    z_layer: Layer
    z_index: int

    inputs: ModelInput
    logits: torch.Tensor

    metadata: dict = field(default_factory=dict)

#z is the true object hidden state.
#Order1ApproxOutput: w, b, {h, h_layer, h_index}, {z, z_layer, z_index}, inputs -> logits (batch_size, length, vocab_size)

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
    """Compute a first-order approximation of the LM between `h` and `z`.
        h_layer: Layer to take h from.
        h_index: Token index for h.
        z_layer: Layer to take z from.
        z_index: Token index for z.
        inputs: Precomputed tokenized inputs, recomputed if not set.
    Returns:
        The approximation.
    """
    approxes = []
    device = models.determine_device(mt)
    
    def save_grads(h_layer, z_layer, mt, prompt, prompt_kind, relation_name, subject, h_index, h, z_index, inputs):
        z_layer = z_layer or models.determine_layers(mt)[-1]
        z_index = -1
        inputs = inputs or mt.tokenizer(prompt, return_tensors="pt").to(device)
        inputs = inputs.to(device)
        
        # Precompute everything up to the subject (h_index), if there is anything before it.
        # past_key_values = None
        # input_ids = inputs.input_ids
        # _h_index = h_index
        # if _h_index > 0:
        #     outputs = mt.model(input_ids=input_ids[:, :_h_index], use_cache=True)
        #     past_key_values = outputs.past_key_values
        #     input_ids = input_ids[:, _h_index:]
        #     _h_index = 0
        # use_cache = past_key_values is not None
        
        # layer transformer.h.27
        # transformer.h.5.ln_1
        
        #SET H_LAYER_NAME TO AFTER LAYERNORM
        [h_layer_name, z_layer_name] = models.determine_layer_paths(mt, [h_layer, z_layer])
        #h_layer_name = h_layer_name + '.ln_1'
        logging.info(f'{h_layer_name=} {z_layer_name=}')

        #edit _h_index to s_j
        def edit_output(output: tuple, layer: str) -> tuple:
            if layer != h_layer_name:
                return output
            untuple(output)[:, _h_index] = s_j
            return output
            
        #Runs the model while tracking with TraceDict
        with TraceDict(mt.model, layers=(h_layer_name, z_layer_name), edit_output=edit_output) as ret:
            outputs = mt.model(
                input_ids=input_ids,
                use_cache=False,
                past_key_values=past_key_values,
            )
            
        s_j = untuple(ret[h_layer_name].output)[0, _h_index] #s_j
        o_j = untuple(ret[h_layer_name].output)[0, z_index] #o_j
        s_j1 = untuple(ret[z_layer_name].output)[0, _h_index] #s_j+1
        o_j1 = untuple(ret[z_layer_name].output)[0, z_index] #o_j+1
    
        logging.info(f"""[order_1_approx] weight calculation finished \n
                        s_j: {s_j} \n
                        o_j: {o_j} \n
                        s_j1: {s_j1} \n
                        o_j1: {o_j1} \n
                    """)
    
        def compute_o_j1_from_s_j(s_j: torch.Tensor) -> torch.Tensor:
            
            def insert_s_j(output: tuple, layer: str) -> tuple:
                hs = untuple(output)
                if layer != h_layer_name:
                    logger.warn(f"[insert_s_j] layer {layer} does not match {h_layer_name}")
                    return output
                hs[0, _h_index] = s_j
                return output
                
            with TraceDict(mt.model, (h_layer_name, z_layer_name), edit_output=insert_s_j) as ret:
                mt.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=False)
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
                
            with TraceDict(mt.model, (h_layer_name, z_layer_name), edit_output=insert_s_j) as ret:
                mt.model(input_ids=input_ids,past_key_values=past_key_values, use_cache=use_cache)
            _h = untuple(ret[z_layer_name].output)[0, _h_index] #subj position
            return _h.half().to(device)
    
        logging.info(f"[order_1_approx] starting weight calculation for {prompt}")
        s_j = s_j.half().to(device)
        o_j = o_j.half().to(device)

        s_o_weight = torch.autograd.functional.jacobian(compute_o_j1_from_s_j, s_j).half().to(device)
        #s_s_weight = torch.autograd.functional.jacobian(compute_s_j1_from_s_j, s_j).half().to(device)
        
        #edit to o_j
        def edit_output(output: tuple, layer: str) -> tuple:
            if layer != h_layer_name:
                return output
            untuple(output)[:, -1] = o_j
            return output
            
        def compute_o_j1_from_o_j(o_j: torch.Tensor) -> torch.Tensor:
            def insert_o_j(output: tuple, layer: str) -> tuple:
                hs = untuple(output)
                if layer != h_layer_name:
                    logger.warn(f"[insert_o_j] layer {layer} does not match {h_layer_name}")
                    return output
                hs[0, -1] = o_j
                return output
            with TraceDict(mt.model, (h_layer_name, z_layer_name), edit_output=insert_o_j) as ret:
                mt.model(input_ids=input_ids,past_key_values=past_key_values, use_cache=use_cache)
            z = untuple(ret[z_layer_name].output)[0, -1] #obj position
            return z.half().to(device)
            
        #o_o_weight = torch.autograd.functional.jacobian(compute_o_j1_from_o_j, o_j).half().to(device)

        #s_o_weight = torch.eye(4096).half().to(device)
        # s_s_weight = torch.eye(4096).half().to(device)
        # o_o_weight = torch.eye(4096).half().to(device)
        
        #weight = torch.eye(4096).half().to(device)
        
        logging.info(f"""[order_1_approx] weight calculation finished \n
                        s_o: {s_o_weight} \n
                    """)
        
        s_o_bias = o_j1[None] - s_j[None].mm(s_o_weight.t())
        #s_s_bias = s_j1[None] - s_j[None].mm(s_s_weight.t())
        #o_o_bias = o_j1[None] - o_j[None].mm(o_o_weight.t())
    
            
        torch.save(s_o_weight, f'dapprox/{relation_name}/{subject}/s_o_weight_{h_layer}_{z_layer}.pt')
        torch.save(s_o_bias, f'dapprox/{relation_name}/{subject}/s_o_bias_{h_layer}_{z_layer}.pt')
        
        # torch.save(s_s_weight, f'dapprox/{relation_name}/{subject}/s_s_weight_{h_layer}_{z_layer}.pt')# _{prompt_kind}.pt')
        # torch.save(s_s_bias, f'dapprox/{relation_name}/{subject}/s_s_bias_{h_layer}_{z_layer}.pt')# _{prompt_kind}.pt')
    
        # torch.save(o_o_weight, f'dapprox/{relation_name}/{subject}/o_o_weight_{h_layer}_{z_layer}.pt')# _{prompt_kind}.pt')
        # torch.save(o_o_bias, f'dapprox/{relation_name}/{subject}/o_o_bias_{h_layer}_{z_layer}.pt')# _{prompt_kind}.pt')
    
    save_grads(5, 27, mt, prompt, prompt_kind, relation_name, subject, h_index, h, z_index, inputs)
    # save_grads(21, 26, mt, prompt, prompt_kind, relation_name, subject, h_index, h, z_index, inputs)
    # save_grads(26, 27, mt, prompt, prompt_kind, relation_name, subject, h_index, h, z_index, inputs)
    
    #save_grads(1, 27, mt, prompt, prompt_kind, relation_name, subject, h_index, h, z_index, inputs)
    # for j in range(h_layer, z_layer):
    #     save_grads(j, j, mt, prompt, prompt_kind, relation_name, subject, h_index, h, z_index, inputs)

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