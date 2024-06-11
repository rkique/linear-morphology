#We take three methods from functional: make_prompt, find_subject_token_index, compute_hidden_states
from dataclasses_json import DataClassJsonMixin
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Sequence

from lre.data import Relation,RelationSample
import lre.models as models
from lre.lretyping import Layer, ModelInput, ModelOutput, StrSequence
from lre.metrics import is_nontrivial_prefix
import lre.tokenizer_utils as tokenizer_utils
from baukit.baukit import TraceDict

import random
import torch
import logging

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 1
DEFAULT_N_ICL_LM = 5 #not used
DEFAULT_N_TOP_LM = 1

'''
Builds a prompt from a template string, subject, and examples. 
'''
def make_prompt(*,
                prompt_template: str,
                subject: RelationSample,
                examples: Sequence[RelationSample] | None = None,
                mt: models.ModelAndTokenizer | None = None,
                ) -> str:
    #replace {} with subject.
    #Modified to work with multiple objects
    # Examples are already filtered for the subject, no need to do that.
    #print(f'the subject is {subject} with type {type(subject)}')
    prompt = prompt_template.format(subject.subject)
    if examples is not None:
            objects = []
            others = [x for x in examples if x.subject != subject]
            for x in others:
                for object in x.object:
                    objects.append((x.subject, object))
            objects = random.sample(objects, 8)
            prompt = (
                        "\n".join(
                            prompt_template.format(x[0]) + f" {x[1]}" for x in objects
                        )
                        + "\n"
                        + prompt
                    )
            
    #TODO: Prefix prompt with EOS token if model has no special start token.
    #prompt = models.maybe_prefix_eos(mt, prompt)
    return prompt

#(Misleading) Returns the subject token index, but also the list of tokens.
def find_subject_token_index(*,
                             prompt: str,
                             subject: str,
                             offset: int = -1,
                             mt: models.ModelAndTokenizer) -> tuple[int, ModelInput]:
    inputs = mt.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True).to(
        mt.model.device
    )
    offset_mapping = inputs.pop("offset_mapping")
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
    if (prompt is None) == (inputs is None):
        raise ValueError("Must pass either `prompt` or `inputs`, not both.")

    if inputs is None:
        assert prompt is not None
        inputs = mt.tokenizer(prompt, return_tensors="pt", padding="longest").to(
            mt.model.device
        )

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
    h_layer: Layer,
    h_index: int,
    h: torch.Tensor | None = None,
    z_layer: Layer | None = None,
    z_index: int | None = None,
    inputs: ModelInput | None = None,
) -> Order1ApproxOutput:
    """Compute a first-order approximation of the LM between `h` and `z`.

    Very simply, this computes the Jacobian of z with respect to h, as well as
    z - Jh to approximate the bias.

    Args:
        mt: The model.
        prompt: Prompt to approximate.
        h_layer: Layer to take h from.
        h_index: Token index for h.
        h: will calculate approximation based on this hidden state, if provided.
        z_layer: Layer to take z from.
        z_index: Token index for z.
        inputs: Precomputed tokenized inputs, recomputed if not set.

    Returns:
        The approximation.

    """
    if z_layer is None:
        z_layer = models.determine_layers(mt)[-1]
    if z_index is None:
        z_index = -1
    if inputs is None:
        inputs = mt.tokenizer(prompt, return_tensors="pt").to(mt.model.device)

    # Precompute everything up to the subject, if there is anything before it.
    past_key_values = None
    input_ids = inputs.input_ids
    _h_index = h_index
    # Sets outputs if h_index > 0
    if _h_index > 0:
        outputs = mt.model(input_ids=input_ids[:, :_h_index], use_cache=True)
        past_key_values = outputs.past_key_values
        input_ids = input_ids[:, _h_index:]
        _h_index = 0
    #sets flag for precomputing
    use_cache = past_key_values is not None

    # Precompute initial h and z.
    #These are names (?)
    [h_layer_name, z_layer_name] = models.determine_layer_paths(mt, [h_layer, z_layer])

    #defines an edit_output function (?)
    edit_output: function | None = None
    if h is not None:
        def edit_output(output: tuple, layer: str) -> tuple:
            if layer != h_layer_name:
                return output
            untuple(output)[:, _h_index] = h
            return output

    else:
        edit_output = None

    #Runs the model to produce outputs with past_key_values and use_cache from precompute
    with TraceDict(
        mt.model, layers=(h_layer_name, z_layer_name), edit_output=edit_output
    ) as ret:
        outputs = mt.model(
            input_ids=input_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
    h = untuple(ret[h_layer_name].output)[0, _h_index]
    z = untuple(ret[z_layer_name].output)[0, z_index]

    # Now compute J and b.
    # The function the Jacobian comes from.
    def compute_z_from_h(h: torch.Tensor) -> torch.Tensor:
        #What does edit_output do?
        def insert_h(output: tuple, layer: str) -> tuple:
            hs = untuple(output)
            if layer != h_layer_name:
                return output
            hs[0, _h_index] = h
            return output

        with TraceDict(
            mt.model, (h_layer_name, z_layer_name), edit_output=insert_h
        ) as ret:
            mt.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        return untuple(ret[z_layer_name].output)[0, -1]

    assert h is not None
    #weight and bias are calculated here, for a single input.
    weight = torch.autograd.functional.jacobian(compute_z_from_h, h, vectorize=True)
    bias = z[None] - h[None].mm(weight.t())

    approx = Order1ApproxOutput(
        h=h,
        h_layer=h_layer,
        h_index=h_index,
        z=z,
        z_layer=z_layer,
        z_index=z_index,
        weight=weight,
        bias=bias,
        inputs=inputs.to("cpu"),
        logits=outputs.logits.cpu(),
        metadata={
            "Jh": weight @ h,
        },
    )

    # NB(evan): Something about the jacobian computation causes a lot of memory
    # fragmentation, or some kind of memory leak. This seems to help.
    torch.cuda.empty_cache()

    return approx

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
    if isinstance(prompt, str):
        prompt = [prompt]
    #pad all inputs left to the longest length.
    with models.set_padding_side(mt, padding_side="left"):
        inputs = mt.tokenizer(prompt, return_tensors="pt", padding="longest").to("cuda")
    print(f'model input is {inputs[0]}')
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
    examples: Sequence[RelationSample] = [],
    subj_token_filter: Literal["all", "single", "multi"] = "all",
) -> Relation:
    if len(examples) > 0:
        logger.debug(f'filtering for knowns using prompt "{prompt_template}"')
        prompt_template = make_prompt(
            mt=mt,
            prompt_template=prompt_template,
            subject="{}",
            examples=examples
        )
    logger.debug(f'filtering for knowns using prompt "{prompt_template}')

    test_prompts = [
        prompt_template.format(sample.subject) for sample in test_relation.samples
    ]
    predictions = predict_next_token(
        mt=mt, prompt=test_prompts, k=n_top_lm, batch_size=batch_size
    )

    filtered_samples = []
    for sample, prediction in zip(test_relation.samples, predictions):
        known_flag = is_nontrivial_prefix(
            prediction=prediction[0].token, target=sample.object
        )
        logger.info()
    return test_relation.set(samples=sorted(filtered_samples, key=lambda x: x.subject))
            
def untuple(x: Any) -> Any:
    """If `x` is a tuple, return the first element."""
    if isinstance(x, tuple):
        return x[0]
    return x