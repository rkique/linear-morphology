#We take three methods from functional: make_prompt, find_subject_token_index, compute_hidden_states
from dataclasses_json import DataClassJsonMixin
from dataclasses import dataclass, field
import models
from typing import Any, Literal, NamedTuple, Sequence
from lretyping import Layer, ModelInput, ModelOutput, StrSequence
import baukit
import tokenizer_utils
import torch

'''
An (s,o) pair. 
'''
@dataclass
class RelationSample(DataClassJsonMixin):

    subject: str
    object: str

    def __str__(self) -> str:
        return f"{self.subject} -> {self.object}"

'''
Builds a prompt from a template string and subject.
'''
def make_prompt(*,
                prompt_template: str,
                subject: str,
                examples: Sequence[RelationSample] | None = None,
                mt: models.ModelAndTokenizer | None = None,
                ) -> str:
    #replaces {} with subject.
    prompt = prompt_template.format(subject)
    if examples is not None:
        others = [x for x in examples if x.subject != subject]
        prompt = (
            "\n".join(
                prompt_template.format(x.subject) + f" {x.object}" for x in others
            )
            + "\n"
            + prompt
        )   
    #TODO: uncomment this
    #prompt = models.maybe_prefix_eos(mt, prompt)
    return prompt

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

    #need models.determine_layer_paths for TraceDict
    layer_paths = models.determine_layer_paths(mt, layers=layers, return_dict=True)
    #TraceDict is an important function.
    with baukit.TraceDict(mt.model, layer_paths.values()) as ret:
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
    with baukit.TraceDict(
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

        with baukit.TraceDict(
            mt.model, (h_layer_name, z_layer_name), edit_output=insert_h
        ) as ret:
            mt.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        return untuple(ret[z_layer_name].output)[0, -1]

    assert h is not None
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

def untuple(x: Any) -> Any:
    """If `x` is a tuple, return the first element."""
    if isinstance(x, tuple):
        return x[0]
    return x