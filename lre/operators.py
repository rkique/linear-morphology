import random
import itertools
import logging
from typing import Any, Literal
from dataclasses import dataclass, field
import torch

import lre.models as models
import lre.functional as functional
from lre.lretyping import Layer
import lre.data as data
import baukit

logger = logging.getLogger(__name__)
from dataclasses_json import DataClassJsonMixin

@dataclass
class PredictedToken(DataClassJsonMixin):
    token: str
    prob: float

    def __str__(self) -> str:
        return f"{self.token} (p={self.prob:.3f})"

@dataclass
class RelationOutput:
    predictions: list[PredictedToken]


@dataclass
class RelationOperator:

    def __call__(self, subject: str, **kwargs:Any) -> RelationOutput:
        raise NotImplementedError

@dataclass
class LinearRelationOutput(RelationOutput):
    h: torch.Tensor
    z: torch.Tensor

    def as_relation_output(self) -> RelationOutput:
        return RelationOutput(predictions=self.predictions)

@dataclass
class LinearRelationOperator(RelationOperator):
    mt: models.ModelAndTokenizer
    weight: torch.Tensor | None
    bias: torch.Tensor | None
    h_layer: Layer
    z_layer: Layer
    prompt_template: str
    beta: float | None = None
    metadata: dict = field(default_factory=dict)

    def __call__(
            self,
            subject: str,
            k: int = 5,
            h: torch.Tensor | None = None,
            **kwargs: Any,
    ) -> LinearRelationOutput:
        if h is None:
            prompt = functional.make_prompt(
                mt=self.mt, prompt_template=self.prompt_template, subject=subject
            )
            logger.debug(f'computing h from prompt "{prompt}"')

            h_index, inputs = functional.find_subject_token_index(
                mt=self.mt, prompt=prompt, subject=subject
            )

            [[hs], _] = functional.compute_hidden_states(
                mt=self.mt, layers=[self.h_layer], inputs=inputs
            )
            h = hs[:, h_index]


@dataclass(frozen=True, kw_only=True)
class LinearRelationEstimator:
    """Abstract method for estimating a linear relation operator."""

    mt: models.ModelAndTokenizer

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        raise NotImplementedError

@dataclass(frozen=True, kw_only=True)
class JacobianEstimator(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None
    beta: float | None = None

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        _check_nonempty(
            samples=relation.samples, prompt_templates=relation.prompt_templates
        )
        _warn_gt_1(samples=relation.samples, prompt_templates=relation.prompt_templates)
        
        subject = relation.samples[0].subject
        #Takes first prompt template always (?)
        prompt_template = relation.prompt_templates[0]
        return self.estimate_for_subject(subject, prompt_template)
    
    def estimate_for_subject(
        self, subject: str, prompt_template: str
    ) -> LinearRelationOperator:
        prompt = functional.make_prompt(
            mt=self.mt, prompt_template=prompt_template, subject=subject
        )
        logger.debug(f"estimating J for prompt: \n" + prompt)

        h_index, inputs = functional.find_subject_token_index(
            mt=self.mt, prompt=prompt, subject=subject
        )
        logger.debug(f"subject={subject}, h_index={h_index}")

        #estimates h and z
        approx = functional.order_1_approx(
            mt=self.mt,
            prompt=prompt,
            h_layer=self.h_layer,
            h_index=h_index,
            z_layer=self.z_layer,
            z_index=-1,
            inputs=inputs
        )

        #use the new estimated (h_layer?) and z_layer.
        return LinearRelationOperator(
            mt=self.mt,
            weight=approx.weight,
            bias=approx.bias,
            h_layer=approx.h_layer,
            z_layer=approx.z_layer,
            prompt_template=prompt_template,
            beta=self.beta,
            metadata=approx.metadata
        )
    
@dataclass(frozen=True)
class JacobianIclEstimator(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None
    beta: float | None = None

    #this is called on a Relation to create a LinearRelationOperator
    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        _check_nonempty(
            samples=relation.samples, prompt_templates=relation.prompt_templates
        )
        _warn_gt_1(samples=relation.samples, prompt_templates=relation.prompt_templates)
        train = relation.samples[0]
        examples = relation.samples[1:]
        prompt_template = relation.prompt_templates[0]
        #Make a prompt with every other sample
        prompt_template_icl = functional.make_prompt(
            mt=self.mt, prompt_template=prompt_template, examples=examples, subject="{}"
        )

        return JacobianEstimator(
            mt=self.mt,
            h_layer=self.h_layer,
            z_layer=self.z_layer,
            beta=self.beta,
        ).estimate_for_subject(train.subject, prompt_template_icl)


#This inherits LinearRelationEstimator so it is also called on a Relation.
@dataclass(frozen=True)
class Word2VecIclEstimator(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None
    scaling_factor: float | None = None
    mode: Literal["icl", "zs"] = "icl"

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        _check_nonempty(
            samples=relation.samples, prompt_templates=relation.prompt_templates
        )
        _warn_gt_1(prompt_templates=relation.prompt_templates)
        device = models.determine_device(self.mt)
        dtype = models.determine_dtype(self.mt)
        samples = relation.samples
        #implying subject immediately predicts object (?)
        prompt_template = (
            self.mt.tokenizer.eos_token + " {}" if self.mode == "zs" else relation.prompt_templates[0]
        )
        
        H_stack: list[torch.Tensor] = []
        Z_stack: list[torch.Tensor] = []

        h_layer_name = models.determine_layer_paths(self.mt, [self.h_layer])[0]
        #set z_layer, then z_layer_names
        if self.z_layer is None:
            z_layer = models.determine_layers(self.mt)[-1]
        else:
            z_layer = self.z_layer
        z_layer_name = models.determine_layer_paths(self.mt, [z_layer])[0]

        training_samples = relation.samples
        
        offsets = []

        #We want to calculate Expected(o - s)
        #For each sample
        for sample_idx in range(len(training_samples)):
            #Make a prompt from sample
            sample = training_samples[sample_idx]
            if self.mode == "zs":
                prompt = prompt_template.format(sample.subject)
            elif self.mode == "icl":
                prompt = functional.make_prompt(
                    mt=self.mt,
                    prompt_template=prompt_template,
                    subject=sample.subject,
                    examples=training_samples[0:sample_idx]
                    + training_samples[sample_idx + 1 :],
                )
            #inputs are the tokenized prompt.
            h_index, inputs = functional.find_subject_token_index(
                mt=self.mt,
                prompt=prompt,
                subject=sample.subject
            )
            
            with baukit.TraceDict(
                self.mt.model,
                [h_layer_name, z_layer_name]
            ) as traces:
                self.mt.model(**inputs)
            
            h = functional.untuple(traces[h_layer_name].output)[0][h_index].detach()
            z = functional.untuple(traces[z_layer_name].output)[0][-1].detach()
            #(o - s)
            offsets.append(z - h)
        
        #Averages offset over each sample pair.
        offset = torch.stack(offsets).mean(dim=0)

        if self.mode == "icl":
            prompt_template = function.make_prompt(
                mt=self.mt,
                prompt_template=prompt_template,
                subject="{}",
                examples=training_samples,
            )
        
        operator = LinearRelationOperator(
            mt = self.mt,
            weight=None,
            bias=offset,
            h_layer=self.h_layer,
            z_layer=z_layer,
            prompt_template=prompt_template,
        )

        return operator


#these are both strange niche methods
#Checks that all keys do not pair to empty lists.
def _check_nonempty(**values: list) -> None:
    for key, value in values.items():
        if len(value) == 0:
            raise ValueError(f"expected at least one value for {key}")
        
def _warn_gt_1(**values: list) -> None:
    for key, value in values.items():
        if len(value) > 1:
            logger.warning(f"relation has > 1 {key}, will use first ({value[0]})")