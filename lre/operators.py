import random
from datetime import datetime
import itertools
import logging
from typing import Any, Literal
from dataclasses import dataclass, field
import torch
import copy

import lre.models as models
import lre.functional as functional
from lre.lretyping import Layer
import lre.data as data
from lre.data import RelationSample
from baukit.baukit import TraceDict

from pathlib import Path
logger = logging.getLogger(__name__)
from dataclasses_json import DataClassJsonMixin

DEFAULT_N_ICL = 8

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



#Using a RelationOperator, predict top-k objects.
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
            target: RelationSample,
            k: int = 5,
            h: torch.Tensor | None = None,
            **kwargs: Any,
    ) -> LinearRelationOutput:
        #If no hidden subject state:
        if h is None:
            prompt = self.prompt_template.format(target.subject)
            #logger.info(f'computing h from prompt "{prompt}"')
            #retrieve subject_token_index
            h_index, inputs = functional.find_subject_token_index(
                mt=self.mt, prompt=prompt, subject=target.subject
            )

            [[hs], _] = functional.compute_hidden_states(
                mt=self.mt, layers=[self.h_layer], inputs=inputs
            )
            h = hs[:, h_index]
        else:
            logger.info("using precomputed h")

        #The linear approximation step: o = Ws
        z = h
        if self.weight is not None:
            z = z.mm(self.weight.t())

        #o = Ws + b
        #o = Ws * beta + b (IF there is a bias)
        if self.bias is not None:
            bias = self.bias
            if self.beta is not None:
                z = z * self.beta 
            z = z + bias

        #interpret z by way of lm_head & softmax
        lm_head = self.mt.lm_head if not self.z_layer == "ln_f" else self.mt.lm_head[:1]
        logits = lm_head(z)
        dist = torch.softmax(logits.float(), dim=-1)
        topk = dist.topk(dim=-1, k=k)
        probs = topk.values.view(k).tolist()
        token_ids = topk.indices.view(k).tolist()
        words = [self.mt.tokenizer.decode(token_id) for token_id in token_ids]

        return LinearRelationOutput(
            predictions=[
                functional.PredictedToken(token=w, prob=p) for w, p in zip(words, probs)
            ],
            h=h,
            z=z
        )

@dataclass(frozen=True, kw_only=True)
class LinearRelationEstimator:
    """Abstract method for estimating a linear relation operator.
    Uses multiple ModelAndTokenizers to process.
    """

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
        #Takes first prompt template always 
        prompt_template = relation.prompt_templates[0]
        return self.estimate_for_subject(subject, prompt_template)
    
    #gets J-estimate for one prompt (?)
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
        #approx weight and bias sourced from order_1_approx...
        #use the new estimated subj. rep. and obj. rep.
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
        #estimates for the first sample
        _warn_gt_1(samples=relation.samples, prompt_templates=relation.prompt_templates)
        train = relation.samples[0]
        examples = relation.samples[1:]
        prompt_template = relation.prompt_templates[0]
        #Make a prompt with every other sample
        prompt_template_icl = functional.make_prompt(
            template=prompt_template,
            examples=examples, 
            target="{}"
        )
        print(f'jacobian prompt_template_icl: {prompt_template_icl}')
        return JacobianEstimator(
            mt=self.mt,
            h_layer=self.h_layer,
            z_layer=self.z_layer,
            beta=self.beta,
        ).estimate_for_subject(train.subject, prompt_template_icl)

@dataclass(frozen=True)
class JacobianIclMeanEstimator(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None
    beta: float | None = None
    rank: int | None = None

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        _check_nonempty(
            samples=relation.samples, prompt_templates=relation.prompt_templates
        )
        #_warn_gt_1(prompt_templates=relation.prompt_templates)
        approxes = []
        def prompt_to_approx(mt, prompt_template, samples, prompt_kind):
            for s_o in range(20,27):
                for i in range(0, len(samples)):
                    sample = samples[i]
                    prompt = functional.make_prompt(
                            template=prompt_template,
                            target=sample,
                            examples=samples,
                        )
                    h_index, inputs = functional.find_subject_token_index(
                            mt=mt,
                            prompt=prompt,
                            subject=sample.subject
                    )
                    
                    directory = Path(f'kapprox/{sample.subject}')
                    
                    if not directory.exists():
                        directory.mkdir(parents=True, exist_ok=True)
                        
                    pth = f'kapprox/{sample.subject}/prompt_{s_o}.txt'
                    with open(pth, 'w') as file:
                        file.write(prompt)
                        
                    approx = functional.order_1_approx(
                            mt=mt,
                            prompt=prompt,
                            subject=sample.subject,
                            h_layer=self.h_layer,
                            h_index=h_index,
                            s_o_layer=s_o,
                            z_layer=self.z_layer,
                            z_index=-1,
                            inputs=inputs
                        )
                logger.info(f"{prompt_kind} [Jacobian] Finished order_1_approx for {sample} with s o layer {s_o}")
                approxes.append(approx)
        
        samples = random.sample(relation.samples, DEFAULT_N_ICL)
        prompt_template1 = relation.prompt_templates[0]
        prompt_template2 = relation.prompt_templates[1]
        nocontext_template = "{} "
        mt = self.mt
        
        prompt_to_approx(mt, prompt_template1, samples, "sem1")
        #prompt_to_approx(mt, prompt_template2, samples, "sem2")
        # prompt_to_approx(mt, nocontext_template, samples, "noc")
        
        # weight = torch.eye(4096)
        # bias = torch.ones(4096)
        
        if self.rank is not None:
            weight = functional.low_rank_approx(matrix=weight,rank =self.rank)
        
        #TODO: add metadata
        operator = LinearRelationOperator(
            mt=self.mt,
            weight=weight,
            bias=bias,
            h_layer=self.h_layer,
            z_layer=approxes[0].z_layer,
            prompt_template=prompt_template,
            beta=self.beta
        )

        return operator
    
#This inherits LinearRelationEstimator so it is also called on a Relation.
@dataclass(frozen=True)
class Word2VecIclEstimator(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None
    scaling_factor: float | None = None
    mode: Literal["icl", "zs"] = "icl"

    def __call__(self, relation: data.Relation, beta: int) -> LinearRelationOperator:
        _warn_gt_1(prompt_templates=relation.prompt_templates)
        device = models.determine_device(self.mt)
        dtype = models.determine_dtype(self.mt)
        samples = relation.samples
        #zs mode seems to be no-context (prompt directly followed by "")
        prompt_template = (
            self.mt.tokenizer.eos_token + " {}" if self.mode == "zs" else relation.prompt_templates[0]
        )
        logging.info(f'[W2VEstimator call] using {relation.prompt_templates[0]}')

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

        #Calculate Expected(o - s)
        #For each sample
        for sample_idx in range(len(training_samples)):
            #Make a prompt from the sample
            sample = training_samples[sample_idx]
            if self.mode == "zs":
                prompt = prompt_template.format(sample.subject)
            #omitting the sample_idx.
            elif self.mode == "icl":
                prompt = functional.make_prompt(
                    template=prompt_template,
                    target=sample,
                    examples=training_samples[0:sample_idx]
                    + training_samples[sample_idx + 1 :],
                )
            #inputs are the tokenized prompt.
            h_index, inputs = functional.find_subject_token_index(
                mt=self.mt,
                prompt=prompt,
                subject=sample.subject
            )
            inputs = inputs.to(device)
            with TraceDict(
                self.mt.model,
                [h_layer_name, z_layer_name]
            ) as traces:
                self.mt.model(**inputs)
            
            h = functional.untuple(traces[h_layer_name].output)[0][h_index].detach()
            z = functional.untuple(traces[z_layer_name].output)[0][-1].detach()
            #(o - s)
            offsets.append((z - h))
        
        #Averages offset over each sample pair.
        offset = torch.stack(offsets).mean(dim=0)

        # if self.mode == "icl":
        #     prompt_template = functional.make_prompt(
        #         mt=self.mt,
        #         prompt_template=prompt_template,
        #         subject="{}",
        #         examples=training_samples,
        #     )

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
