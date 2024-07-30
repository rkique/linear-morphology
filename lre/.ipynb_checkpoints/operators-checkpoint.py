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

import os
from pathlib import Path
logger = logging.getLogger(__name__)
from dataclasses_json import DataClassJsonMixin

from lre.functional import START_LAYER, END_LAYER, H_LAYER_NAME,\
                        Z_LAYER_NAME, APPROX_FOLDER, DEFAULT_N_ICL

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

def read_prompts(root_dir):
    subjects = []
    prompts = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath == root_dir:
            continue
        if 'prompt.txt' in filenames:
            file_path = os.path.join(dirpath, 'prompt.txt')
            with open(file_path, 'r') as file:
                prompt = file.read()
                words = prompt.split()
                subjects.append(words[-2])
                prompts.append(prompt)
    return (subjects,prompts)
    
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

        #MAKE RELATION FOLDER
        directory = Path(f'{APPROX_FOLDER}/{relation.name}')
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            
        def prompt_to_approx(mt, prompt_template, samples, prompt_kind):
            for sample in samples:
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
                #MAKE RELATION SAMPLE FOLDER
                directory = Path(f'{APPROX_FOLDER}/{relation.name}/{sample.subject}')
                
                if not directory.exists():
                    directory.mkdir(parents=True, exist_ok=True)
                
                pth = f'{APPROX_FOLDER}/{relation.name}/{sample.subject}/prompt.txt'
                with open(pth, 'w') as file:
                    file.write(prompt)
                    
                approx = functional.order_1_approx(
                        mt=mt,
                        prompt=prompt,
                        prompt_kind=prompt_kind,
                        relation_name=relation.name,
                        subject=sample.subject,
                        h_layer=self.h_layer,
                        h_index=h_index,
                        z_layer=self.z_layer,
                        z_index=-1,
                        inputs=inputs
                    )
            
                logger.info(f"{prompt_kind} [Jacobian] Finished order_1_approx for {prompt}")
                approxes.append(approx)
        
        samples = random.sample(relation.samples, DEFAULT_N_ICL)
            #samples = [sample for sample in relation.samples if sample.subject in spaced_samples]
        prompt_template1 = relation.prompt_templates[0]
        mt = self.mt
        prompt_to_approx(mt, prompt_template1, samples, "sem1")
            
            # weight = torch.eye(4096)
            # bias = torch.ones(4096)
            
        if self.rank is not None:
            weight = functional.low_rank_approx(matrix=weight,rank =self.rank)
    
        return None
        
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

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:
        _warn_gt_1(prompt_templates=relation.prompt_templates)
        h_layer = START_LAYER
        z_layer = END_LAYER
        device = models.determine_device(self.mt)
        dtype = models.determine_dtype(self.mt)
        samples = relation.samples
        prompt_template = relation.prompt_templates[0]
        logging.info(f'[W2VEstimator] using {START_LAYER} and {END_LAYER} layers on {relation.name}')

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
        #print(f'{training_samples=}')
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

        #MAKE RELATION SAMPLE FOLDER
        directory = Path(f'{APPROX_FOLDER}/{relation.name}/{sample.subject}')
        
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
        
        pth = f'{APPROX_FOLDER}/{relation.name}/{sample.subject}/prompt.txt'
        
        with open(pth, 'w') as file:
            file.write(prompt)
            torch.save(offset, f'{APPROX_FOLDER}/{relation.name}/offset_{h_layer}_{z_layer}.pt')
                    
        # if self.mode == "icl":
        #     prompt_template = functional.make_prompt(
        #         mt=self.mt,
        #         prompt_template=prompt_template,
        #         subject="{}",
        #         examples=training_samples,
        #     )
        return None
        
        operator = LinearRelationOperator(
            mt = self.mt,
            weight=None,
            bias=offset,
            h_layer=self.h_layer,
            z_layer=z_layer,
            prompt_template=prompt_template,
        )

        return operator

@dataclass(frozen=True)
class LearnedLinearEstimator(LinearRelationEstimator):
    h_layer: Layer
    z_layer: Layer | None = None
    mode: Literal["zs", "icl"] = "zs"
    n_steps: int = 100
    lr: float = 5e-2
    weight_decay: float = 2e-2

    def __call__(self, relation: data.Relation) -> LinearRelationOperator:

        device = models.determine_device(self.mt)
        dtype = models.determine_dtype(self.mt)
        samples = relation.samples
        prompt_template = (
            self.mt.tokenizer.eos_token + " {}"
            if self.mode == "zs"
            else relation.prompt_templates[0]
        )

        H_stack: list[torch.Tensor] = []
        Z_stack: list[torch.Tensor] = []

        if self.z_layer is None:
            z_layer = models.determine_layers(self.mt)[-1]

        h_layer_name, z_layer_name = models.determine_layer_paths(
            self.mt, [self.h_layer, z_layer]
        )

        for sample in samples:
            if self.mode == "zs":
                prompt = prompt_template.format(sample.subject)
            elif self.mode == "icl":
                prompt = functional.make_prompt(
                    mt=self.mt,
                    prompt_template=prompt_template,
                    subject=sample.subject,
                    examples=samples,
                )
            h_index, inputs = functional.find_subject_token_index(
                mt=self.mt,
                prompt=prompt,
                subject=sample.subject,
            )

            with baukit.TraceDict(
                self.mt.model,
                [h_layer_name, z_layer_name],
            ) as traces:
                self.mt.model(**inputs)
            #stack of subject hidden states
            H_stack.append(
                functional.untuple(traces[h_layer_name].output)[0][h_index].detach()
            )
            #stack of object hidden states
            Z_stack.append(
                functional.untuple(traces[z_layer_name].output)[0][-1].detach()
            )
        #to matrix
        H = torch.stack(H_stack, dim=0).to(torch.float32)
        Z = torch.stack(Z_stack, dim=0).to(torch.float32)

        #4096
        n_embd = models.determine_hidden_size(self.mt)
        #[4096, 4096],[1, 4096]
        weight = torch.empty(n_embd, n_embd, device=device)
        bias = torch.empty(1, n_embd, device=device)
        weight.requires_grad = True
        bias.requires_grad = True

        optimizer = torch.optim.Adam(
            [weight, bias], lr=self.lr, weight_decay=self.weight_decay
        )

        #attempt to learn the relation between H and Z through MSE loss over n_steps
        for _ in range(self.n_steps):
            Z_hat = H.mm(weight.t()) + bias
            loss = (Z - Z_hat).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.mode == "icl":
            prompt_template = functional.make_prompt(
                mt = self.mt,
                prompt_template=prompt_template,
                subject="{}",
                examples=samples
            )
        
        operator = LinearRelationOperator(
            mt = self.mt,
            weight=weight.detach().to(dtype).to(device),
            bias=bias.detach().to(dtype).to(device),
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
