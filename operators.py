import random
import itertools
import logging
from typing import Any, Literal
from dataclasses import dataclass, field
import functional
import models
from lretyping import Layer
import torch
import data

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
    #h is most likely hidden state
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
    
@dataclass(frozen=True)
class Word2VecIclEstimator(LinearRelationEstimator):
    ...
