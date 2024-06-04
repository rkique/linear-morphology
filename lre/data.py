from dataclasses import dataclass, fields
from typing import Literal, Sequence
import random
from dataclasses_json import DataClassJsonMixin
from collections import defaultdict


#(s,o) text pair.
@dataclass(frozen=True)
class RelationSample(DataClassJsonMixin):
    subject: str
    object: str

    def __str__(self) -> str:
        return f"{self.subject} -> {self.object}"
    
#z may refer to the hidden layer (?)
#_domain is an explicit list of possible subjects, accessed from @property domain
#_range is an explicit list of possible objects.
#Relation class specifies both the template and the samples for that template, it is
#really a very large class

@dataclass(frozen=True)
class Relation(DataClassJsonMixin):
    name: str
    prompt_templates: list[str]
    #This is a list of prompts that are sampled randomly for benchmarking
    prompt_templates_zs: list[str]
    samples: list[RelationSample]
    #properties: RelationProperties

    _domain: list[str] | None = None
    _range: list[str] | None = None

    #if domain and range are not explicit, get them from samples.
    @property
    def domain(self) -> set[str]:
        if self._domain is not None:
            return set(self._domain)
        #set
        return {sample.subject for sample in self.samples}
    
    @property
    def range(self) -> set[str]:
        if self._range is not None:
            return set(self._range)
        return {sample.object for sample in self.samples}
    
    def without(self, sample: RelationSample) -> "Relation":
        return self.set(samples=[s for s in self.samples if s != sample])
    
    #train / test split
    def split(self, train_size: int, test_size: int | None = None
              ) -> tuple["Relation", "Relation"]:
        if train_size > len(self.samples):
            raise ValueError(f"size must be <= {len(self.samples)}, got: {train_size}")
        if test_size is None:
            test_size = len(self.samples) - train_size
        
        samples = self.samples.copy()
        random.shuffle(samples)

        samples_by_object = defaultdict(list)
        #groups samples by object and shuffles
        for sample in samples:
            samples_by_object[sample.object].append(sample)

        for samples in samples_by_object.values():
            random.shuffle(samples)
        
        max_coverage_samples = []

        #iterate through grouped samples and append to list.
        while samples_by_object:
            for object in list(samples_by_object.keys()):
                max_coverage_samples.append(samples_by_object[object].pop(0))
                if len(samples_by_object[object]) == 0:
                    del samples_by_object[object]
        
        train_samples = max_coverage_samples[:train_size]
        test_samples = max_coverage_samples[train_size : train_size + test_size]

        #return Relation with train_samples and test_samples.
        return(
            Relation(
                name=self.name,
                prompt_templates=self.prompt_templates,
                prompt_templates_zs=self.prompt_templates_zs,
                properties=self.properties,
                samples=train_samples,
                _domain=list(self.domain),
                _range=list(self.range),
            ),
            Relation(
                name=self.name,
                prompt_templates=self.prompt_templates,
                prompt_templates_zs=self.prompt_templates_zs,
                properties=self.properties,
                samples=test_samples,
                _domain=list(self.domain),
                _range=list(self.range)
            )
        )

