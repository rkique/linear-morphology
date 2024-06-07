from dataclasses import dataclass
from typing import Sequence
import lre.functional
#ArrayLike is list OR tuple OR np.array OR tensor
#StrSequence is list OR tuple of strings
from lre.lretyping import ArrayLike, StrSequence

import numpy as np
from dataclasses_json import DataClassJsonMixin

#matching any character prefix makes for a potentially flawed analysis: does it work in practice?

def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    # if len(prediction) > 0 and target.startswith(prediction):
    #     print(f"{prediction} matches {target}")
    return len(prediction) > 0 and target.startswith(prediction)

def any_is_nontrivial_prefix(predictions: StrSequence, target: str) -> bool:
    return any(is_nontrivial_prefix(p, target) for p in predictions)

#TODO: look at tokenized outputs.
#recall@1: top1 is correct, recall@2: top2 is correct.
def recall(predictions: Sequence[StrSequence], targets: StrSequence) -> list[float]:
    _validate_same_length(predictions=predictions, target=targets)
    if len(predictions) == 0:
        return None
    
    k = max(map(len, predictions))
    recalls = [0.0] * k
    for topk, target in zip(predictions, targets):
        for i in range(k):
            if any_is_nontrivial_prefix(topk[: i + 1], target):
                recalls[i] += 1
    
    return [r / len(targets) for r in recalls]

def _validate_same_length(**kwargs: Sequence | ArrayLike) -> None:
    lengths = {key: len(seq) for key, seq in kwargs.items()}
    # >1 unique lengths
    if len(set(lengths.values())) > 1:
        message = f"inconsistent batch sizes:" + "\n\t"
        message += "\n\t".join(f"{key}={length}" for key, length in lengths.items())
        raise ValueError(message)
