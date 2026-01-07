import math
from typing import List


def entropy(probs: List[float]) -> float:
    total = 0.0
    for p in probs:
        if p > 0:
            total -= p * math.log(p + 1e-12)
    return total


def max_prob(probs: List[float]) -> float:
    return max(probs) if probs else 0.0
