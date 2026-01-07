from typing import List

from .posterior import max_prob


def should_ask(prior_probs: List[float], eig_value: float, tau: float, gamma: float) -> bool:
    return max_prob(prior_probs) < tau or eig_value > gamma
