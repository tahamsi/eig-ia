from collections import Counter
from typing import Any, Dict, List, Tuple

from ..modules.answer_simulator import simulate_answer
from ..modules.hypothesis_scorer import score_hypotheses
from .posterior import entropy, max_prob


def estimate_eig(
    dataset: str,
    observation: str,
    hypotheses: List[str],
    question: str,
    prior_probs: List[float],
    llm_answer,
    llm_scorer,
    m_answers: int,
    estimator: str,
) -> Tuple[float, Dict[str, Any]]:
    answers, meta = simulate_answer(dataset, question, llm_answer, m_answers)
    counts = Counter(answers)
    eig_values = []
    for answer, count in counts.items():
        posterior, _ = score_hypotheses(dataset, observation, hypotheses, llm_scorer, question, answer)
        if estimator == "utility":
            eig_values.append((count / m_answers) * (max_prob(posterior) - max_prob(prior_probs)))
        else:
            eig_values.append((count / m_answers) * (entropy(prior_probs) - entropy(posterior)))
    return sum(eig_values), {"answers": answers, **meta}
