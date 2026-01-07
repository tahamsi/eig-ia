from typing import Any, Dict, List

from ..eig.eig_estimator import estimate_eig
from ..eig.gating import should_ask
from ..modules.answer_simulator import simulate_answer
from ..modules.hypothesis_scorer import score_hypotheses
from ..modules.oracle_answerers import oracle_answer
from ..modules.question_generator import generate_questions


def run_eig_ia(
    dataset: str,
    example: Dict[str, Any],
    llm_q,
    llm_a,
    llm_scorer,
    mode: str,
    k: int,
    m: int,
    estimator: str,
    gate_enabled: bool,
    tau: float,
    gamma: float,
) -> Dict[str, Any]:
    observation = example["observation"] if dataset == "art" else example["question"]
    hypotheses = example["hypotheses"] if dataset == "art" else example["rewrites"]
    prior_probs, prior_meta = score_hypotheses(dataset, observation, hypotheses, llm_scorer)

    questions, q_meta = generate_questions(dataset, observation, hypotheses, llm_q, k)
    eig_scores: List[float] = []
    eig_meta: List[Dict[str, Any]] = []
    for q in questions:
        eig_value, meta = estimate_eig(dataset, observation, hypotheses, q, prior_probs, llm_a, llm_scorer, m, estimator)
        eig_scores.append(eig_value)
        eig_meta.append(meta)

    best_idx = max(range(len(questions)), key=lambda i: eig_scores[i])
    best_q = questions[best_idx]
    best_eig = eig_scores[best_idx]

    if gate_enabled and not should_ask(prior_probs, best_eig, tau, gamma):
        pred = int(prior_probs.index(max(prior_probs)))
        return {
            "asked": False,
            "question": "",
            "answer": "",
            "prior_probs": prior_probs,
            "posterior_probs": prior_probs,
            "eig": best_eig,
            "pred": pred,
            "meta": {"question": q_meta, "eig": eig_meta, "scorer": prior_meta},
        }

    if mode == "oracle":
        answer = oracle_answer(dataset, best_q, example)
        a_meta = {"source": "oracle"}
    else:
        answers, a_meta = simulate_answer(dataset, best_q, llm_a, 1)
        answer = answers[0]

    posterior_probs, post_meta = score_hypotheses(dataset, observation, hypotheses, llm_scorer, best_q, answer)
    pred = int(posterior_probs.index(max(posterior_probs)))
    return {
        "asked": True,
        "question": best_q,
        "answer": answer,
        "prior_probs": prior_probs,
        "posterior_probs": posterior_probs,
        "eig": best_eig,
        "pred": pred,
        "meta": {"question": q_meta, "eig": eig_meta, "scorer": post_meta, "answer": a_meta},
    }
