from typing import Any, Dict

from .eig_ia import run_eig_ia


def run_dpo_question_ranker(
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
    result = run_eig_ia(
        dataset,
        example,
        llm_q,
        llm_a,
        llm_scorer,
        mode,
        k,
        m,
        estimator,
        gate_enabled,
        tau,
        gamma,
    )
    result["meta"]["dpo_note"] = "DPO ranker not trained; using EIG proxy ranking."
    return result
