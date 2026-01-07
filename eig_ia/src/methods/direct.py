from typing import Any, Dict

from ..modules.hypothesis_scorer import score_hypotheses


def run_direct(dataset: str, example: Dict[str, Any], llm_scorer) -> Dict[str, Any]:
    observation = example["observation"] if dataset == "art" else example["question"]
    hypotheses = example["hypotheses"] if dataset == "art" else example["rewrites"]
    probs, meta = score_hypotheses(dataset, observation, hypotheses, llm_scorer)
    pred = int(probs.index(max(probs)))
    return {
        "asked": False,
        "question": "",
        "answer": "",
        "prior_probs": probs,
        "posterior_probs": probs,
        "pred": pred,
        "meta": {"scorer": meta},
    }
