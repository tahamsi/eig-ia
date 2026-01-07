import math
from typing import Any, Dict, List, Tuple

from ..data.prompt_templates import (
    AMBIGQA_PRIOR_PROMPT,
    AMBIGQA_SCORE_PROMPT,
    ART_PRIOR_PROMPT,
    ART_SCORE_PROMPT,
)
from ..llm.llm_base import LLMBase


def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    max_s = max(scores)
    exp_scores = [math.exp(s - max_s) for s in scores]
    denom = sum(exp_scores)
    return [s / denom for s in exp_scores]


def score_hypotheses_art(observation: str, hypotheses: List[str], llm: LLMBase, question: str = "", answer: str = "") -> Tuple[List[float], Dict[str, Any]]:
    if question:
        prompt = ART_SCORE_PROMPT.format(observation=observation, question=question, answer=answer)
    else:
        prompt = ART_PRIOR_PROMPT.format(observation=observation)
    scores, meta = llm.score(prompt, hypotheses)
    return _normalize(scores), {"prompt": prompt, **meta}


def score_hypotheses_ambig(question: str, rewrites: List[str], llm: LLMBase, clarifying_question: str = "", answer: str = "") -> Tuple[List[float], Dict[str, Any]]:
    if clarifying_question:
        prompt = AMBIGQA_SCORE_PROMPT.format(question=question, clarifying_question=clarifying_question, answer=answer)
    else:
        prompt = AMBIGQA_PRIOR_PROMPT.format(question=question)
    scores, meta = llm.score(prompt, rewrites)
    return _normalize(scores), {"prompt": prompt, **meta}


def score_hypotheses(dataset: str, observation: str, hypotheses: List[str], llm: LLMBase, question: str = "", answer: str = "") -> Tuple[List[float], Dict[str, Any]]:
    if dataset == "art":
        return score_hypotheses_art(observation, hypotheses, llm, question, answer)
    return score_hypotheses_ambig(observation, hypotheses, llm, question, answer)
