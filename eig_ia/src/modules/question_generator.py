from typing import Any, Dict, List, Tuple

from ..data.prompt_templates import ART_QUESTION_TEMPLATE, AMBIGQA_QGEN_PROMPT
from ..llm.llm_base import LLMBase


def generate_questions_art(observation: str, hypotheses: List[str], llm: LLMBase, k: int) -> Tuple[List[str], Dict[str, Any]]:
    questions = [ART_QUESTION_TEMPLATE.format(hypothesis=h) for h in hypotheses if h]
    while len(questions) < k:
        questions.append(questions[len(questions) % max(1, len(questions))])
    return questions[:k], {"source": "template"}


def generate_questions_ambig(question: str, rewrites: List[str], llm: LLMBase, k: int) -> Tuple[List[str], Dict[str, Any]]:
    prompt = AMBIGQA_QGEN_PROMPT.format(
        question=question,
        rewrites="\n".join([f"- {r}" for r in rewrites[:10]]),
    )
    outputs, meta = llm.generate(prompt, n=k)
    questions = [o.split("\n")[0].strip() for o in outputs]
    return questions, {"prompt": prompt, **meta}


def generate_questions(dataset: str, observation: str, hypotheses: List[str], llm: LLMBase, k: int) -> Tuple[List[str], Dict[str, Any]]:
    if dataset == "art":
        return generate_questions_art(observation, hypotheses, llm, k)
    return generate_questions_ambig(observation, hypotheses, llm, k)
