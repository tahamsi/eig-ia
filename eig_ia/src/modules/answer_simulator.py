from typing import Any, Dict, List, Tuple

from ..llm.llm_base import LLMBase


def simulate_answer_art(question: str, llm: LLMBase, n: int) -> Tuple[List[str], Dict[str, Any]]:
    prompt = f"Answer yes or no.\nQuestion: {question}\nAnswer:"
    outputs, meta = llm.generate(prompt, n=n)
    answers = []
    for out in outputs:
        text = out.strip().lower()
        if text.startswith("y"):
            answers.append("yes")
        elif text.startswith("n"):
            answers.append("no")
        else:
            answers.append("yes" if "yes" in text else "no")
    return answers, {"prompt": prompt, **meta}


def simulate_answer_ambig(question: str, llm: LLMBase, n: int) -> Tuple[List[str], Dict[str, Any]]:
    prompt = f"Question: {question}\nAnswer:"
    outputs, meta = llm.generate(prompt, n=n)
    answers = [o.strip() for o in outputs]
    return answers, {"prompt": prompt, **meta}


def simulate_answer(dataset: str, question: str, llm: LLMBase, n: int) -> Tuple[List[str], Dict[str, Any]]:
    if dataset == "art":
        return simulate_answer_art(question, llm, n)
    return simulate_answer_ambig(question, llm, n)
