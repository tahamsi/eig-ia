from typing import Dict


def oracle_answer_art(question: str, example: Dict) -> str:
    h1, h2 = example["hypotheses"]
    label = int(example["label"])
    q_lower = question.lower()
    if h1 and h1.lower() in q_lower:
        return "yes" if label == 1 else "no"
    if h2 and h2.lower() in q_lower:
        return "yes" if label == 2 else "no"
    return "yes" if label == 1 else "no"


def oracle_answer_ambig(rewrite: str) -> str:
    return rewrite.strip()


def oracle_answer(dataset: str, question: str, example: Dict) -> str:
    if dataset == "art":
        return oracle_answer_art(question, example)
    if not example.get("rewrites"):
        return ""
    return oracle_answer_ambig(example["rewrites"][0])
