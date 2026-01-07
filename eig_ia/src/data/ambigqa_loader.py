from typing import Any, Dict, List

from datasets import load_dataset


def _extract_example(example: Dict[str, Any]) -> Dict[str, Any]:
    question = example.get("question") or example.get("ambiguous_question") or ""
    rewrites = example.get("rewrites") or example.get("rewrite") or example.get("disambiguated_questions")
    answers = example.get("answers") or example.get("answer") or example.get("annotations")

    rewrite_texts = []
    answer_sets = []
    if isinstance(rewrites, list):
        for item in rewrites:
            if isinstance(item, dict):
                rewrite_texts.append(item.get("question", ""))
                answer_sets.append(item.get("answers", []))
            else:
                rewrite_texts.append(str(item))
                answer_sets.append([])
    elif isinstance(rewrites, dict):
        for _, item in rewrites.items():
            rewrite_texts.append(str(item))
            answer_sets.append([])

    if not answer_sets and isinstance(answers, list):
        for item in answers:
            if isinstance(item, dict):
                answer_sets.append(item.get("answer", item.get("answers", [])))
            else:
                answer_sets.append(item)

    while len(answer_sets) < len(rewrite_texts):
        answer_sets.append([])

    return {
        "id": str(example.get("id", example.get("question_id", ""))),
        "question": question.strip(),
        "rewrites": rewrite_texts,
        "answer_sets": answer_sets,
    }


def load_ambigqa(split: str = "validation", max_examples: int = 0) -> List[Dict[str, Any]]:
    dataset = None
    for name in ["ambig_qa", "ambigqa", "ambignq", "ambig_nq"]:
        try:
            dataset = load_dataset(name)
            break
        except Exception:
            continue
    if dataset is None:
        raise RuntimeError("Could not load AmbigQA/AmbigNQ dataset from Hugging Face.")
    ds_split = "validation" if split in {"dev", "validation"} else split
    examples = []
    for ex in dataset[ds_split]:
        parsed = _extract_example(ex)
        if not parsed["rewrites"]:
            continue
        examples.append(parsed)
        if max_examples and len(examples) >= max_examples:
            break
    return examples
