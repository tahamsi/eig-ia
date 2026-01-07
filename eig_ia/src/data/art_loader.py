from typing import Any, Dict, List

from datasets import load_dataset


def _extract_example(example: Dict[str, Any]) -> Dict[str, Any]:
    keys = set(example.keys())
    if "obs1" in keys and "obs2" in keys:
        obs = f"{example['obs1']} {example['obs2']}"
    elif "observation_1" in keys and "observation_2" in keys:
        obs = f"{example['observation_1']} {example['observation_2']}"
    else:
        obs = " ".join(str(example[k]) for k in keys if "obs" in k)
    h1 = example.get("hypothesis_1") or example.get("hyp1") or example.get("h1")
    h2 = example.get("hypothesis_2") or example.get("hyp2") or example.get("h2")
    label = example.get("label") or example.get("answer") or example.get("correct")
    if isinstance(label, str):
        try:
            label = int(label)
        except Exception:
            label = 1 if label.lower().strip() in {"1", "a", "h1"} else 2
    if isinstance(label, int) and label in {0, 1}:
        label = label + 1
    return {
        "id": str(example.get("id", example.get("story_id", ""))),
        "observation": obs.strip(),
        "hypotheses": [h1, h2],
        "label": int(label),
    }


def load_art(split: str = "validation", max_examples: int = 0) -> List[Dict[str, Any]]:
    dataset = None
    for name in ["art", "alpha_nli", "anli"]:
        try:
            dataset = load_dataset(name)
            break
        except Exception:
            continue
    if dataset is None:
        raise RuntimeError("Could not load ART/alphaNLI dataset from Hugging Face.")
    ds_split = "validation" if split in {"dev", "validation"} else split
    examples = []
    for ex in dataset[ds_split]:
        examples.append(_extract_example(ex))
        if max_examples and len(examples) >= max_examples:
            break
    return examples
