import csv
from typing import Any, Dict, List

from ..utils.io import ensure_dir


def make_human_eval_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    out_dir = out_csv.rsplit("/", 1)[0] if "/" in out_csv else ""
    if out_dir:
        ensure_dir(out_dir)
    fieldnames = [
        "id",
        "dataset",
        "O",
        "method",
        "q",
        "a",
        "prediction",
        "gold",
        "helpfulness_1_5",
        "relevance_1_5",
        "plausibility_1_5",
        "hallucination_1_5",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.get("example_id"),
                    "dataset": row.get("dataset"),
                    "O": row.get("observation"),
                    "method": row.get("method"),
                    "q": row.get("q"),
                    "a": row.get("a"),
                    "prediction": row.get("pred"),
                    "gold": row.get("gold"),
                    "helpfulness_1_5": "",
                    "relevance_1_5": "",
                    "plausibility_1_5": "",
                    "hallucination_1_5": "",
                }
            )
