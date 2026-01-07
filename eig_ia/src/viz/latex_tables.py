import os
from typing import Dict, List

from ..utils.io import ensure_dir, read_csv, read_jsonl


def _table_from_metrics(metrics_rows: List[Dict[str, str]], out_path: str, caption: str, label: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    headers = ["dataset", "method", "accuracy", "em", "f1", "delta_entropy", "eig", "ece", "latency_mean", "tokens_mean"]
    lines = ["\\begin{table}[t]", "\\centering", "\\begin{tabular}{l l r r r r r r r r}", "\\toprule"]
    lines.append(" & ".join(headers) + " \\")
    lines.append("\\midrule")
    for row in metrics_rows:
        cells = [row.get(h, "") for h in headers]
        lines.append(" & ".join(cells) + " \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _bucket_table(rows: List[Dict[str, str]], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    lines = ["\\begin{table}[t]", "\\centering", "\\begin{tabular}{l l l r}", "\\toprule"]
    lines.append("dataset & bucket & method & accuracy \\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(f"{row['dataset']} & {row['bucket']} & {row['method']} & {row['accuracy']:.4f} \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\caption{Robustness buckets.}", "\\label{tab:robustness}", "\\end{table}"])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _compute_bucket_rows(per_example: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows = []
    for dataset in sorted(set(r["dataset"] for r in per_example)):
        dataset_rows = [r for r in per_example if r["dataset"] == dataset]
        methods = sorted(set(r["method"] for r in dataset_rows))
        if dataset == "art":
            for method in methods:
                for bucket_name, lo, hi in [("low", 0.0, 0.33), ("med", 0.33, 0.66), ("high", 0.66, 1.01)]:
                    subset = [r for r in dataset_rows if r["method"] == method and lo <= float(r["confidence"]) < hi]
                    if not subset:
                        continue
                    acc = sum(int(r["accuracy"]) for r in subset) / len(subset)
                    rows.append({"dataset": dataset, "bucket": bucket_name, "method": method, "accuracy": acc})
        else:
            for method in methods:
                for bucket_name, lo, hi in [("2", 0, 2), ("3-4", 3, 4), ("5+", 5, 100)]:
                    subset = [r for r in dataset_rows if r["method"] == method and lo <= len(r.get("hypotheses", [])) <= hi]
                    if not subset:
                        continue
                    acc = sum(int(r["accuracy"]) for r in subset) / len(subset)
                    rows.append({"dataset": dataset, "bucket": bucket_name, "method": method, "accuracy": acc})
    return rows


def make_tables(results_dir: str) -> None:
    metrics_path = os.path.join(results_dir, "metrics.csv")
    metrics_rows = read_csv(metrics_path)
    table_dir = os.path.join(results_dir, "tables")
    _table_from_metrics(metrics_rows, os.path.join(table_dir, "table1_main.tex"), "Main results.", "tab:main")
    _table_from_metrics(metrics_rows, os.path.join(table_dir, "table2_ablations.tex"), "Ablations.", "tab:ablations")
    _table_from_metrics(metrics_rows, os.path.join(table_dir, "table3_human.tex"), "Human evaluation summary.", "tab:human")

    per_example = read_jsonl(os.path.join(results_dir, "per_example.jsonl"))
    bucket_rows = _compute_bucket_rows(per_example)
    _bucket_table(bucket_rows, os.path.join(table_dir, "table4_robustness.tex"))
