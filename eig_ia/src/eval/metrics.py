import re
from typing import Any, Dict, List


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_accuracy(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for r in rows if int(r["pred"]) == int(r["gold"]))
    return correct / len(rows)


def compute_entropy_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    prior = [r.get("prior_entropy", 0.0) for r in rows]
    post = [r.get("posterior_entropy", 0.0) for r in rows]
    delta = [r.get("delta_entropy", 0.0) for r in rows]
    return {
        "prior_entropy": sum(prior) / len(rows) if rows else 0.0,
        "posterior_entropy": sum(post) / len(rows) if rows else 0.0,
        "delta_entropy": sum(delta) / len(rows) if rows else 0.0,
    }


def compute_eig(rows: List[Dict[str, Any]]) -> float:
    values = [r.get("eig_estimate", 0.0) for r in rows]
    return sum(values) / len(values) if values else 0.0


def compute_latency(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    latencies = [r.get("latency_total", 0.0) for r in rows]
    return {
        "latency_mean": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_median": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
    }


def compute_tokens(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    tokens = [r.get("tokens_total", 0.0) for r in rows]
    return {
        "tokens_mean": sum(tokens) / len(tokens) if tokens else 0.0,
        "tokens_total": sum(tokens),
    }


def compute_em_f1(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    em = [r.get("em", 0.0) for r in rows]
    f1 = [r.get("f1", 0.0) for r in rows]
    return {
        "em": sum(em) / len(em) if em else 0.0,
        "f1": sum(f1) / len(f1) if f1 else 0.0,
    }


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    metrics = {
        "accuracy": compute_accuracy(rows),
        "eig": compute_eig(rows),
    }
    metrics.update(compute_entropy_metrics(rows))
    metrics.update(compute_latency(rows))
    metrics.update(compute_tokens(rows))
    metrics.update(compute_em_f1(rows))
    return metrics
