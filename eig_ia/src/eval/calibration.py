from typing import Any, Dict, List, Tuple


def compute_ece(rows: List[Dict[str, Any]], n_bins: int = 10) -> Tuple[float, List[Dict[str, float]]]:
    if not rows:
        return 0.0, []
    bins = [{"count": 0, "conf": 0.0, "acc": 0.0} for _ in range(n_bins)]
    for r in rows:
        conf = float(r.get("confidence", 0.0))
        correct = 1.0 if int(r["pred"]) == int(r["gold"]) else 0.0
        idx = min(n_bins - 1, int(conf * n_bins))
        bins[idx]["count"] += 1
        bins[idx]["conf"] += conf
        bins[idx]["acc"] += correct
    ece = 0.0
    total = len(rows)
    for b in bins:
        if b["count"] == 0:
            continue
        b["conf"] /= b["count"]
        b["acc"] /= b["count"]
        ece += (b["count"] / total) * abs(b["acc"] - b["conf"])
    reliability = [
        {"bin": i, "count": b["count"], "conf": b["conf"], "acc": b["acc"]}
        for i, b in enumerate(bins)
    ]
    return ece, reliability
