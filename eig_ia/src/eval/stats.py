import csv
import math
import random
from typing import Any, Dict, List


def paired_bootstrap(acc_a: List[int], acc_b: List[int], n: int, alpha: float, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        sample = [rng.randrange(len(acc_a)) for _ in range(len(acc_a))]
        a = sum(acc_a[i] for i in sample) / len(sample)
        b = sum(acc_b[i] for i in sample) / len(sample)
        diffs.append(a - b)
    diffs.sort()
    lower = diffs[int((alpha / 2) * n)]
    upper = diffs[int((1 - alpha / 2) * n) - 1]
    return {"diffs": diffs, "ci_low": lower, "ci_high": upper}


def cohens_kappa(rater_a: List[int], rater_b: List[int], n_classes: int = 5) -> float:
    total = len(rater_a)
    if total == 0:
        return 0.0
    agree = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / total
    marg_a = [0] * n_classes
    marg_b = [0] * n_classes
    for a, b in zip(rater_a, rater_b):
        marg_a[a - 1] += 1
        marg_b[b - 1] += 1
    exp = sum((marg_a[i] / total) * (marg_b[i] / total) for i in range(n_classes))
    if exp == 1.0:
        return 0.0
    return (agree - exp) / (1 - exp)


def krippendorff_alpha_nominal(ratings: List[List[int]]) -> float:
    values = [r for row in ratings for r in row if r is not None]
    if not values:
        return 0.0
    categories = sorted(set(values))
    n = len(values)

    def delta(a: int, b: int) -> float:
        return 0.0 if a == b else 1.0

    Do = 0.0
    for row in ratings:
        valid = [r for r in row if r is not None]
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                Do += delta(valid[i], valid[j])
    if n <= 1:
        return 0.0
    Do /= (n - 1)

    De = 0.0
    for c in categories:
        for c2 in categories:
            p_c = values.count(c) / n
            p_c2 = values.count(c2) / n
            De += p_c * p_c2 * delta(c, c2)
    if De == 0.0:
        return 0.0
    return 1 - (Do / De)


def _read_ratings(csv_path: str, columns: List[str]) -> List[List[int]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratings = []
            for col in columns:
                value = row.get(col)
                ratings.append(int(value) if value and value.isdigit() else None)
            rows.append(ratings)
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--columns", required=True, help="Comma-separated rater columns")
    args = parser.parse_args()

    columns = [c.strip() for c in args.columns.split(",")]
    ratings = _read_ratings(args.csv, columns)
    if len(columns) >= 2:
        r1 = [r[0] for r in ratings if r[0] is not None and r[1] is not None]
        r2 = [r[1] for r in ratings if r[0] is not None and r[1] is not None]
        print(f"Cohen's kappa: {cohens_kappa(r1, r2):.4f}")
    print(f"Krippendorff's alpha (nominal): {krippendorff_alpha_nominal(ratings):.4f}")


if __name__ == "__main__":
    main()
