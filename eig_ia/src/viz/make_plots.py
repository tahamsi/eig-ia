import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from ..eval.calibration import compute_ece
from ..utils.io import ensure_dir, read_jsonl


def plot_method_diagram(out_dir: str) -> None:
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.1, 0.8, "Observation", fontsize=12, bbox=dict(boxstyle="round", fc="lightgray"))
    ax.text(0.35, 0.8, "Generate Q", fontsize=12, bbox=dict(boxstyle="round", fc="lightgray"))
    ax.text(0.6, 0.8, "EIG + Gate", fontsize=12, bbox=dict(boxstyle="round", fc="lightgray"))
    ax.text(0.85, 0.8, "Predict", fontsize=12, bbox=dict(boxstyle="round", fc="lightgray"))
    ax.annotate("", xy=(0.3, 0.8), xytext=(0.2, 0.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.55, 0.8), xytext=(0.45, 0.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.8, 0.8), xytext=(0.7, 0.8), arrowprops=dict(arrowstyle="->"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure1_method_diagram.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "figure1_method_diagram.pdf"))
    with open(os.path.join(out_dir, "figure1_method_diagram.txt"), "w", encoding="utf-8") as f:
        f.write("Method overview: Observation -> Question generation -> EIG selection and gate -> Prediction.\n")
    plt.close(fig)


def plot_delta_entropy(rows: List[Dict[str, Any]], out_dir: str) -> None:
    ensure_dir(out_dir)
    methods = sorted(set(r["method"] for r in rows))
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in methods:
        values = [r["delta_entropy"] for r in rows if r["method"] == method]
        ax.hist(values, bins=20, alpha=0.5, label=method)
    ax.set_xlabel("Delta entropy")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure2_delta_entropy.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "figure2_delta_entropy.pdf"))
    with open(os.path.join(out_dir, "figure2_delta_entropy.txt"), "w", encoding="utf-8") as f:
        f.write("Distribution of entropy reduction by method.\n")
    plt.close(fig)


def plot_reliability(rows: List[Dict[str, Any]], out_dir: str) -> None:
    ensure_dir(out_dir)
    ece, bins = compute_ece(rows)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], marker="o")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability (ECE={ece:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure3_reliability.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "figure3_reliability.pdf"))
    with open(os.path.join(out_dir, "figure3_reliability.txt"), "w", encoding="utf-8") as f:
        f.write("Reliability diagram with expected calibration error.\n")
    plt.close(fig)


def plot_cost_accuracy(rows: List[Dict[str, Any]], out_dir: str) -> None:
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter([r["tokens_total"] for r in rows], [r["accuracy"] for r in rows], alpha=0.6)
    ax.set_xlabel("Tokens total")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure4_accuracy_vs_tokens.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "figure4_accuracy_vs_tokens.pdf"))
    with open(os.path.join(out_dir, "figure4_accuracy_vs_tokens.txt"), "w", encoding="utf-8") as f:
        f.write("Accuracy vs token cost scatter plot.\n")
    plt.close(fig)


def make_plots(results_dir: str) -> None:
    rows = read_jsonl(os.path.join(results_dir, "per_example.jsonl"))
    fig_dir = os.path.join(results_dir, "figures")
    plot_method_diagram(fig_dir)
    plot_delta_entropy(rows, fig_dir)
    plot_reliability(rows, fig_dir)
    plot_cost_accuracy(rows, fig_dir)
