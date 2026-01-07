import argparse
import os
from typing import Any, Dict, List

import yaml

from src.data.art_loader import load_art
from src.data.ambigqa_loader import load_ambigqa
from src.eig.posterior import entropy, max_prob
from src.eval.calibration import compute_ece
from src.eval.metrics import compute_metrics, f1_score, normalize_text
from src.eval.stats import paired_bootstrap
from src.eval.human_eval_prep import make_human_eval_csv
from src.llm.api_llm import APILLM
from src.llm.hf_llm import HFLLM
from src.methods.direct import run_direct
from src.methods.random_question import run_random_question
from src.methods.generic_clarify import run_generic_clarify
from src.methods.eig_ia import run_eig_ia
from src.methods.dpo_question_ranker import run_dpo_question_ranker
from src.utils.io import write_csv, write_jsonl, read_jsonl
from src.utils.logging import get_run_dir, save_json
from src.utils.seeds import set_seeds


METHODS = {
    "direct": run_direct,
    "random_question": run_random_question,
    "generic_clarify": run_generic_clarify,
    "eig_ia": run_eig_ia,
    "dpo_question_ranker": run_dpo_question_ranker,
}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    max_ex = os.environ.get("EIG_IA_MAX_EX")
    if max_ex is not None:
        cfg["dataset"]["max_examples"] = int(max_ex)
    mode = os.environ.get("EIG_IA_MODE")
    if mode:
        cfg["mode"] = mode
    dataset_override = os.environ.get("EIG_IA_DATASET")
    if dataset_override:
        cfg["dataset"]["name"] = dataset_override
    model_type = os.environ.get("EIG_IA_MODEL_TYPE")
    if model_type:
        for key in ["question_model", "answer_model", "scorer_model"]:
            cfg["models"][key]["type"] = model_type
    return cfg


def build_llm(cfg: Dict[str, Any]):
    if cfg["type"] == "api":
        return APILLM(cfg["name_or_path"], cfg.get("decoding", {}))
    return HFLLM(cfg["name_or_path"], cfg.get("decoding", {}))


def get_dataset(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    name = cfg["dataset"]["name"]
    split = cfg["dataset"].get("split", "dev")
    max_examples = int(cfg["dataset"].get("max_examples", 0))
    if name == "art":
        return load_art(split, max_examples)
    return load_ambigqa(split, max_examples)


def download_data(cfg: Dict[str, Any]) -> None:
    _ = get_dataset(cfg)


def preprocess(cfg: Dict[str, Any]) -> None:
    _ = get_dataset(cfg)


def _aggregate_usage(meta: Dict[str, Any]) -> Dict[str, int]:
    usage = {"tokens_in": 0, "tokens_out": 0, "tokens_total": 0}
    if not meta:
        return usage
    for value in meta.values():
        if isinstance(value, dict) and "usage" in value:
            u = value["usage"]
            for k in usage:
                usage[k] += int(u.get(k, 0))
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "usage" in item:
                    u = item["usage"]
                    for k in usage:
                        usage[k] += int(u.get(k, 0))
    return usage


def _usage_by_module(meta: Dict[str, Any]) -> Dict[str, Any]:
    usage = {}
    for key, value in meta.items():
        if isinstance(value, dict) and "usage" in value:
            usage[key] = value["usage"]
        if isinstance(value, list):
            usage[key] = [item.get("usage", {}) for item in value if isinstance(item, dict)]
    return usage


def _latency_by_module(meta: Dict[str, Any]) -> Dict[str, Any]:
    latency = {}
    for key, value in meta.items():
        if isinstance(value, dict) and "latency" in value:
            latency[key] = value["latency"]
        if isinstance(value, list):
            latency[key] = [item.get("latency", 0.0) for item in value if isinstance(item, dict)]
    return latency


def _extract_prompts(meta: Dict[str, Any]) -> Dict[str, Any]:
    prompts = {}
    for key, value in meta.items():
        if isinstance(value, dict) and "prompt" in value:
            prompts[key] = value["prompt"]
        if isinstance(value, list):
            prompts[key] = [item.get("prompt", "") for item in value if isinstance(item, dict)]
    return prompts


def _aggregate_latency(meta: Dict[str, Any]) -> float:
    total = 0.0
    if not meta:
        return total
    for value in meta.values():
        if isinstance(value, dict) and "latency" in value:
            total += float(value.get("latency", 0.0))
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "latency" in item:
                    total += float(item.get("latency", 0.0))
    return total


def _answer_ambigqa(llm_a, rewrite: str, mode: str, answer_set: List[str]) -> str:
    if mode == "oracle" and answer_set:
        return str(answer_set[0])
    prompt = f"Question: {rewrite}\nAnswer:"
    outputs, _ = llm_a.generate(prompt, n=1)
    return outputs[0].strip() if outputs else ""


def _em_f1(pred_answer: str, gold_answers: List[str]) -> Dict[str, float]:
    if not gold_answers:
        return {"em": 0.0, "f1": 0.0}
    norm_pred = normalize_text(pred_answer)
    em = 1.0 if any(norm_pred == normalize_text(g) for g in gold_answers) else 0.0
    f1 = max(f1_score(pred_answer, g) for g in gold_answers)
    return {"em": em, "f1": f1}


def run_method(cfg: Dict[str, Any], method: str) -> List[Dict[str, Any]]:
    data = get_dataset(cfg)
    mode = cfg.get("mode", "oracle")
    llm_q = build_llm(cfg["models"]["question_model"])
    llm_a = build_llm(cfg["models"]["answer_model"])
    llm_scorer = build_llm(cfg["models"]["scorer_model"])

    rows = []
    for idx, ex in enumerate(data):
        example_id = ex.get("id") or str(idx)
        if cfg["dataset"]["name"] == "art":
            gold = int(ex["label"]) - 1
            observation = ex["observation"]
            hypotheses = ex["hypotheses"]
            answer_sets = []
        else:
            gold = 0
            observation = ex["question"]
            hypotheses = ex["rewrites"]
            answer_sets = ex.get("answer_sets", [])

        if method == "direct":
            result = METHODS[method](cfg["dataset"]["name"], ex, llm_scorer)
        elif method in {"random_question"}:
            result = METHODS[method](cfg["dataset"]["name"], ex, llm_q, llm_a, llm_scorer, mode, int(cfg["eig"]["K_questions"]))
        elif method in {"generic_clarify"}:
            result = METHODS[method](cfg["dataset"]["name"], ex, llm_q, llm_a, llm_scorer, mode)
        else:
            result = METHODS[method](
                cfg["dataset"]["name"],
                ex,
                llm_q,
                llm_a,
                llm_scorer,
                mode,
                int(cfg["eig"]["K_questions"]),
                int(cfg["eig"]["M_answers"]),
                cfg["eig"]["estimator"],
                bool(cfg["gating"]["enabled"]),
                float(cfg["gating"]["tau"]),
                float(cfg["gating"]["gamma"]),
            )

        prior_probs = result.get("prior_probs") or []
        posterior_probs = result.get("posterior_probs") or []
        prior_entropy = entropy(prior_probs) if prior_probs else 0.0
        posterior_entropy = entropy(posterior_probs) if posterior_probs else 0.0
        delta_entropy = prior_entropy - posterior_entropy
        confidence = max_prob(posterior_probs) if posterior_probs else 0.0
        correct = 1 if int(result["pred"]) == int(gold) else 0

        usage = _aggregate_usage(result.get("meta", {}))
        latency_total = _aggregate_latency(result.get("meta", {}))
        usage_by_module = _usage_by_module(result.get("meta", {}))
        latency_by_module = _latency_by_module(result.get("meta", {}))
        prompts = _extract_prompts(result.get("meta", {})) if cfg["logging"].get("save_prompts", False) else {}

        pred_answer = ""
        em_f1 = {"em": 0.0, "f1": 0.0}
        if cfg["dataset"]["name"] != "art":
            pred_idx = int(result["pred"])
            pred_idx = max(0, min(pred_idx, len(hypotheses) - 1))
            ans_set_pred = answer_sets[pred_idx] if pred_idx < len(answer_sets) else []
            gold_answers = answer_sets[0] if answer_sets else []
            pred_answer = _answer_ambigqa(llm_a, hypotheses[pred_idx], mode, ans_set_pred)
            em_f1 = _em_f1(pred_answer, gold_answers)

        row = {
            "example_id": example_id,
            "dataset": cfg["dataset"]["name"],
            "method": method,
            "asked": result.get("asked", False),
            "q": result.get("question", ""),
            "a": result.get("answer", ""),
            "prior_probs": prior_probs,
            "posterior_probs": posterior_probs,
            "prior_entropy": prior_entropy,
            "posterior_entropy": posterior_entropy,
            "delta_entropy": delta_entropy,
            "eig_estimate": result.get("eig", 0.0),
            "pred": result["pred"],
            "gold": gold,
            "confidence": confidence,
            "accuracy": correct,
            "tokens_in": usage["tokens_in"],
            "tokens_out": usage["tokens_out"],
            "tokens_total": usage["tokens_total"],
            "latency_total": latency_total,
            "latency_per_module": latency_by_module,
            "tokens_per_module": usage_by_module,
            "final_answer": pred_answer,
            "em": em_f1["em"],
            "f1": em_f1["f1"],
            "prompts": prompts,
            "model_ids": {
                "question": cfg["models"]["question_model"]["name_or_path"],
                "answer": cfg["models"]["answer_model"]["name_or_path"],
                "scorer": cfg["models"]["scorer_model"]["name_or_path"],
            },
            "decoding_params": {
                "question": cfg["models"]["question_model"].get("decoding", {}),
                "answer": cfg["models"]["answer_model"].get("decoding", {}),
                "scorer": cfg["models"]["scorer_model"].get("decoding", {}),
            },
            "seed": cfg["evaluation"]["seed"],
            "observation": observation,
            "hypotheses": hypotheses,
        }
        rows.append(row)

    return rows


def summarize_metrics(rows: List[Dict[str, Any]], run_dir: str) -> List[Dict[str, Any]]:
    metrics_rows = []
    for dataset in sorted(set(r["dataset"] for r in rows)):
        for method in sorted(set(r["method"] for r in rows)):
            subset = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            metrics = compute_metrics(subset)
            ece, reliability = compute_ece(subset)
            metrics["ece"] = ece
            metrics_rows.append({
                "dataset": dataset,
                "method": method,
                **{k: f"{v:.4f}" for k, v in metrics.items()},
            })
            save_json(os.path.join(run_dir, f"reliability_{dataset}_{method}.json"), {"ece": ece, "bins": reliability})
    return metrics_rows


def save_bootstrap(rows: List[Dict[str, Any]], run_dir: str, seed: int, n: int, alpha: float) -> None:
    datasets = sorted(set(r["dataset"] for r in rows))
    for dataset in datasets:
        subset = [r for r in rows if r["dataset"] == dataset]
        eig_rows = [r for r in subset if r["method"] == "eig_ia"]
        if not eig_rows:
            continue
        for method in sorted(set(r["method"] for r in subset)):
            if method == "eig_ia":
                continue
            base_rows = [r for r in subset if r["method"] == method]
            if len(base_rows) != len(eig_rows):
                continue
            acc_eig = [r["accuracy"] for r in eig_rows]
            acc_base = [r["accuracy"] for r in base_rows]
            result = paired_bootstrap(acc_eig, acc_base, n, alpha, seed)
            save_json(
                os.path.join(run_dir, f"bootstrap_{dataset}_eig_vs_{method}.json"),
                result,
            )


def run(cfg_path: str, method: str) -> str:
    cfg = load_config(cfg_path)
    set_seeds(int(cfg["evaluation"]["seed"]))
    run_dir = get_run_dir(cfg["logging"]["out_dir"])
    rows = run_method(cfg, method)
    write_jsonl(os.path.join(run_dir, "per_example.jsonl"), rows)
    metrics_rows = summarize_metrics(rows, run_dir)
    write_csv(os.path.join(run_dir, "metrics.csv"), metrics_rows)
    save_json(os.path.join(run_dir, "config.json"), cfg)
    return run_dir


def run_all(cfg_path: str) -> str:
    cfg = load_config(cfg_path)
    set_seeds(int(cfg["evaluation"]["seed"]))
    run_dir = get_run_dir(cfg["logging"]["out_dir"])
    all_rows = []
    for method in ["direct", "random_question", "generic_clarify", "eig_ia"]:
        all_rows.extend(run_method(cfg, method))
    existing = []
    if os.environ.get("EIG_IA_APPEND") and os.path.exists(os.path.join(run_dir, "per_example.jsonl")):
        existing = read_jsonl(os.path.join(run_dir, "per_example.jsonl"))
    combined = existing + all_rows
    write_jsonl(os.path.join(run_dir, "per_example.jsonl"), combined)
    metrics_rows = summarize_metrics(combined, run_dir)
    write_csv(os.path.join(run_dir, "metrics.csv"), metrics_rows)
    save_json(os.path.join(run_dir, "config.json"), cfg)
    save_bootstrap(combined, run_dir, int(cfg["evaluation"]["seed"]), int(cfg["evaluation"]["bootstrap"]["n"]), float(cfg["evaluation"]["bootstrap"]["alpha"]))
    return run_dir


def sweep(cfg_path: str) -> str:
    cfg = load_config(cfg_path)
    run_dir = get_run_dir(cfg["logging"]["out_dir"])
    all_rows = []
    for estimator in ["entropy", "utility"]:
        for k in [cfg["eig"]["K_questions"], cfg["eig"]["K_questions"] * 2]:
            cfg["eig"]["estimator"] = estimator
            cfg["eig"]["K_questions"] = k
            all_rows.extend(run_method(cfg, "eig_ia"))
    write_jsonl(os.path.join(run_dir, "per_example.jsonl"), all_rows)
    metrics_rows = summarize_metrics(all_rows, run_dir)
    write_csv(os.path.join(run_dir, "metrics.csv"), metrics_rows)
    save_json(os.path.join(run_dir, "config.json"), cfg)
    return run_dir


def make_tables(results_dir: str) -> None:
    from src.viz.latex_tables import make_tables as _make_tables

    _make_tables(results_dir)


def make_plots(results_dir: str) -> None:
    from src.viz.make_plots import make_plots as _make_plots

    _make_plots(results_dir)


def make_human_eval(results_dir: str, out_csv: str) -> None:
    rows = read_jsonl(os.path.join(results_dir, "per_example.jsonl"))
    make_human_eval_csv(rows, out_csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download_data")
    preprocess_p = sub.add_parser("preprocess")
    preprocess_p.add_argument("--dataset", default="art")

    run_p = sub.add_parser("run")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--method", required=True)

    run_all_p = sub.add_parser("run_all")
    run_all_p.add_argument("--config", required=True)

    sweep_p = sub.add_parser("sweep")
    sweep_p.add_argument("--config", required=True)

    tables_p = sub.add_parser("make_tables")
    tables_p.add_argument("--results_dir", required=True)

    plots_p = sub.add_parser("make_plots")
    plots_p.add_argument("--results_dir", required=True)

    human_p = sub.add_parser("make_human_eval")
    human_p.add_argument("--results_dir", required=True)
    human_p.add_argument("--out_csv", required=True)

    args = parser.parse_args()

    if args.command == "download_data":
        cfg = load_config("configs/default.yaml")
        download_data(cfg)
    elif args.command == "preprocess":
        cfg = load_config("configs/default.yaml")
        cfg["dataset"]["name"] = args.dataset
        preprocess(cfg)
    elif args.command == "run":
        run(args.config, args.method)
    elif args.command == "run_all":
        run_all(args.config)
    elif args.command == "sweep":
        sweep(args.config)
    elif args.command == "make_tables":
        make_tables(args.results_dir)
    elif args.command == "make_plots":
        make_plots(args.results_dir)
    elif args.command == "make_human_eval":
        make_human_eval(args.results_dir, args.out_csv)


if __name__ == "__main__":
    main()
