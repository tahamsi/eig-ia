# EIG-Guided Interactive Abduction (EIG-IA)

Complete, reproducible research codebase for EIG-Guided Interactive Abduction (EIG-IA). The system decides whether to ask one clarifying question, selects the most informative question using expected information gain (EIG), and then answers with reduced uncertainty.

Python requirement: 3.10.12.

## Problem

Interactive abduction requires choosing the most plausible hypothesis given an observation. In many cases, a single clarifying question can sharply reduce ambiguity. EIG-IA formalizes this decision by estimating the expected information gain for candidate questions and using a gate to decide whether to ask or answer directly.

Datasets covered:
- ART / alphaNLI (two-choice abduction)
- AmbigQA / AmbigNQ (ambiguous QA with disambiguated rewrites)

## Method Summary

For each example:
1) Score hypotheses to obtain a prior P(H | O).
2) Generate K candidate clarifying questions.
3) Estimate EIG(q_k) for each question using sampled answers.
4) Select q* = argmax EIG(q_k).
5) Gate: ask iff max_h P(h|O) < tau or EIG(q*) > gamma.
6) If asked, obtain answer a* (oracle or simulator) and update posterior.
7) Predict the final hypothesis from the posterior (or prior if not asked).

EIG estimators supported:
- Entropy: H(P(H|O)) - E_a[H(P(H|O,q,a))]
- Utility: E_a[U(P(H|O,q,a))] - U(P(H|O)), with U(P)=max_h P(h)

## Repository Layout

```
eig_ia/
  README.md
  LICENSE
  CITATION.cff
  pyproject.toml
  requirements.txt
  main.py
  Makefile
  configs/
  scripts/
  artifacts/
  src/
```

## Requirements

- Python 3.10.12
- CPU or GPU (CUDA optional)
- Packages listed in `requirements.txt`

## Setup

From the repo root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r eig_ia/requirements.txt
```

Then run commands from inside `eig_ia/`:

```bash
cd eig_ia
```

## Quickstart (small subset)

```bash
bash scripts/reproduce.sh
```

Designed to run a small subset and produce:
- `outputs/<timestamp>/metrics.csv`
- `outputs/<timestamp>/tables/*.tex`
- `outputs/<timestamp>/figures/*.png` and `.pdf`
- `artifacts/eig_ia_acl_artifacts_<timestamp>.zip`

Expected runtime: designed to be small for quick reproduction.

## Core Commands

All commands below are run from `eig_ia/`.

```bash
# download data
python main.py download_data

# preprocess (lightweight loader check)
python main.py preprocess --dataset art
python main.py preprocess --dataset ambigqa

# single method run
python main.py run --config configs/art_open.yaml --method eig_ia
python main.py run --config configs/ambigqa_open.yaml --method eig_ia

# run baselines + EIG-IA (dataset set by config or env override)
python main.py run_all --config configs/default.yaml

# sweep basic ablations
python main.py sweep --config configs/default.yaml

# tables and figures from an existing results directory
python main.py make_tables --results_dir outputs/<timestamp>
python main.py make_plots --results_dir outputs/<timestamp>

# human evaluation CSV (file path, not a directory)
python main.py make_human_eval --results_dir outputs/<timestamp> --out_csv human_eval.csv
```

## Full Reproduction Script

```bash
bash scripts/reproduce.sh
```

Optional flags:
- `--full` runs full splits
- `--oracle_only` or `--simulator_only`
- `--open_only` or `--api_only`

This script runs ART and AmbigQA into the same timestamped output directory.

## Required Steps for Experiments

Minimal end-to-end run:
```bash
python main.py download_data
python main.py run_all --config configs/default.yaml
python main.py make_tables --results_dir outputs/<timestamp>
python main.py make_plots --results_dir outputs/<timestamp>
```

Run ART and AmbigQA explicitly in one output directory:
```bash
export EIG_IA_RUN_DIR=outputs/$(date +%Y%m%d_%H%M%S)
export EIG_IA_DATASET=art
python main.py run_all --config configs/default.yaml
export EIG_IA_DATASET=ambigqa
export EIG_IA_APPEND=1
python main.py run_all --config configs/default.yaml
python main.py make_tables --results_dir $EIG_IA_RUN_DIR
python main.py make_plots --results_dir $EIG_IA_RUN_DIR
```

## Outputs

Expected output structure:
```
outputs/<timestamp>/
  per_example.jsonl
  metrics.csv
  tables/
  figures/
  env.json
  env.txt
```

Per-example JSONL includes:
- observation, hypotheses, question/answer, prior/posterior, entropy, EIG
- prediction, gold, confidence, EM/F1 (AmbigQA)
- tokens/latency per module, model IDs, decoding params, seed

## Configuration

Edit YAML files in `configs/` to change datasets, models, EIG settings, and gating. Default uses a small subset with open models and runs without API keys.

Key fields:
- `dataset.name`: `art` or `ambigqa`
- `mode`: `oracle` or `simulator`
- `models.*`: `type` (`hf` or `api`), `name_or_path`, decoding params
- `eig`: `K_questions`, `M_answers`, `estimator`
- `gating`: `enabled`, `tau`, `gamma`

Environment overrides:
- `EIG_IA_MAX_EX` to cap examples
- `EIG_IA_MODE` to override `oracle` or `simulator`
- `EIG_IA_MODEL_TYPE` to force `hf` or `api`
- `EIG_IA_DATASET` to switch between `art` and `ambigqa`
- `EIG_IA_RUN_DIR` to force a specific output directory
- `EIG_IA_APPEND=1` to append runs into the same output directory

## AmbigQA Evaluation

For AmbigQA, the oracle target is taken as the first rewrite. Final answers are generated with the answer model (or from the oracle answer set when available) and scored with exact match + token F1 against the gold answers for the target rewrite.

## API Models

If you want to use API models, set environment variables (e.g., `OPENAI_API_KEY`). If no keys are set, API configs will fail fast with a clear error.

## Reproducibility

All runs log seeds, prompts (optional), decoding params, model IDs, timestamps, token counts, and latency. To capture environment details:

```bash
python scripts/env_report.py --out_dir outputs/<timestamp>
```

Determinism notes:
- Random sampling and GPU kernels can introduce nondeterminism.
- Set `evaluation.seed` in config and consider CPU-only for stricter determinism.

## Artifact Pack

```bash
bash scripts/make_artifacts.sh outputs/<timestamp>
```

This produces:
- `artifacts/eig_ia_acl_artifacts_<timestamp>.zip`
- `artifacts/eig_ia_acl_artifacts_<timestamp>.sha256`

## Human Evaluation Agreement

Add rater columns to the human eval CSV and run:

```bash
python -m src.eval.stats --csv human_eval.csv --columns rater1_helpfulness,rater2_helpfulness
```
