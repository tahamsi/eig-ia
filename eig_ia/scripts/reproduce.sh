#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FULL=0
ORACLE_ONLY=0
SIMULATOR_ONLY=0
OPEN_ONLY=0
API_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --oracle_only) ORACLE_ONLY=1 ;;
    --simulator_only) SIMULATOR_ONLY=1 ;;
    --open_only) OPEN_ONLY=1 ;;
    --api_only) API_ONLY=1 ;;
  esac
done

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [[ $FULL -eq 1 ]]; then
  export EIG_IA_MAX_EX=0
else
  export EIG_IA_MAX_EX=200
fi

if [[ $ORACLE_ONLY -eq 1 ]]; then
  export EIG_IA_MODE=oracle
fi
if [[ $SIMULATOR_ONLY -eq 1 ]]; then
  export EIG_IA_MODE=simulator
fi
if [[ $OPEN_ONLY -eq 1 ]]; then
  export EIG_IA_MODEL_TYPE=hf
fi
if [[ $API_ONLY -eq 1 ]]; then
  export EIG_IA_MODEL_TYPE=api
fi

python main.py download_data

RUN_DIR="outputs/$(date +%Y%m%d_%H%M%S)"
export EIG_IA_RUN_DIR="$RUN_DIR"
mkdir -p "$RUN_DIR"
python scripts/env_report.py --out_dir "$RUN_DIR"

if [[ $API_ONLY -eq 1 ]]; then
  CONFIG_ART="configs/art_gpt4.yaml"
  CONFIG_AMBIG="configs/ambigqa_gpt4.yaml"
else
  CONFIG_ART="configs/default.yaml"
  CONFIG_AMBIG="configs/default.yaml"
fi

export EIG_IA_DATASET=art
unset EIG_IA_APPEND
python main.py run_all --config "$CONFIG_ART"

export EIG_IA_DATASET=ambigqa
export EIG_IA_APPEND=1
python main.py run_all --config "$CONFIG_AMBIG"

python main.py make_tables --results_dir "$RUN_DIR"
python main.py make_plots --results_dir "$RUN_DIR"

bash scripts/make_artifacts.sh "$RUN_DIR"

mkdir -p artifacts
zip -r artifacts/paper_artifacts.zip "$RUN_DIR" >/dev/null

echo "Done. Results in $RUN_DIR"
