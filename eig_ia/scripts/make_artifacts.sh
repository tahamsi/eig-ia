#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${1:-}"
if [[ -z "$RESULTS_DIR" ]]; then
  RESULTS_DIR=$(ls -td outputs/* | head -n 1)
fi

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="artifacts"
mkdir -p "$OUT_DIR"

ART_DIR="$OUT_DIR/eig_ia_acl_artifacts_$TS"
mkdir -p "$ART_DIR"

cp -r "$RESULTS_DIR"/tables "$ART_DIR/" 2>/dev/null || true
cp -r "$RESULTS_DIR"/figures "$ART_DIR/" 2>/dev/null || true
cp "$RESULTS_DIR"/metrics.csv "$ART_DIR/outputs_summary.csv" 2>/dev/null || true
cp "$RESULTS_DIR"/env.txt "$ART_DIR/" 2>/dev/null || true
cp "$RESULTS_DIR"/env.json "$ART_DIR/" 2>/dev/null || true
cp "$RESULTS_DIR"/config.json "$ART_DIR/" 2>/dev/null || true

mkdir -p "$ART_DIR/configs_used"
cp configs/*.yaml "$ART_DIR/configs_used/" 2>/dev/null || true

head -n 50 "$RESULTS_DIR"/per_example.jsonl > "$ART_DIR/per_example_sample.jsonl" 2>/dev/null || true

cat <<EOT > "$ART_DIR/README_ARTIFACTS.md"
Artifact bundle for EIG-IA.
Includes tables, figures, configs, and summary outputs.
EOT

ZIP_PATH="$OUT_DIR/eig_ia_acl_artifacts_$TS.zip"
(cd "$OUT_DIR" && zip -r "$(basename "$ZIP_PATH")" "eig_ia_acl_artifacts_$TS" >/dev/null)
sha256sum "$ZIP_PATH" > "$OUT_DIR/eig_ia_acl_artifacts_$TS.sha256"

echo "Artifacts written to $ZIP_PATH"
