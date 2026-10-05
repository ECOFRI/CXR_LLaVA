#!/usr/bin/env bash
# One-shot CheXpert Table 3 matched-cohort evaluation with resume and status.
set -Eeuo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
PYTHON="${INFERENCE_PYTHON:-${CONDA_PREFIX:-}/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo 'Activate cxr-llava-k80 or set INFERENCE_PYTHON' >&2
  exit 2
fi
RUN_DIR="${1:?Usage: bash scripts/run_chexpert_table3_k80.sh RUN_DIR}"
mkdir -p "$RUN_DIR"
RUN_DIR="$(readlink -f "$RUN_DIR")"
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1
STAGE=preflight
echo RUNNING > "$RUN_DIR/status.txt"
finish() {
  code=$?
  if [[ "$code" -ne 0 ]]; then
    echo "FAILED stage=$STAGE exit=$code" > "$RUN_DIR/status.txt"
  fi
}
trap finish EXIT

"$PYTHON" -m scripts.chexpert_table3_k80 --check
STAGE=inference
args=(--run --output "$RUN_DIR/inference")
if [[ -f "$RUN_DIR/inference/run_config.json" ]]; then
  args+=(--resume)
fi
"$PYTHON" -u -m scripts.chexpert_table3_k80 "${args[@]}"
STAGE=evaluation
"$PYTHON" -u -m scripts.chexpert_table3_k80 \
  --evaluate --output "$RUN_DIR/inference"
echo SUCCESS > "$RUN_DIR/status.txt"
echo "CheXpert Table 3 local evaluation completed: $RUN_DIR"
