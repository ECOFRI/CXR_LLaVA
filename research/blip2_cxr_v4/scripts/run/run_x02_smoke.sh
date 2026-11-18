#!/usr/bin/env bash
set -euo pipefail
VR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$VR/configs/experiments/X02_smoke.yaml"
OUT=""
GPU=0
DRY=0
RESUME=0
while (($#)); do
  case "$1" in
    --help) echo 'Usage: run_x02_smoke.sh [--config PATH] [--output-dir PATH] [--gpu N] [--dry-run] [--resume]'; exit 0;;
    --config) CONFIG="$2"; shift 2;;
    --output-dir) OUT="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --resume) RESUME=1; shift;;
    *) echo 'Unknown argument' >&2; exit 2;;
  esac
done
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'GPU must be an integer' >&2; exit 2; }
if [[ -z "$OUT" ]]; then OUT="$VR/runs/X02_smoke_$(date -u +%Y%m%dT%H%M%SZ)"; fi
[[ "$OUT" == "$VR"/runs/* ]] || { echo 'Output must be under V4 runs' >&2; exit 2; }
mkdir -p "$OUT" "$VR/.cache/tmp" "$VR/.cache/hf" "$VR/.cache/torch" "$VR/.cache/pip" "$VR/.cache/nltk"
export TMPDIR="$VR/.cache/tmp" HF_HOME="$VR/.cache/hf" TORCH_HOME="$VR/.cache/torch"
export PIP_CACHE_DIR="$VR/.cache/pip" XDG_CACHE_HOME="$VR/.cache" NLTK_DATA="$VR/.cache/nltk"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU"
PYTHON=/home/student4/anaconda3/envs/cxr-llava-k80/bin/python
EVAL_PYTHON=/home/student4/anaconda3/envs/chexpert-label/bin/python
EVAL_CONFIG="$VR/configs/eval/x02_smoke.yaml"
printf '%s\n' "$$" > "$OUT/pid"
: > "$OUT/stdout.log"; : > "$OUT/stderr.log"
if ((DRY)); then
  timeout 60 "$PYTHON" "$VR/scripts/experiments/infer_blip2.py" --config "$CONFIG" --output-dir "$OUT" --gpu "$GPU" --start-index 0 --max-cases 1 --dry-run > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  timeout 60 "$EVAL_PYTHON" "$VR/scripts/eval/evaluate_reports.py" --config "$EVAL_CONFIG" --predictions "$OUT/predictions.jsonl" --output-dir "$OUT/evaluation" --dry-run >> "$OUT/stdout.log" 2>> "$OUT/stderr.log"
  printf '{"status":"DRY_RUN","gpu_job_started":false,"fp16_probe":{"status":"NOT_RUN_DRY_RUN"}}\n' > "$OUT/status.json"
  cat "$OUT/stdout.log"
  exit 0
fi
EXTRA=()
if ((RESUME)); then EXTRA+=(--resume); fi
if ! timeout 1800 "$PYTHON" "$VR/scripts/experiments/infer_blip2.py" --config "$CONFIG" --output-dir "$OUT" --gpu "$GPU" --start-index 0 --max-cases 1 "${EXTRA[@]}" >> "$OUT/stdout.log" 2>> "$OUT/stderr.log"; then
  echo 'Stage 1 blocked; see status.json' >&2; exit 2
fi
if ! timeout 1800 "$PYTHON" "$VR/scripts/experiments/infer_blip2.py" --config "$CONFIG" --output-dir "$OUT" --gpu "$GPU" --start-index 1 --max-cases 9 --resume >> "$OUT/stdout.log" 2>> "$OUT/stderr.log"; then
  echo 'Stage 2 blocked; see status.json' >&2; exit 2
fi
if ! timeout 1800 "$EVAL_PYTHON" "$VR/scripts/eval/evaluate_reports.py" --config "$EVAL_CONFIG" --predictions "$OUT/predictions.jsonl" --output-dir "$OUT/evaluation" >> "$OUT/stdout.log" 2>> "$OUT/stderr.log"; then
  "$PYTHON" -c 'import json,sys; p=sys.argv[1]; x=json.load(open(p)); x.update(status="FAIL",reason="evaluation_failed"); open(p,"w").write(json.dumps(x,indent=2)+"\n")' "$OUT/status.json"
  echo 'X02 evaluation failed; see logs' >&2; exit 2
fi
"$PYTHON" -c 'import json,sys; p,m=sys.argv[1:]; x=json.load(open(p)); y=json.load(open(m)); clinical=y["clinical_historical"]["micro_f1"]["value"]; x.update(status="PASS" if clinical is not None else "PARTIAL",generation_status="PASS",n_valid=10,evaluation="PASS",evaluation_metrics="evaluation/metrics.json",clinical_f1="available" if clinical is not None else "N/A"); open(p,"w").write(json.dumps(x,indent=2)+"\n")' "$OUT/status.json" "$OUT/evaluation/metrics.json"
echo 'X02 generation PASS; exploratory evaluation recorded'
