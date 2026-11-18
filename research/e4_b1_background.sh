#!/usr/bin/env bash
# Resume the frozen E4-B1 full inference grid independently of the Codex session.
set -u

ROOT=/storage/student4/hanhpm/CXR_LLaVA_Improvement
OUT="$ROOT/research/runs/20261017_E4_B1_evidence_format_gate_v1"
CACHE=/storage/student4/hanhpm/.e4b1_cache
PYTHON=/home/student4/anaconda3/envs/cxr-llava-k80/bin/python

cd "$ROOT" || exit 1
export TMPDIR="$CACHE/tmp"
export HF_HOME="$CACHE/hf"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=GPU-27c3b5ab-b82d-579b-14da-ceece5dfaa16,GPU-64ba6275-3071-8d64-069d-3c23552c6764,GPU-7c841195-e5a5-796c-f50e-06b4028639ed,GPU-9bdf39c7-917e-ba7f-73e5-ee97a9e87083

mkdir -p "$CACHE/tmp" "$CACHE/hf" "$CACHE/torch" "$CACHE/pip" "$CACHE/xdg"
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '{"status":"RUNNING","started_utc":"%s","pid":%s}\n' "$started_utc" "$$" > "$OUT/E4_B1_background_status.json"
printf 'E4-B1 detached resume started %s, PID %s\n' "$started_utc" "$$"

"$PYTHON" research/e4_b1_gpu.py full
code=$?

finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [ "$code" -eq 0 ]; then
  final_status=COMPLETE
else
  final_status=FAILED
fi
printf '{"status":"%s","started_utc":"%s","finished_utc":"%s","pid":%s,"exit_code":%s}\n' \
  "$final_status" "$started_utc" "$finished_utc" "$$" "$code" > "$OUT/E4_B1_background_status.json"
printf 'E4-B1 detached resume %s %s, exit %s\n' "$final_status" "$finished_utc" "$code"
exit "$code"
