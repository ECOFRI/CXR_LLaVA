#!/usr/bin/env bash
# Frozen NLM-view Indiana cohort: resume gate, full inference, official labels, metrics.
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cd "$PROJECT"

MODE="${1:---check}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  DEFAULT_INFERENCE_PYTHON="$CONDA_PREFIX/bin/python"
  DEFAULT_LABELER_PYTHON="$(dirname "$CONDA_PREFIX")/chexpert-label/bin/python"
else
  DEFAULT_INFERENCE_PYTHON=""
  DEFAULT_LABELER_PYTHON=""
fi
INFERENCE_PYTHON="${INFERENCE_PYTHON:-$DEFAULT_INFERENCE_PYTHON}"
LABELER_PYTHON="${LABELER_PYTHON:-$DEFAULT_LABELER_PYTHON}"
TOOLS_ROOT="${TOOLS_ROOT:-$PROJECT/model_cache/chexpert_tools}"
MODEL_DIR="${MODEL_DIR:-$PROJECT/model_cache/cxr-llava-v2-original}"
COHORT="${COHORT:-$PROJECT/result/indiana_full_k80_20261004_inventory02/nlm_review_01/selection/cohort.csv}"
GT_DIR="${GT_DIR:-$PROJECT/result/indiana_full_k80_20261004_inventory02/nlm_review_01/gt_labeling}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

preflight() {
  if [[ -z "$INFERENCE_PYTHON" || -z "$LABELER_PYTHON" ]]; then
    echo 'Activate cxr-llava-k80 or set INFERENCE_PYTHON and LABELER_PYTHON' >&2
    return 2
  fi
  test -x "$INFERENCE_PYTHON"
  test -x "$LABELER_PYTHON"
  test -f "$COHORT"
  test -f "$GT_DIR/provenance.json"
  test -d "$MODEL_DIR"
  test -f "$TOOLS_ROOT/chexpert-labeler/label.py"
  test -d "$TOOLS_ROOT/NegBio"
  command -v java >/dev/null
  "$INFERENCE_PYTHON" - "$COHORT" "$GT_DIR" "$MODEL_DIR" "$TOOLS_ROOT" <<'PY'
import json,subprocess,sys
from pathlib import Path
import torch,transformers
from scripts.exp02_provenance import MODEL_REVISION
from scripts.label_cohort import LABELER_REVISION,NEGBIO_REVISION
from scripts.manual_experiment import sha256
cohort,gt,model,tools=map(Path,sys.argv[1:])
assert torch.__version__=='1.12.1+cu102',torch.__version__
assert transformers.__version__=='4.36.2',transformers.__version__
assert torch.cuda.is_available() and torch.cuda.device_count()>=4
for i in range(4):
 free,_=torch.cuda.mem_get_info(i)
 assert free/1024**3 >= (10 if i==0 else 8),f'GPU {i} free {free/1024**3:.2f} GiB'
 print(f'GPU {i}: {torch.cuda.get_device_name(i)}, free {free/1024**3:.2f} GiB')
config=json.loads((cohort.parent/'cohort_config.json').read_text())
provenance=json.loads((gt/'provenance.json').read_text())
assert sha256(cohort)==config['cohort_sha256']==provenance['cohort_sha256']
assert provenance['sample_gate']=='PASS' and provenance['alignment']=='PASS'
index=json.loads((model/'model.safetensors.index.json').read_text())
assert all((model/shard).is_file() for shard in set(index['weight_map'].values()))
for name,revision in [('chexpert-labeler',LABELER_REVISION),('NegBio',NEGBIO_REVISION)]:
 actual=subprocess.check_output(['git','-C',str(tools/name),'rev-parse','HEAD'],text=True).strip()
 assert actual==revision,(name,actual)
assert MODEL_REVISION=='b2224786bb90d54b1e1291171866706cfbb44e2b'
print('Pinned cohort, GT labels, environment, GPUs and checkpoint: PASS')
PY
}

case "$MODE" in
  --check)
    preflight
    exit 0
    ;;
  --background)
    RUN_ROOT="${RUN_ROOT:-$PROJECT/result/indiana_full_k80_$(date +%Y%m%d_%H%M%S)_$$}"
    mkdir "$RUN_ROOT"
    nohup bash "$SCRIPT_PATH" --worker "$RUN_ROOT" > "$RUN_ROOT/launcher.log" 2>&1 < /dev/null &
    echo "$!" > "$RUN_ROOT/pid.txt"
    echo "Started PID $!"
    echo "Results: $RUN_ROOT"
    echo "Monitor: tail -f '$RUN_ROOT/launcher.log'"
    echo "Final state: cat '$RUN_ROOT/status.txt'"
    exit 0
    ;;
  --worker|--resume)
    RUN_ROOT="${2:?Pass an existing run directory}"
    RUN_ROOT="$(readlink -f "$RUN_ROOT")"
    test -d "$RUN_ROOT"
    ;;
  *)
    echo "Usage: bash scripts/run_indiana_full_k80.sh [--check|--background|--worker RUN_DIR|--resume RUN_DIR]" >&2
    exit 2
    ;;
esac

exec > >(tee -a "$RUN_ROOT/pipeline.log") 2>&1
STAGE=preflight
echo RUNNING > "$RUN_ROOT/status.txt"
finish() {
  CODE=$?
  if [[ "$CODE" -ne 0 ]]; then
    echo "FAILED stage=$STAGE exit=$CODE" > "$RUN_ROOT/status.txt"
    echo "Inspect $RUN_ROOT/pipeline.log and stage logs"
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Run: $RUN_ROOT"
echo "Start: $(date --iso-8601=seconds)"
preflight
if [[ "$MODE" == --worker ]]; then
  STAGE=snapshot
  mkdir "$RUN_ROOT/snapshots"
  git rev-parse HEAD > "$RUN_ROOT/snapshots/git_commit.txt"
  git status --porcelain > "$RUN_ROOT/snapshots/git_status.txt"
  git diff --binary > "$RUN_ROOT/snapshots/tracked_changes.patch"
  "$INFERENCE_PYTHON" -m pip freeze > "$RUN_ROOT/snapshots/inference_pip_freeze.txt"
  "$LABELER_PYTHON" -m pip freeze > "$RUN_ROOT/snapshots/labeler_pip_freeze.txt"
fi

if [[ ! -d "$RUN_ROOT/gate" ]]; then
  STAGE=gate_first_case
  "$INFERENCE_PYTHON" -u -m scripts.k80_full_inference \
    --cohort "$COHORT" --model-dir "$MODEL_DIR" --max-cases 10 \
    --stop-after 1 --execute --output "$RUN_ROOT/gate"
  test "$(cat "$RUN_ROOT/gate/status.txt")" = PARTIAL
fi
FIRST_HASH="$(sha256sum "$RUN_ROOT/gate/cases/00000.json" | cut -d' ' -f1)"

if [[ "$(cat "$RUN_ROOT/gate/status.txt")" != COMPLETED ]]; then
  STAGE=gate_resume
  "$INFERENCE_PYTHON" -u -m scripts.k80_full_inference \
    --cohort "$COHORT" --model-dir "$MODEL_DIR" --max-cases 10 \
    --execute --resume --output "$RUN_ROOT/gate"
fi
test "$(cat "$RUN_ROOT/gate/status.txt")" = COMPLETED
test "$FIRST_HASH" = "$(sha256sum "$RUN_ROOT/gate/cases/00000.json" | cut -d' ' -f1)"
if [[ ! -f "$RUN_ROOT/gate/config_mismatch_gate.log" ]]; then
  STAGE=gate_config_mismatch
  if "$INFERENCE_PYTHON" -m scripts.k80_full_inference \
      --cohort "$COHORT" --model-dir "$MODEL_DIR" --max-cases 10 \
      --seed 43 --execute --resume --output "$RUN_ROOT/gate" \
      > "$RUN_ROOT/gate/config_mismatch_gate.log" 2>&1; then
    echo 'Resume config mismatch was unexpectedly accepted' >&2
    exit 1
  fi
fi
grep -q 'Resume config differs at seed' "$RUN_ROOT/gate/config_mismatch_gate.log"
"$INFERENCE_PYTHON" - "$RUN_ROOT/gate" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
counts=json.loads((root/'progress.json').read_text())
assert counts['requested']==10 and counts['success']==10 and counts['failed']==0
assert len(list((root/'cases').glob('*.json')))==10
print('Gate 10/10 and resume without duplicate first case: PASS')
PY

STAGE=full_inference
INFERENCE_ARGS=(--cohort "$COHORT" --model-dir "$MODEL_DIR" --execute --output "$RUN_ROOT/inference")
if [[ -d "$RUN_ROOT/inference" ]]; then
  INFERENCE_ARGS+=(--resume)
fi
"$INFERENCE_PYTHON" -u -m scripts.k80_full_inference "${INFERENCE_ARGS[@]}"
test "$(cat "$RUN_ROOT/inference/status.txt")" = COMPLETED

if [[ ! -d "$RUN_ROOT/metrics" ]]; then
  STAGE=generated_labeling_and_metrics
  "$INFERENCE_PYTHON" -u -m scripts.evaluate_indiana_full \
    --cohort "$COHORT" --gt-labeling "$GT_DIR" --inference "$RUN_ROOT/inference" \
    --labeler-dir "$TOOLS_ROOT/chexpert-labeler" --negbio-dir "$TOOLS_ROOT/NegBio" \
    --labeler-python "$LABELER_PYTHON" --bootstrap-iterations 1000 --seed 42 \
    --output "$RUN_ROOT/metrics"
fi
test -f "$RUN_ROOT/metrics/summary.json" || { echo 'Metrics output is incomplete; inspect stage logs before retry' >&2; exit 1; }

STAGE=summary
"$INFERENCE_PYTHON" - "$RUN_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); summary=json.loads((root/'metrics/summary.json').read_text())
lines=['# Indiana full cohort — K80 results','',
       'Classification: reconstructed cohort with NLM image-level view labels; not exact Table 4 reproduction.',
       f"Completed: {summary['generated_success']}/{summary['cohort_n']}",
       f"Macro F1: {summary['macro_f1_defined_only']} ({summary['f1_pathologies_defined']}/7 defined)",
       f"Mean latency: {summary['mean_latency_sec_per_success']} sec/success",
       f"Peak VRAM max device: {summary['peak_vram_gb_max_device']} GiB",'',
       'Detailed per-pathology metrics, CI, coverage and omissions: metrics/pathology_metrics.csv',
       'Full provenance: inference/run_config.json, metrics/summary.json, GT labeling provenance.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
echo SUCCESS > "$RUN_ROOT/status.txt"
STAGE=report
if [[ ! -f "$RUN_ROOT/report_path.txt" ]]; then
  REPORT_PATH="$PROJECT/reports/$(date +%Y-%m-%d)_k80_indiana_full_$(basename "$RUN_ROOT").md"
  "$INFERENCE_PYTHON" -m scripts.write_indiana_full_report \
    --run "$RUN_ROOT" \
    --selection "$(dirname "$(dirname "$COHORT")")" \
    --gt-labeling "$GT_DIR" \
    --output "$REPORT_PATH"
  echo "$REPORT_PATH" > "$RUN_ROOT/report_path.txt"
fi
echo "End: $(date --iso-8601=seconds)"
