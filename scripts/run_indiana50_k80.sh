#!/usr/bin/env bash
# Complete local Indiana subset workflow; uses existing environments/checkpoint.
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cd "$PROJECT"

MODE="${1:-}"
case "$MODE" in
  ""|--background|--check|--worker) ;;
  *) echo "Usage: bash scripts/run_indiana50_k80.sh [--background|--check]"; exit 2 ;;
esac

INFERENCE_PYTHON="${INFERENCE_PYTHON:-/home/student4/anaconda3/envs/cxr-llava-k80/bin/python}"
LABELER_PYTHON="${LABELER_PYTHON:-/home/student4/anaconda3/envs/chexpert-label/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT/../datasets/NLMCRX}"
TOOLS_ROOT="${TOOLS_ROOT:-$PROJECT/model_cache/chexpert_tools}"
MODEL_DIR="${MODEL_DIR:-$PROJECT/model_cache/cxr-llava-v2-original}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

check_environment() {
  test -x "$INFERENCE_PYTHON"
  test -x "$LABELER_PYTHON"
  test -d "$DATASET_ROOT/NLMCXR_png"
  test -d "$DATASET_ROOT/NLMCXR_reports/ecgen-radiology"
  test -f "$TOOLS_ROOT/chexpert-labeler/label.py"
  test -d "$TOOLS_ROOT/NegBio"
  command -v java >/dev/null
  "$INFERENCE_PYTHON" - "$MODEL_DIR" "$TOOLS_ROOT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path
import torch
import transformers
import pandas
import PIL
import jinja2
from scripts.label_cohort import LABELER_REVISION, NEGBIO_REVISION

assert torch.__version__ == '1.12.1+cu102', torch.__version__
assert transformers.__version__ == '4.36.2', transformers.__version__
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() >= 4, 'Expose at least four GPUs'
for i in range(4):
    free, total = torch.cuda.mem_get_info(i)
    assert free / 1024**3 >= (10 if i == 0 else 8), f'GPU {i} has insufficient free VRAM'
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, free {free/1024**3:.2f} GiB')
model = Path(sys.argv[1])
index = json.loads((model / 'model.safetensors.index.json').read_text())
for shard in set(index['weight_map'].values()):
    assert (model / shard).is_file(), f'Missing checkpoint shard: {shard}'
tools = Path(sys.argv[2])
for name, expected in [('chexpert-labeler', LABELER_REVISION), ('NegBio', NEGBIO_REVISION)]:
    actual = subprocess.check_output(['git', '-C', str(tools/name), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == expected, f'Unexpected {name} revision'
print('Inference environment, GPU, checkpoint and labeler revisions: PASS')
PY
  PYTHONPATH="$TOOLS_ROOT/NegBio" "$LABELER_PYTHON" -c \
    'import nltk, pandas, bllipparser, negbio; print("Legacy labeler imports: PASS")'
}

if [[ "$MODE" == --check ]]; then
  check_environment
  exit 0
fi

if [[ "$MODE" == --worker ]]; then
  RUN_ROOT="${2:?Missing worker run directory}"
  test -d "$RUN_ROOT"
else
  RUN_ROOT="$PROJECT/result/indiana_k80_50_$(date +%Y%m%d_%H%M%S)_$$"
  mkdir "$RUN_ROOT"
fi

if [[ "$MODE" == --background ]]; then
  nohup bash "$SCRIPT_PATH" --worker "$RUN_ROOT" >"$RUN_ROOT/launcher.log" 2>&1 < /dev/null &
  JOB_PID=$!
  echo "$JOB_PID" > "$RUN_ROOT/pid.txt"
  echo "Started PID $JOB_PID"
  echo "Results: $RUN_ROOT"
  echo "Monitor: tail -f '$RUN_ROOT/launcher.log'"
  echo "Final state: cat '$RUN_ROOT/status.txt'"
  exit 0
fi

exec > >(tee -a "$RUN_ROOT/pipeline.log") 2>&1
STAGE=preflight
echo RUNNING > "$RUN_ROOT/status.txt"
finish() {
  CODE=$?
  if [[ "$CODE" -eq 0 ]]; then
    echo SUCCESS > "$RUN_ROOT/status.txt"
  else
    echo "FAILED stage=$STAGE exit=$CODE" > "$RUN_ROOT/status.txt"
    echo "Inspect $RUN_ROOT/pipeline.log and $RUN_ROOT/inference/labeler_logs/"
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Run: $RUN_ROOT"
echo "Start: $(date --iso-8601=seconds)"
check_environment
"$INFERENCE_PYTHON" -m pip freeze > "$RUN_ROOT/inference_pip_freeze.txt"
git rev-parse HEAD > "$RUN_ROOT/git_commit.txt"
git status --porcelain > "$RUN_ROOT/git_status.txt"
git diff --binary > "$RUN_ROOT/tracked_changes.patch"
mkdir "$RUN_ROOT/scripts_snapshot"
cp scripts/{run_indiana50_k80.sh,k80_original.py,prepare_local_indiana.py,label_cohort.py,exp02_reevaluate.py,exp02_provenance.py,manual_experiment.py} "$RUN_ROOT/scripts_snapshot/"
cp requirements-k80.txt "$RUN_ROOT/scripts_snapshot/"

STAGE=prepare_cohort
"$INFERENCE_PYTHON" -m scripts.prepare_local_indiana \
  --dataset-root "$DATASET_ROOT" --output "$RUN_ROOT/cohort_50.csv" --size 50 --seed 42

STAGE=inference
"$INFERENCE_PYTHON" -u -m scripts.k80_original \
  --dataset indiana --cohort "$RUN_ROOT/cohort_50.csv" \
  --image-root "$DATASET_ROOT/NLMCXR_png" --max-cases 50 \
  --model-dir "$MODEL_DIR" --execute --output "$RUN_ROOT/inference"

STAGE=chexpert_labeling
"$INFERENCE_PYTHON" -u -m scripts.label_cohort \
  --source "$RUN_ROOT/inference" --labeler-dir "$TOOLS_ROOT/chexpert-labeler" \
  --negbio-dir "$TOOLS_ROOT/NegBio" --labeler-python "$LABELER_PYTHON"

STAGE=evaluation
"$INFERENCE_PYTHON" -m scripts.exp02_reevaluate \
  --dataset indiana --source "$RUN_ROOT/inference" --output "$RUN_ROOT/metrics" \
  --sizes 50 --bootstrap-iterations 1000 --seed 42

STAGE=summary
"$INFERENCE_PYTHON" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
folder = root / 'metrics/metrics'
summary = json.loads((folder / 'indiana_f1_subset_50_summary.json').read_text())
system = pd.read_csv(folder / 'system_metrics_v2.csv').iloc[0]
metrics = pd.read_csv(folder / 'indiana_f1_subset_50_pathology_metrics_v2.csv')
lines = ['# Indiana 50 — K80 pipeline results', '',
         f"Completed: {summary['cases_successful']}/{summary['cases_requested']}",
         f"Macro precision: {summary['macro_precision']:.4f}",
         f"Macro recall: {summary['macro_recall']:.4f}",
         f"Macro F1: {summary['macro_f1']:.4f}",
         f"Defined F1 pathologies: {summary['defined_f1_pathologies']}/7",
         f"Mean latency: {system['mean_latency_sec']:.2f} sec/image",
         f"Peak allocated VRAM (max device): {system['peak_vram_gb_max']:.2f} GiB", '',
         'Macro means exclude undefined values. Uncertain/unmentioned labels are excluded.',
         'Random 50-report subset; first available XML image. Views are not verified.',
         'Research evaluation with metrics; not original paper benchmark reproduction.', '',
         '```text', metrics.to_string(index=False), '```', '',
         'Detailed metrics and bootstrap CIs: metrics/metrics/',
         'CheXpert logs/provenance: inference/labeler_logs/']
(root / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
echo "End: $(date --iso-8601=seconds)"
echo "Complete. Results: $RUN_ROOT/RESULTS.md"
