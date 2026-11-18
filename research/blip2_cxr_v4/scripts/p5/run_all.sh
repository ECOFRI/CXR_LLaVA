#!/usr/bin/env bash
set -euo pipefail
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --resume ]]; }; then
  echo 'usage: run_all.sh [--resume]' >&2
  exit 2
fi
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
run_dir="$repo_root/research/blip2_cxr_v4/runs/P5_qformer_20261025"
mkdir -p "$run_dir"
cd "$repo_root"
p5_python=${P5_PYTHON:-/home/student4/anaconda3/envs/cxr-llava-k80/bin/python}
session_name=vlm_p5_20261025
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "P5 supervisor tmux session already exists: $session_name" >&2
  exit 1
fi
resume_arg=${1:-}
tmux new-session -d -s "$session_name" -c "$repo_root" \
  "exec $p5_python research/blip2_cxr_v4/scripts/p5/run_supervisor.py $resume_arg >>'$run_dir/supervisor.stdout.log' 2>>'$run_dir/supervisor.stderr.log'"
tmux display-message -p -t "$session_name" '#{pane_pid}'
