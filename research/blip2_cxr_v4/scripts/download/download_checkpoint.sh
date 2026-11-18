#!/usr/bin/env bash
set -euo pipefail
VR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TMPDIR="$VR/.cache/tmp" HF_HOME="$VR/.cache/hf" TORCH_HOME="$VR/.cache/torch"
export PIP_CACHE_DIR="$VR/.cache/pip" XDG_CACHE_HOME="$VR/.cache" NLTK_DATA="$VR/.cache/nltk"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$NLTK_DATA"
exec /home/student4/anaconda3/envs/cxr-llava-k80/bin/python "$VR/scripts/download/download_checkpoint.py" "$@"
