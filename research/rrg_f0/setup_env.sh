#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CONDA_PKGS_DIRS="$PWD/model_cache/rrg_f0/conda_pkgs"
export PIP_CACHE_DIR="$PWD/model_cache/rrg_f0/install_cache/pip"
export TMPDIR="$PWD/model_cache/rrg_f0/install_cache/tmp"
mkdir -p "$TMPDIR"
conda_bin="${CONDA_EXE:-conda}"
env_prefix="$PWD/research/envs/rrg-qwen"
if [[ ! -x "$env_prefix/bin/python" ]]; then
    "$conda_bin" create -p "$env_prefix" python=3.11 pip -y
fi
"$env_prefix/bin/python" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$env_prefix/bin/python" -m pip install transformers==4.51.3 peft==0.15.2 accelerate==1.6.0 'numpy<2' pillow ftfy regex huggingface-hub==0.30.2
"$env_prefix/bin/python" -m pip freeze > research/rrg_f0/env.lock.txt
"$env_prefix/bin/python" -m pip check
