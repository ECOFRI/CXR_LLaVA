#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
umask 077
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
mkdir -p research/rrg_f1/private
exec python3 -m research.rrg_f1.cohort --resume
