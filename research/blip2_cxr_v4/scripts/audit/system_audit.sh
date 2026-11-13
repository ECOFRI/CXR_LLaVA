#!/usr/bin/env bash
set -u
pwd
whoami
hostname
uname -a
id -Gn
timeout 20 git rev-parse HEAD
timeout 20 git branch --show-current
timeout 20 git remote
timeout 60 df -h "$PWD/research/blip2_cxr_v4" "$HOME" /storage/shared
for c in sbatch squeue tmux nvcc conda; do command -v "$c" || true; done
timeout 60 tmux list-sessions 2>/dev/null | wc -l
timeout 60 nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv 2>&1
timeout 60 nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap,driver_version --format=csv 2>&1 || true
timeout 60 nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader 2>&1
