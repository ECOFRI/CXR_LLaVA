# P0 system inventory (2026-10-22)

Verified: user student4; host ictserver6; Linux 4.19.0-14-amd64; group student4. Project root `/storage/student4/hanhpm/CXR_LLaVA_Improvement`. `df -h` reports /storage 813T total, 1.8T available, 100% used; /home 738G total, 0 available, 100% used. Scheduler commands sbatch/squeue NOT_FOUND; tmux session count 13. nvcc `/usr/local/cuda-10.2/bin/nvcc`.

GPU: 8 Tesla K80 devices, 11441 MiB each, 0 MiB used at query, driver 440.33.01. Running compute process owner/memory: none returned by `nvidia-smi --query-compute-apps`; compute capability query unsupported by this nvidia-smi, K80 architecture 3.7 not runtime verified.

Conda: `/home/student4/anaconda3/bin/conda env list`; most `conda run -p` probes failed `NoSpaceLeftError`; the project-local `cxr-clear-cpu` prefix returned package data. Direct env interpreters were probed under 60s timeout for the others; see `manifests/conda_envs.csv`. No packages installed. Repository Git SHA/remotes/branch: UNKNOWN (not required by WO and no Git writes).

Repository read-only Git check (`timeout 20 git rev-parse HEAD`, `git branch --show-current`, `git remote`, `git status --short`): SHA `2127c60f296691daa85c2e51a3c2044d9fae2d38`, branch `research/e4-b1-evidence-format-gate`, remote name `origin` (URL withheld). Existing modified/deleted files were observed and left untouched.

Code/weights: `model_cache/cxr-llava-v2-original` contains three safetensors shards (14G directory); `cxr-llava-v2-k80-fp32` contains symlinks to those shards (2.5M apparent directory size). CLEAR upstream repo SHA `57d2eae57ddabf8e56c8013ec675336d71b3d8e7`; Hugging Face CLEAR cache main ref `142f169c8b006b0f28cf55c6a237c3003db5ed8f`. Qwen3-0.6B cache exists, but compatibility is UNKNOWN. BLIP-2/InstructBLIP/MiniGPT-4/Vicuna/R2-LLM/META-CXR/Swin named local weights NOT_FOUND in the enumerated roots. No checkpoint was loaded. `code_artifacts.csv` lists sizes and license-file presence; absence of a local license file does not establish licensing rights.

The local Salesforce cache is `blip-image-captioning-base`, which is not evidence of BLIP-2 weights. Qwen3-0.6B cache entry is present; Swin-Qwen3 combined checkpoint NOT_FOUND. Named MiniGPT-4, Vicuna, R2-LLM and META-CXR checkpoints NOT_FOUND in audited roots. `conda_envs.csv` records 25 environment prefixes; some package probes are UNKNOWN because their interpreter did not return JSON.
