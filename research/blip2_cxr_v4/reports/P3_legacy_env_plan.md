# P3.4 legacy R2/META environment plan — unexecuted

**Status: PLAN_ONLY / BLOCKED.** No environment creation, install, model load, GPU probe, or training was run. Official R2 weights are **NOT_PUBLICLY_RELEASED**, per Claude/QA's independent Hugging Face author listing check; this environment cannot reproduce official R2 inference. META checkpoint names, access, and root reuse license remain UNKNOWN (`reports/P3_weights_licensing.md`).

## Compatibility

| Item | Current K80 node | Upstream R2 / META | Proposed isolated R2 recipe |
|---|---|---|---|
| GPU/driver | 8× K80, 11,441 MiB each, sm37, driver 440.33.01 (`manifests/gpu_inventory.json`) | paper GPU UNKNOWN | CUDA 10.2 ceiling; no CUDA 11+/torch 2.x assumption |
| Python/torch | Python 3.10.21, torch 1.12.1+cu102 (`manifests/conda_envs.csv`) | R2 Python 3.9, torch 2.0.0 (`third_party/R2-LLM/environment.yaml:1-11`); META torch 1.13, torchvision .14 (`third_party/META-CXR/requirements.txt:17-18`) | Python 3.9, torch 1.12.1, torchvision .13.1, cudatoolkit 10.2 **candidate pins**, solver compatibility UNKNOWN |
| User packages | transformers 4.36.2, tokenizers .15.2; omegaconf/timm/peft absent (import-only probe below) | R2 transformers 4.30, tokenizers .13.2, timm .6.13, peft .2, accelerate .20.3, bnb .37 (`third_party/R2-LLM/environment.yaml:13-30`); META timm .4.12 and PEFT Git SHA (`third_party/META-CXR/requirements.txt:16-20`) | Proposed R2 pins in `envs/legacy_r2llm_cu102/requirements.lock.txt`; not dependency-resolved or hash-locked |

On K80 sm37, **bf16, FlashAttention, bitsandbytes LLM.int8 (requires sm75), supported NF4/FP4 (requires sm60), torch.compile, and torch 2 SDPA** are unavailable or outside torch1.12/cu102. [bitsandbytes requirements](https://huggingface.co/docs/bitsandbytes/main/installation#requirements) specify the hardware minima. FP16 arithmetic was probed successfully for X02 on this node (`reports/05_results_and_decisions.md`, X02 section), but that is not an R2 or META compatibility test.

## Required source changes before any runtime attempt

- R2 `third_party/R2-LLM/minigpt4/models/mini_gpt4.py:35-47,127-137`: disable `low_resource`/`load_in_8bit`; use explicit local Vicuna checkpoint path and avoid unsupported bnb path. Lines 103-119 initialize/freeze Q-Former and call `load_from_pretrained` with a URL default; require local-only pretrained Q-Former and ViT-G assets before construction. Lines 143-156 configure LoRA; confirm PEFT API with proposed pins. `generate_reports.py:100-105` chooses then hardcodes CUDA: make device mapping explicit and verify any two-K80 partitioning. Source modifications are *proposed*, not applied to third-party checkout.
- META `third_party/META-CXR/inference.py:309-327` loads Vicuna and adapters and initializes both models at import time; guard initialization under main and require local-only paths. `inference.py:312-323` uses `device_map='auto'` plus PEFT adapter; verify split placement and adapter source. `third_party/META-CXR/requirements.txt:16-20` targets a distinct torch/timm/PEFT stack, so the R2 recipe does not establish META compatibility. Its `.pth` name is still UNKNOWN (`third_party/META-CXR/README.md:107-120`).

## Memory and execution decision

Vicuna 7B FP16 **base weights alone** are approximately 14 GB decimal (7 billion × 2 bytes), about 13.0 GiB; 13B is ~26 GB decimal. A single 11.17 GiB K80 cannot hold even the 7B base before ViT-G, Q-Former, projection, activations, and KV cache. A two-K80 split would start near ~6.5 GiB per card for 7B weights, plus those other modules and activation/KV overhead; fit is UNKNOWN, and `device_map`/cross-device behavior needs a separately approved GPU test. Supported bitsandbytes 8/4-bit paths cannot rescue K80 sm37. No runtime or throughput estimate is verified for these models.

## Import-only probe and validation sequence

`timeout 60 env PYTHONDONTWRITEBYTECODE=1 HF_HOME=research/blip2_cxr_v4/.cache/hf XDG_CACHE_HOME=research/blip2_cxr_v4/.cache/xdg PYTHONPATH=research/blip2_cxr_v4/third_party/R2-LLM /home/student4/anaconda3/envs/cxr-llava-k80/bin/python -c 'import minigpt4.models.mini_gpt4'` stops at `ModuleNotFoundError: omegaconf` before any model instantiation. The dataset and common-config module imports stop at the same missing dependency. Thus existing `cxr-llava-k80` cannot run R2 unchanged. If licenses/assets ever become available and a separate installation is approved: solve recipe in isolated `/storage` prefix; import-only test; CPU synthetic shape/config test without weights; verify all local checkpoint hashes and no network requests; then separately authorize a tiny multi-GPU fit test. Do not call the proposed pins verified until the solver and imports pass.

`install.sh` and `rollback.sh` require `--execute /storage/<isolated-prefix>` and were **not executed**. Install refuses an existing prefix; rollback refuses any prefix lacking its exact marker and removes only that prefix through `conda env remove`. The script writes caches under `/storage`. No environment or third-party source file was changed during this task.
