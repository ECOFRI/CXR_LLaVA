# F0 local technical preflight

Scope: [WO_F0](../../plan/work_orders/WO_F0_2026-11-13.md), plus the user's
448×448 / disk-256 / load-64 amendment and optional 256-token timing.
No quality evaluation, clinical claim, official MIMIC test, or locked CheXpert test.

Only public model weights are downloaded. MIMIC images, reports, features,
labels, checkpoints and generated reports stay on this server. Never print
clinical text or upload this directory's ignored artifacts.

`config.json` records timing assumptions. The original 2000/500 P4 manifests
remain unchanged and are verified by hash. `features/` stores 256×768 FP16
patches after spatial 2×2 pooling of the native 32×32 patch grid. Loading applies
another spatial 2×2 mean for the 64-token arms, with FP32 accumulation.
Paired zero-shot scores use lowercase label / `no label`, normalized cosine
softmax without temperature scaling. This is the direct encoder route, not the
full CLEAR concept-bank method.

The bounded timing minibatch uses original training rows with Findings; missing
Findings are skipped, never replaced by Impression. The LM target is Findings
only, truncated to 256 tokens including EOS; prompt/prefix/padding are masked.
Text pads to 320 positions. The 14 auxiliary targets are synthetic zeros for
execution only: P4 saved labeler outputs cover evaluation, not the training set.
Gradient checkpointing, FP16 frozen Qwen, FP32 bridge/LoRA trainables, AdamW,
GradScaler, and repeated timing minibatches are F0 assumptions, not a training
protocol. Batch probes use backward but **no optimizer update**; primary arms
are capped at 20 updates each. The optional 256-token arm has 3 updates.
Generation is greedy, thinking disabled, at most 128 new tokens, on the first
8 P4 validation feature rows per primary arm; references are never prompt evidence.

Commands from repository root (run sequentially; inspect each stage before the
next; do not rerun over existing evidence):

```bash
bash research/rrg_f0/setup_env.sh
export HF_HOME="$PWD/model_cache/rrg_f0/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
# Fresh run only; preserves existing download evidence.
CUDA_VISIBLE_DEVICES='' research/envs/rrg-qwen/bin/python -m research.rrg_f0.download_models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' research/envs/rrg-qwen/bin/python -m unittest research.rrg_f0.test_core research.rrg_f0.test_qwen -v
CUDA_VISIBLE_DEVICES='' research/envs/cxr-clear-cpu/bin/python -m research.rrg_f0.extract --device cpu --benchmark
CUDA_VISIBLE_DEVICES=0 python3 -m research.rrg_f0.gpu_guard --stage clear20 --seconds 300 research/envs/rrg-qwen/bin/python -m research.rrg_f0.extract --device cuda --benchmark
# Choose faster device after inspecting both 20-image measurements.
CUDA_VISIBLE_DEVICES=0 python3 -m research.rrg_f0.gpu_guard --stage clear2500 --seconds 1200 research/envs/rrg-qwen/bin/python -m research.rrg_f0.extract --device cuda
CUDA_VISIBLE_DEVICES='' research/envs/rrg-qwen/bin/python -m research.rrg_f0.audit_features
CUDA_VISIBLE_DEVICES=0 python3 -m research.rrg_f0.gpu_guard --stage lora64 --seconds 900 research/envs/rrg-qwen/bin/python -m research.rrg_f0.preflight --arm lora64
CUDA_VISIBLE_DEVICES=0 python3 -m research.rrg_f0.gpu_guard --stage bridge64 --seconds 900 research/envs/rrg-qwen/bin/python -m research.rrg_f0.preflight --arm bridge64
# Optional only if cumulative GPU ledger has enough time left.
CUDA_VISIBLE_DEVICES=0 python3 -m research.rrg_f0.gpu_guard --stage lora256 --seconds 300 research/envs/rrg-qwen/bin/python -m research.rrg_f0.preflight --arm lora256
```

`gpu_guard.py` locks the GPU stage, refuses an occupied GPU, pre-reserves stage
wall time in `gpu_budget.json`, and uses an independent `timeout` watchdog plus process-group termination. Sum of
charged process wall times is capped below 3600 seconds, conservatively counting
imports/loading/I/O in each GPU child. A crashed guard leaves its reservation
charged. Never remove/reset this ledger to regain budget. CPU extraction is
outside the GPU ledger. Stop on batch-1 OOM, unavailable weights or SHA mismatch;
do not change architecture or quantize/offload as an automatic recovery.

Checkpoint resume is explicit (`--resume`) and validates config/index/model
revision, trainable parameter names and timing contract. Rolling checkpoints every 5 steps
retain optimizer/scaler/RNG; the final checkpoint is verified at step 20; no step beyond 20 is allowed for a primary arm.
Previous generated reports must be preserved.
