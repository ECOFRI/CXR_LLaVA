# F0 — CLEAR finding tokens + Qwen3: technical preflight

## 1. Experiment Information

Date: 2026-11-13. Project: CXR_LLaVA_Improvement. Execution: Codex on ict11.
Status: **COMPLETE_TECHNICAL_PREFLIGHT**. Classification: **smoke / technical performance preflight**, not quality evaluation or clinical validation.
Branch `research/e4-b1-evidence-format-gate`, commit `ecb7fddbe709dc60432baed8b0f24b15da93422d`; uncommitted F0 files and user work order preserved. No commit/push.

## 2. Objective

Measure actual CLEAR extraction, optimizer-step throughput, memory and generation latency under WO_F0; no full training or quality metrics.
User amendments: keep 448×448 input, save spatial 2×2-pooled **256×768 FP16** patches, pool again at load for **64 tokens**; optional 256-token timing if budget remains.

## 3. Survey / Technical Motivation

Reuses local official CLEAR loader, native DINOv2 patch outputs and `research/e2_direct.py` paired scoring; does not use the full CLEAR concept bank.
[Qwen model card](https://huggingface.co/Qwen/Qwen3-1.7B) documents Transformers >=4.51 and thinking control.
[Official PyTorch installation](https://pytorch.org/get-started/previous-versions/) supplies the CUDA 12.4 wheels.
Public model revisions and CLEAR SHA are recorded in [model_download.json](model_download.json).

## 4. Environment

ict11: Quadro RTX 4000, 8192 MiB, driver 550.163.01, CUDA driver 12.4.
New isolated prefix `research/envs/rrg-qwen`; existing environments unchanged.
Python 3.11.18; torch 2.6.0+cu124; Transformers 4.51.3; PEFT 0.15.2; Accelerate 1.6.0.
[env.lock.txt](env.lock.txt), [runtime.json](runtime.json), [hardware.json](hardware.json).
Storage reports 100% with ~1.1T available; personal quota unknown. Large data/checkpoint outputs stay in ignored `features/`; weights/install caches under `model_cache/rrg_f0/`; isolated env is the authorized install output.

## 5. Dependency Setup

Executed conda prefix creation with Python 3.11/pip, then pinned torch 2.6.0 + torchvision 0.21.0 from cu124, Transformers 4.51.3, PEFT 0.15.2, Accelerate 1.6.0, NumPy<2, hub 0.30.2 plus tokenizer/preprocess dependencies.
Exact reusable commands: [setup_env.sh](setup_env.sh); installation evidence: `dependencies.log`, `env_setup.log`. `pip check`: no broken requirements.
Initial installer caches were under task-specific research cache paths; after install they were moved into model_cache, and the setup script now uses that location. No existing env was upgraded.

## 6. Code / Notebook Changes

New F0-only modules: cohort/token helpers, extractor, feature audit, dataset, model bridge, bounded preflight, cumulative GPU guard, CPU tests, report renderer and README. Root `.gitignore` adds `/research/rrg_f0/features/`.
Existing model code, manifests, checkpoints and experiment reports were not edited.

## 7. Experiment Configuration

Original P4 manifests: 2,000 MIMIC train + 500 official validation; hashes match, all paths exist, train/val patients disjoint. No official MIMIC test or locked CheXpert test used.
CLEAR: frozen FP32, official 448×448 preprocess, 32×32 native patch grid; disk256 FP16; primary load64; 14 lowercase positive/`no` prompt pairs, normalized cosine paired softmax without logit scale.
Bridge: 768→hidden→hidden GELU MLP, 14 learned finding embeddings + scalar-score MLP; auxiliary linear head on mean native features.
Qwen: frozen FP16; trainables FP32. LoRA rank16/alpha32/all-linear/dropout0 vs no LoRA. AdamW lr1e-4, eps1e-6, clip1, GradScaler initial scale128; gradient checkpointing non-reentrant.
LM loss masks image/finding prefix, prompt and padding; Findings-only target capped at 256 tokens including EOS; text pads to 320 positions (total length398 for64,590 for256).
BCE targets: **synthetic all-zero 14 labels**, weight0.1, permitted only for this execution preflight. P4 labeler outputs cover validation/generation references, not the training set; no validation labels are used to train.
Timing repeats a bounded original train minibatch with Findings. Missing Findings are skipped in the timing sampler, not replaced by Impression; no cohort manifests are regenerated.
Primary arms: at most 20 optimizer updates each. Capacity probes use backward, no optimizer updates; Adam state included in final capacity verification. Optional LoRA256: batch 1, at most3 updates, timing/memory only.
Generation: saved step 20 state verified/reloaded; first 8 validation feature rows/arm, no reference prompt evidence; greedy, thinking disabled, max 128 new tokens.

## 8. Code Executed

Reproduction sequence and bounded commands: [README.md](README.md).
Each GPU command runs through `gpu_guard.py` with a cumulative3600s cap, exclusive stage lock, an independent timeout watchdog and process-group termination. Model load/import/I/O inside GPU children are charged conservatively.
Per-stage allowance: CLEAR20 300s; CLEAR2500 1200s; primary arms 900s each; optional256 300s, additionally clipped to cumulative remaining budget.
Checkpoint/resume: atomic rolling saves every 5 steps, optimizer/scaler/RNG, config/cohort/revision validation; primary cap remains 20 after resume.

## 9. Output

CPU CLEAR20: **0.836 s/image**.
GPU CLEAR20: **0.222 s/image**, peak allocated/reserved **627.9/724.0 MiB**.
Full extraction: n=2500; compute/preprocess **0.228 s/image**; end-to-end loop including feature writes/hash **0.257 s/image** (excludes model load).
Raw aggregate evidence: benchmark JSONs, extraction.json, feature_contract.json, arm JSONs and local .log files. Private feature/checkpoint/generated-report artifacts remain ignored and local; no MIMIC text/images are reproduced here.

## 10. Output Analysis

First128 training references audited:31 lack Findings. Bounded sampler omissions are separately counted per arm in its JSON.
The CPU and GPU extraction benchmarks use different torch builds; no CPU/GPU numerical-equivalence claim is made.
Batch capacity and loss trend apply only to the fixed padded preflight shape and repeated timing minibatch. Synthetic BCE and repeated samples do not measure learning/generalization.

## 11. Quantitative Results

| Arm | Patch tokens | Batch | Updates | s/step | s/sample | Peak allocated / reserved MiB | s/report |
|---|---:|---:|---:|---:|---:|---:|---:|
| lora64 | 64 | 5 | 20 | 1.370 | 0.274 | 7321.8 / 7528.0 | 2.817 |
| bridge64 | 64 | 5 | 20 | 0.893 | 0.179 | 7124.8 / 7826.0 | 3.374 |
| lora256 | 256 | 1 | 3 | 0.567 | — | 4715.2 / 5078.0 | N/A |

GPU **charged wall total: 1191.4 s = 19.86 min**, limit 3600s. [gpu_budget.json](gpu_budget.json).
VRAM columns are PyTorch allocated/reserved peaks, not a sampled peak of whole-device memory; CUDA context/other driver allocations are additional. Largest batch is conditional on this sequence shape and optimizer state.
Step timing includes zero_grad, forward, backward and optimizer update; excludes batch loading and checkpoint writes. Raw step timings/probes are retained.

## 12. Qualitative Results

No clinical/qualitative assessment. Generated reports are experimental outputs, retained locally only for generation timing; no assertion of correct findings or diagnosis.

- lora64: loss 4.545 → 0.265; decreased=True; finite=True; generation n=8, tokens=[35, 45, 60, 37, 67, 58, 128, 46]; cap128=1/8.
- bridge64: loss 4.614 → 2.809; decreased=True; finite=True; generation n=8, tokens=[128, 128, 128, 128, 128, 128, 128, 128]; cap128=8/8.

## 13. Comparison with Baseline / Revised Estimates

| Arm | 30k samples / epoch (h) | 100k samples / epoch (h) | 3,858 reports (h) |
|---|---:|---:|---:|
| lora64 | 2.283 | 7.609 | 3.019 |
| bridge64 | 1.488 | 4.961 | 3.616 |

The plan's 0.25–0.4 s/sample implies 30k=2.083–3.333h and 100k=6.944–11.111h per epoch. Compare only with the measured optimizer s/sample above; extraction latency is a separate quantity. LoRA64 measured 0.274 s/sample is within the plan's range; bridge64 measured 0.179 s/sample is below that range, conditional on this fixed padded preflight and cached features.
One-time feature extraction at measured end-to-end loop rate: **30k=2.140h; 100k=7.135h; 3,858=0.275h**. Add this once for first-epoch or uncached-test estimates; model loading/setup is additional. The 3,858-case count is the WO's extrapolation input, not independently re-audited here.

Extrapolations assume identical hardware, precision, token limits and batch; training estimates omit dataloader/checkpoint/setup overhead, and generation estimates use 8 reports/arm at max 128 new tokens. Capped output is not necessarily a completed report. LoRA64 hit the cap 1/8 times; bridge64 hit it 8/8 times. Mean generation lengths differ, so s/report is not a matched-length speed comparison. These are estimates, not runs on 30k/100k or the 3,858-case test; the test set was not accessed.

## 14. Problems Encountered / Risks

- Original 2×2→64 request conflicts with 448 input; resolved explicitly by user's disk256/load64 amendment.
- New env installation and public weight downloads took substantial wall time on shared storage; excluded from GPU consumption, but GPU-child load times are charged.
- CPU tests emitted a Transformers SDPA sliding-window warning for the tiny test config; actual pretrained config is preserved. xFormers is absent; no backend or architecture substitution was made.
- The optional 256-token row reports only timing/memory; 3 updates at batch 1 are a small timing sample, not a matched-batch 64/256 comparison. Its JSON inherits generic capacity-limit wording; no batch search was run for that arm (fixed batch 1).
- Generation config emitted warnings that sampling-only flags are ignored under greedy decoding; no sampling was used.
- Train labels remain a pre-training decision; synthetic labels here cannot establish utility. Timing sample is small and repeated; 320-position text padding and 128-token generation limit constrain extrapolation.
- Hardware is RTX4000 on ict11; historical K80 settings were not applied to Qwen. No precision, offload or quantization recovery was substituted after a hard-stop error.
- Check stage logs/budget for any failure or skipped optional arm; N/A values were not imputed as zero.

## 15. Conclusion

COMPLETE_TECHNICAL_PREFLIGHT. F0 records feasibility/resource timing only. No clinical benefit, accuracy, benchmark improvement or readiness for full training is established. No MIMIC data was uploaded; no commit/push was performed.

## 16. Next Experiment

Any real training requires its own approved labels, data/target protocol, model configuration and budget. Keep 256-token cached features for later authorized token-count ablation. Do not infer permission for30k/100k training or official test execution from the estimates.

## 17. Reproducibility Checklist

Source/index/config hashes and git state: [provenance.json](provenance.json). Model revisions/SHA, env lock, raw aggregate outputs, budgets, fixed shape/seed/optimizer and CPU tests are retained. Private artifacts are excluded from Git. CPU checkpoint audit verifies both saved step 20 states and every optimizer counter at 20; see `checkpoint_audit.json`.
CPU integration: 5/5 PASS; root catalog tests: 2/2 PASS; AST syntax, shell syntax and diff whitespace checked. GPU results are reported separately above; CPU tests do not establish clinical accuracy.

## 18. Short Lab Update

Local smoke/preflight of frozen CLEAR finding tokens + frozen Qwen3 with/without rank16 LoRA under a 1h cumulative GPU cap. Disk256/load64 pooling follows the user's amendment. Measurements and assumptions are retained; no quality evaluation or full training was run.
