"""Render a new technical report from local aggregate evidence; no clinical text."""
import json
import re

from research.rrg_f0.core import ROOT

BASE = ROOT / 'research/rrg_f0'


def read(name):
    path = BASE / name
    return json.loads(path.read_text()) if path.exists() else {}


def number(value, digits=3):
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) else 'N/A'


def render(prefix):
    cpu, gpu, extraction = read('benchmark_cpu.json'), read('benchmark_cuda.json'), read('extraction.json')
    runtime, provenance, ledger = read('runtime.json'), read('provenance.json'), read('gpu_budget.json')
    arms = [(name, read(name + '.json')) for name in ('lora64', 'bridge64', 'lora256')]
    primary_done = all(row.get('status') == 'PASS' and row.get('optimizer_steps') == 20 and row.get('generation_count') == 8 for _, row in arms[:2])
    complete = primary_done and read('feature_contract.json').get('count') == 2500
    status = 'COMPLETE_TECHNICAL_PREFLIGHT' if complete else 'STOP_OR_PARTIAL; see stage logs and budget'
    charged = sum(a['charged_seconds'] for a in ledger.get('attempts', []))
    table = ['| Arm | Patch tokens | Batch | Updates | s/step | s/sample | Peak allocated / reserved MiB | s/report |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    estimates = ['| Arm | 30k samples / epoch (h) | 100k samples / epoch (h) | 3,858 reports (h) |', '|---|---:|---:|---:|']
    trends = []
    for name, row in arms:
        c = row.get('contract', {})
        table.append(f"| {name} | {256 if name.endswith('256') else 64} | {c.get('batch_size', 'N/A')} | {row.get('optimizer_steps', 'N/A')} | {number(row.get('seconds_per_step'))} | {number(row.get('seconds_per_sample')) if name.endswith('64') else '—'} | {number(row.get('peak_allocated_mib'), 1)} / {number(row.get('peak_reserved_mib'), 1)} | {number(row.get('seconds_per_report'))} |")
        if name.endswith('64'):
            sample, report = row.get('seconds_per_sample'), row.get('seconds_per_report')
            estimates.append(f"| {name} | {number(sample * 30000 / 3600 if sample else None)} | {number(sample * 100000 / 3600 if sample else None)} | {number(report * 3858 / 3600 if report else None)} |")
            cap = sum(t == 128 for t in row.get('generation_tokens', []))
            trends.append(f"- {name}: loss {number(row.get('first_loss'))} → {number(row.get('last_loss'))}; decreased={row.get('loss_decreased', 'N/A')}; finite={row.get('finite', 'N/A')}; generation n={row.get('generation_count', 0)}, tokens={row.get('generation_tokens', [])}; cap128={cap}/8.")
    io_per_image = extraction.get('wall_seconds_including_feature_writes', 0) / extraction.get('count', 1) if extraction else None
    feature_estimate = 'N/A' if io_per_image is None else f'30k={io_per_image * 30000 / 3600:.3f}h; 100k={io_per_image * 100000 / 3600:.3f}h; 3,858={io_per_image * 3858 / 3600:.3f}h'
    rendered = f'''# F0 — CLEAR finding tokens + Qwen3: technical preflight

## 1. Experiment Information
Date: 2026-11-13. Project: CXR_LLaVA_Improvement. Execution: Codex on ict11.
Status: **{status}**. Classification: **smoke / technical performance preflight**, not quality evaluation or clinical validation.
Branch `{provenance.get('git_branch', 'N/A')}`, commit `{provenance.get('git_commit', 'N/A')}`; uncommitted F0 files and user work order preserved. No commit/push.

## 2. Objective
Measure actual CLEAR extraction, optimizer-step throughput, memory and generation latency under WO_F0; no full training or quality metrics.
User amendments: keep 448×448 input, save spatial 2×2-pooled **256×768 FP16** patches, pool again at load for **64 tokens**; optional 256-token timing if budget remains.

## 3. Survey / Technical Motivation
Reuses local official CLEAR loader, native DINOv2 patch outputs and `research/e2_direct.py` paired scoring; does not use the full CLEAR concept bank.
[Qwen model card](https://huggingface.co/Qwen/Qwen3-1.7B) documents Transformers >=4.51 and thinking control.
[Official PyTorch installation](https://pytorch.org/get-started/previous-versions/) supplies the CUDA 12.4 wheels.
Public model revisions and CLEAR SHA are recorded in [{prefix}model_download.json]({prefix}model_download.json).

## 4. Environment
ict11: Quadro RTX 4000, 8192 MiB, driver 550.163.01, CUDA driver 12.4.
New isolated prefix `research/envs/rrg-qwen`; existing environments unchanged.
Python {runtime.get('python', 'N/A')}; torch {runtime.get('torch', 'N/A')}; Transformers {runtime.get('transformers', 'N/A')}; PEFT {runtime.get('peft', 'N/A')}; Accelerate {runtime.get('accelerate', 'N/A')}.
[{prefix}env.lock.txt]({prefix}env.lock.txt), [{prefix}runtime.json]({prefix}runtime.json), [{prefix}hardware.json]({prefix}hardware.json).
Storage reports 100% with ~1.1T available; personal quota unknown. Large data/checkpoint outputs stay in ignored `features/`; weights/install caches under `model_cache/rrg_f0/`; isolated env is the authorized install output.

## 5. Dependency Setup
Executed conda prefix creation with Python 3.11/pip, then pinned torch 2.6.0 + torchvision 0.21.0 from cu124, Transformers 4.51.3, PEFT 0.15.2, Accelerate 1.6.0, NumPy<2, hub 0.30.2 plus tokenizer/preprocess dependencies.
Exact reusable commands: [{prefix}setup_env.sh]({prefix}setup_env.sh); installation evidence: `dependencies.log`, `env_setup.log`. `pip check`: no broken requirements.
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
Reproduction sequence and bounded commands: [{prefix}README.md]({prefix}README.md).
Each GPU command runs through `gpu_guard.py` with a cumulative3600s cap, exclusive stage lock, an independent timeout watchdog and process-group termination. Model load/import/I/O inside GPU children are charged conservatively.
Per-stage allowance: CLEAR20 300s; CLEAR2500 1200s; primary arms 900s each; optional256 300s, additionally clipped to cumulative remaining budget.
Checkpoint/resume: atomic rolling saves every 5 steps, optimizer/scaler/RNG, config/cohort/revision validation; primary cap remains 20 after resume.

## 9. Output
CPU CLEAR20: **{number(cpu.get('seconds_per_image_compute_and_preprocess'))} s/image**.
GPU CLEAR20: **{number(gpu.get('seconds_per_image_compute_and_preprocess'))} s/image**, peak allocated/reserved **{number(gpu.get('peak_allocated_mib'),1)}/{number(gpu.get('peak_reserved_mib'),1)} MiB**.
Full extraction: n={extraction.get('count', 'N/A')}; compute/preprocess **{number(extraction.get('seconds_per_image_compute_and_preprocess'))} s/image**; end-to-end loop including feature writes/hash **{number(io_per_image)} s/image** (excludes model load).
Raw aggregate evidence: benchmark JSONs, extraction.json, feature_contract.json, arm JSONs and local .log files. Private feature/checkpoint/generated-report artifacts remain ignored and local; no MIMIC text/images are reproduced here.

## 10. Output Analysis
First128 training references audited:31 lack Findings. Bounded sampler omissions are separately counted per arm in its JSON.
The CPU and GPU extraction benchmarks use different torch builds; no CPU/GPU numerical-equivalence claim is made.
Batch capacity and loss trend apply only to the fixed padded preflight shape and repeated timing minibatch. Synthetic BCE and repeated samples do not measure learning/generalization.

## 11. Quantitative Results
{chr(10).join(table)}

GPU **charged wall total: {number(charged,1)} s = {number(charged / 60,2)} min**, limit 3600s. [{prefix}gpu_budget.json]({prefix}gpu_budget.json).
VRAM columns are PyTorch allocated/reserved peaks, not a sampled peak of whole-device memory; CUDA context/other driver allocations are additional. Largest batch is conditional on this sequence shape and optimizer state.
Step timing includes zero_grad, forward, backward and optimizer update; excludes batch loading and checkpoint writes. Raw step timings/probes are retained.

## 12. Qualitative Results
No clinical/qualitative assessment. Generated reports are experimental outputs, retained locally only for generation timing; no assertion of correct findings or diagnosis.

{chr(10).join(trends)}

## 13. Comparison with Baseline / Revised Estimates
{chr(10).join(estimates)}

The plan's 0.25–0.4 s/sample implies 30k=2.083–3.333h and 100k=6.944–11.111h per epoch. Compare only with the measured optimizer s/sample above; extraction latency is a separate quantity. LoRA64 measured 0.274 s/sample is within the plan's range; bridge64 measured 0.179 s/sample is below that range, conditional on this fixed padded preflight and cached features.
One-time feature extraction at measured end-to-end loop rate: **{feature_estimate}**. Add this once for first-epoch or uncached-test estimates; model loading/setup is additional. The 3,858-case count is the WO's extrapolation input, not independently re-audited here.

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
{status}. F0 records feasibility/resource timing only. No clinical benefit, accuracy, benchmark improvement or readiness for full training is established. No MIMIC data was uploaded; no commit/push was performed.

## 16. Next Experiment
Any real training requires its own approved labels, data/target protocol, model configuration and budget. Keep 256-token cached features for later authorized token-count ablation. Do not infer permission for30k/100k training or official test execution from the estimates.

## 17. Reproducibility Checklist
Source/index/config hashes and git state: [{prefix}provenance.json]({prefix}provenance.json). Model revisions/SHA, env lock, raw aggregate outputs, budgets, fixed shape/seed/optimizer and CPU tests are retained. Private artifacts are excluded from Git. CPU checkpoint audit verifies both saved step 20 states and every optimizer counter at 20; see `checkpoint_audit.json`.
CPU integration: 5/5 PASS; root catalog tests: 2/2 PASS; AST syntax, shell syntax and diff whitespace checked. GPU results are reported separately above; CPU tests do not establish clinical accuracy.

## 18. Short Lab Update
Local smoke/preflight of frozen CLEAR finding tokens + frozen Qwen3 with/without rank16 LoRA under a 1h cumulative GPU cap. Disk256/load64 pooling follows the user's amendment. Measurements and assumptions are retained; no quality evaluation or full training was run.
'''
    return re.sub(r'(?m)^(## [^\n]+)\n(?!\n)', r'\1\n\n', rendered)


def main():
    paths = [(BASE / 'F0_preflight_report.md', ''),
             (ROOT / 'reports/2026-11-13_F0_preflight_analysis.md', '../research/rrg_f0/')]
    if any(path.exists() for path, _ in paths):
        raise FileExistsError('Preserve existing F0 reports')
    for path, prefix in paths:
        with path.open('x') as stream:
            stream.write(render(prefix))
    print('Created two new template-based F0 reports; clinical text excluded')


if __name__ == '__main__':
    main()
