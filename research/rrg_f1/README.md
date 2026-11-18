# F1: 30k CLEAR tokens + Qwen, fixed exposed dev evaluation

Authorized by `plan/work_orders/WO_F1_2026-11-13.md`. This is research only.
No clinical claim or independent test benchmark. No private data leaves the server.

`private/` and `paths.local.json` are ignored. They contain the actual patient
manifest, Findings/token IDs, case labels, features, generated reports, checkpoints,
worker logs, budgets, locks and heartbeat. **Never stage these files.** Public JSON
summaries contain counts, hashes, configuration and aggregate metrics only.

The selected 30,000 studies are a seeded uniform sample of eligible official TRAIN
frontal studies with nonempty Findings, using the first available AP/PA image in
split-file order. Selection consumes a seeded permutation in order, even with16
I/O workers. The original P4 500 MIMIC validation and500 Indiana rows remain fixed.
Coarse checkpoints are atomic; resume truncates only an uncheckpointed manifest
tail and verifies source/config hashes. The completed manifest is immutable.

Legacy P4 Python3.6/NLTK3.3 is separate from F0 Qwen Python3.11. The original
labeler and NegBio revisions are pinned. Public parser GENIA+PubMed, CoreNLP3.5.2,
NLTK punkt/universal tagset and the pinned MKL2019.1/OpenMP/libffi shared libraries
were restored into model cache; the original WordNet ZIP hash is unchanged.
`runtime_env()` supplies legacy libraries only to CPU labeler/evaluator children.
No package in the existing environments was upgraded. Sixteen labeler processes
share an affinity mask of16 CPUs; the200-report throughput gate must project no
more than8 hours. Findings labels retain all four states. BCE uses Positive=1,
Negative/Blank=0 and masks Uncertain. Columns are mapped by name, not position.

Shared F0 `FindingBridge`, loss masking, spatial pooling, tokenizer prompt and
CLEAR encoding are reused. F0 source was factored into reusable helpers; its
historical raw runs/reports were preserved. F1 caches256×768 FP16 tokens after
2×2 pooling at448 input, and pools again to64 at loading. The500 MIMIC validation
features are reused from F0. Feature extraction batch4 has8 CPU loaders.

A2/A3/A4 all have rank16 LoRA, FP16 frozen Qwen and FP32 trainables. The shared
projector starts from the same seed. A2 has visual tokens only; A3 adds BCE with
train-only smoothed prior log-odds added to logits; A4 also adds14 finding tokens
from CLEAR zero-shot scores. F1's BCE head reads the projected visual tokens so
its gradient reaches the projector; F0's detached native-feature timing head is
not used for the training ablation. The binary logit adjustment is a declared
extension of the [logit-adjustment approach](https://arxiv.org/abs/2007.07314).

`train_config.json` fixes all three arms: microbatch4, accumulation up to8 (effective
batch32), AdamW1e-4, 48-update warmup then cosine over960 planned updates. Accumulation
flushes at1000-sample dev checks and epoch end. Text pads to320, target Findings
caps at256 including EOS. Each1000 examples produces16 dev generations (first8
MIMIC + first8 Indiana), greedy256, thinking off. Two successive checks with
unique<4 OR cap-count>=12 stop that arm as COLLAPSE. Save points include every
2000 examples and the1000-example gates for idempotent recovery. Resume verifies
config/code/cohort/labels, optimizer/scaler/RNG and the single-epoch position;
terminal arms cannot train again. Raw failed-attempt logs remain private.

`supervisor.py` is detached, locked and writes a heartbeat every10 seconds.
Every GPU child runs through the generalized F0 guard with an independent timeout,
a cumulative **50,400s /14h** ledger and cumulative **14,400s /4h per training arm**.
Loading/I/O/resumes are charged conservatively. Never remove/reset the ledger.
Any OOM or failed stage stops the supervisor. Three collapsed arms stop the whole
WO. No automatic batch reduction, extra arm, epoch or architecture change.

Inference is batched greedy256 on the original dev cohorts. Private per-case
outputs are checkpoint-bound and resume without repeating completed cases. A4
also uses zero-score and seeded shuffled-score controls on first50 MIMIC + first50
Indiana cases. Control direction is measured with label changes and score changes,
without GT-based prompts. P4's tokenizer, BLEU/ROUGE/METEOR and clinical evaluator
functions are reused in the restored legacy CPU environment. Historical jointly
definite and blank-as-negative F1 use P4's7-class ontology, with coverage and actual
F1 denominators. Bootstrap uses1000 paired patient-cluster draws (Indiana study
clusters), and unadjusted percentile intervals. Empty generations are retained as
zero text scores/blank labels; historical baseline predictions are not regenerated.
MIMIC references remain P4 Findings+Impression for direct comparison, although F1
trains Findings only; generation caps and model protocols also differ from P4.

Configure local paths by copying `paths.example.json` to `paths.local.json` and
filling authorized server paths. Existing F0 weights/features must be available.
Restore public missing assets with `python3 -m research.rrg_f1.setup_assets` (legacy Python itself must already be restored).
From the repository root, validate then launch (no training starts during tests):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 research/envs/rrg-qwen/bin/python -m unittest research.rrg_f1.test_cohort research.rrg_f1.test_model research.rrg_f1.test_budget research.rrg_f1.test_encoder -v
nohup setsid bash research/rrg_f1/run_f1.sh > research/rrg_f1/private/supervisor.log 2>&1 < /dev/null &
```

Use the same launch command to resume after ensuring the prior supervisor is gone;
the lock refuses concurrent execution. Inspect `private/supervisor_status.json`,
`private/supervisor_heartbeat.json`, arm heartbeat files and `private/gpu_budget.json`.
Source/config changes after protocol lock require explicit review, not silent resume.
The runner does not commit or push. Commit only reviewed public source/config/docs
and aggregate evidence after the final privacy scan; never upload private outputs.

Read-only progress: `PYTHONDONTWRITEBYTECODE=1 python3 -m research.rrg_f1.status`.
