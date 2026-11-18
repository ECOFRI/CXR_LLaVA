"""New aggregate-only F1 report, with explicit partial/stop classification."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from research.rrg_f0.core import ROOT
from research.rrg_f1.cohort import BASE, PRIVATE


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def fmt(value, digits=4):
    return ('%.*f' % (digits, value)) if isinstance(value, (int, float)) else 'N/A'


def render():
    cohort, labels, bench = read(BASE/'cohort_summary.json'), read(BASE/'train_label_summary.json'), read(BASE/'label_benchmark_summary.json')
    execution, metrics = read(BASE/'execution_summary.json'), read(BASE/'metrics.json')
    ledger = read(PRIVATE/'gpu_budget.json')
    arms = {a: read(BASE/(a+'_summary.json')) for a in ('A2','A3','A4')}
    total = sum(a['charged_seconds'] for a in ledger.get('attempts',[]))
    classification = 'metric-complete evaluation on fixed exposed dev cohorts; NOT test benchmark' if metrics.get('status') == 'PASS' else 'partial / stopped execution; quality evaluation incomplete'
    times = ['| Stage | Seconds | Status |','|---|---:|---|',
             '| Cohort (checkpointed cumulative) | '+fmt(cohort.get('wall_seconds_cumulative'),1)+' | '+cohort.get('status','N/A')+' |',
             '| Labeler 200-report gate | '+fmt(bench.get('wall_seconds'),1)+' | '+bench.get('status','N/A')+' |']
    times.extend('| '+str(s['stage'])+' | '+fmt(s.get('wall_seconds'),1)+' | '+str(s['status'])+' |' for s in execution.get('stages',[]))
    train = ['| Arm | Status | Samples seen | Optimizer updates | Last LM / BCE loss |','|---|---|---:|---:|---|']
    for arm, state in arms.items():
        loss=state.get('last_training_step') or {}
        train.append('| '+arm+' | '+state.get('status','NOT_STARTED')+' | '+str(state.get('examples_seen','N/A'))+' | '+str(state.get('optimizer_updates','N/A'))+' | '+fmt(loss.get('lm_loss'))+' / '+fmt(loss.get('bce_loss'))+' |')
    tables=[]
    for family, data in metrics.get('datasets',{}).items():
        tables += ['### '+family,'', '| Model | BLEU1 | BLEU2 | BLEU3 | BLEU4 | ROUGE-L | METEOR | Hist micro-F1 | Blank-neg micro-F1 | Hist paired/possible | F1 denominator |',
                   '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for model, row in data['models'].items():
            t=row['text']; h=row['clinical']['historical']; alt=row['clinical']['blank_as_negative']
            tables.append('| '+model+' | '+' | '.join(fmt(t[k]) for k in ('bleu1','bleu2','bleu3','bleu4','rouge_l','meteor'))+' | '+fmt(h['micro_f1'])+' | '+fmt(alt['micro_f1'])+' | '+str(h['paired_label_count'])+'/'+str(h['possible_class_cases'])+' | '+str(h['micro_f1_denominator'])+' |')
        tables.append('')
    if not tables:
        tables=['No new quality metrics are available. N/A is not zero; no metric-complete claim.']
    intervention=read(BASE/'intervention_summary.json')
    intervention_lines=[]
    for mode,row in intervention.get('modes',{}).items():
        intervention_lines.append('- '+mode+': changed reports '+str(row['changed_reports'])+'/100; directional label changes '+str(row['direction_concordant'])+' concordant / '+str(row['direction_opposite'])+' opposite; paired definite class-cases '+str(row['paired_definite_class_cases'])+'.')
    checks=[]
    for arm,state in arms.items():
        for c in state.get('checks',[]):
            checks.append('| '+arm+' | '+str(c['position'])+' | '+str(c['unique_reports'])+' | '+str(c['cap_count'])+' | '+fmt(c['mean_repeated_4gram_fraction'])+' |')
    parent=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    created=datetime.now(timezone.utc).isoformat()
    return f'''# F1 — CLEAR tokens + Qwen3, 30k training and fixed-dev evaluation

## 1. Experiment Information

Created UTC: {created}. Project: CXR_LLaVA_Improvement. Parent commit: `{parent}`.
Work order: `plan/work_orders/WO_F1_2026-11-13.md`.
Execution status: **{execution.get('status','UNKNOWN')}**; reason: **{execution.get('reason') or 'none'}**.
Classification: **{classification}**.

## 2. Objective

Label Findings for 30k official-train studies, train the matched A2/A3/A4 arms once,
and evaluate on the original P4 500 MIMIC validation + 500 Indiana cases. No official
test or locked CheXpert test; no clinical validation, expert adjudication or diagnosis.

## 3. Survey / Technical Motivation

F0 established feasibility only. F1 reuses F0 pooling, CLEAR forward, bridge and LM
loss masking. [Logit adjustment](https://arxiv.org/abs/2007.07314) motivates the declared
binary prior-log-odds extension. Labeler and NegBio revisions are the pinned P4
versions. P4 text/clinical scoring and the vendored NLTK3.6.7 METEOR source are reused,
with the restored NLTK3.3 runtime and unchanged WordNet ZIP.

## 4. Environment

ict11, Quadro RTX4000, 8192MiB, driver550.163.01/CUDA12.4. Qwen env remains isolated:
Python3.11, torch2.6+cu124, Transformers4.51.3, PEFT0.15.2, Accelerate1.6.0.
CPU labeler: Python3.6.7, NumPy1.15.4, pandas0.23.4, NLTK3.3. Public missing assets
and pinned MKL2019.1/OpenMP/libffi libraries were restored to local model cache;
no existing package was upgraded. `labeler_assets_summary.json` records hashes.
Legacy library search paths are passed only to CPU children. Labeling uses16
processes on a16-CPU affinity mask, below the32-worker limit. Cache/output stays
on the shared storage volume; filesystem free space is not proof of personal quota.

## 5. Dependency Setup

The legacy env was restored independently; the task supplied missing public parser,
CoreNLP3.5.2, punkt/tagset and pinned dynamic libraries. Synthetic-labeler import and
one artificial sentence passed before any Findings labeling. No patient data was
sent to a model API, cloud notebook, package host or other external service.

## 6. Code / Notebook Changes

New `research/rrg_f1/` cohort/resume, labeler, model/data, extractor, train, inference,
evaluation, interventions, supervisor, status, tests and aggregate reporting.
F0 source helpers were factored for reuse while preserving default interfaces;
old raw F0/P4/P5 runs, checkpoints and reports were not edited. Private F1 outputs
and local path configuration are ignored. Only reviewed public source/config/docs
and aggregate evidence may be committed; never push from this task.

## 7. Experiment Configuration

Cohort: {cohort.get('train_studies','N/A')} official-train studies, seed42, nonempty
Findings, one first available AP/PA image/study, patient overlap with P4 val=0.
Selection is uniform over eligible studies via a seeded permutation, not tuned on
validation. Exact official split/metadata linkage is checked. Clinical manifests,
raw Findings and labels remain private; public manifest summary contains counts/hashes.
MIMIC image content was treated as read-only; the F1 manifest does not constitute a
full pixel-content rehash. P4 Indiana expected image hashes are checked by evaluation.

BCE retains4 raw states. Positive=1; Negative/Blank=0; Uncertain excluded. Label columns
are name-aligned because Lung Opacity/Lung Lesion orders differ between F0 and labeler.
Train-only prior: (positives+1)/(non-uncertain+2); add log(p/(1-p)), tau1, to BCE logits.
BCE head reads mean projected tokens so its gradient reaches the projector.

A2: image projector + rank16 LoRA, no finding tokens/BCE. A3: A2+BCE. A4: A3+14 tokens
from image-only CLEAR zero-shot scores. Same seed/projector initialization/order and
hyperparameters, frozen FP16 Qwen/FP32 trainables, non-reentrant checkpointing.
Microbatch4, accumulation up to8 (max effective32), AdamW1e-4/eps1e-6/weight_decay0.01,
48-update warmup then cosine over960 planned updates. Flush at1000 samples and epoch end.
Findings target cap256 including EOS, text padding320. Greedy generation cap256, thinking off.
CLEAR448 input -> native1024 patches -> disk256 FP16 -> load64. F0 MIMIC val features are
reused; new train/Indiana extraction batch4 with8 image-loader workers.

Every1000 samples: first8 MIMIC + first8 Indiana dev reports, diversity/cap/repeated4gram
logged. Two consecutive bad checks (unique<4 OR cap>=12) stop the arm. Checkpoint saves
include every2000 samples and dev gates; optimizer/scaler/RNG/position and code hashes
are verified at resume. No additional epoch/arm or automatic OOM batch fallback.

## 8. Code Executed

Exact commands and resume instructions: `research/rrg_f1/README.md`.
Detached `nohup`/`setsid` supervisor uses a process lock, heartbeat and independent
GPU watchdogs: cumulative50400s (14h), each training arm cumulatively14400s (4h).
Loading/I/O/retries inside GPU children count toward the cap. Reservations survive
an interrupted guard conservatively; the ledger must never be reset to gain time.

## 9. Output

### Stage wall times

{chr(10).join(times)}

200-report labeler projected time for30k: {fmt(bench.get('projected_30000_seconds'),1)}s.
GPU charged total: **{fmt(total,1)}s /50400s = {fmt(total/3600,3)}h**.
Authoritative budget is the private GPU ledger; supervisor wall times include heartbeat
polling overhead. Raw clinical outputs, case labels/features/checkpoints and worker
logs stay in `research/rrg_f1/private/`.

## 10. Output Analysis

{chr(10).join(train)}

Training loss is not clinical quality. Last-step values can be absent after a terminal
checkpoint recovery; raw optimizer logs remain private. Official full-report label
audit covers1000 studies, overall four-state agreement={fmt(labels.get('official_full_report_audit',{}).get('overall_four_state_agreement'))}.
This compares different text sections and can be dominated by blank agreement;
it is an audit only and never replaces Findings-specific labels.

## 11. Quantitative Results

{chr(10).join(tables)}

M0/M1 are the saved P4 BLIP-2 baselines; M2 is historical CXR-LLaVA on Indiana only.
`metrics.json` contains all per-class confusion counts, both-policy coverage/support,
F1 denominators, macro/micro F1, bootstrap intervals and paired differences. Undefined
F1 remains null/N/A. Clinical evaluation follows P4's7-class ontology, distinct from
14-label training. All500 outputs/dataset are retained; empty outputs score zero text
and blank clinical labels rather than being favorably dropped.

## 12. Qualitative Results

No patient images, reference text or generated examples are published here. Outputs
remain experimental. No expert image/report review or clinical correctness assessment.

### Collapse monitoring

| Arm | Samples | Unique /16 | Hit256 /16 | Repeated4gram fraction |
|---|---:|---:|---:|---:|
{chr(10).join(checks) if checks else '| N/A | N/A | N/A | N/A | N/A |'}

## 13. Comparison with Baseline

A2/A3/A4 share training cohort/order and hyperparameters. Comparisons to P4 or CXR-LLaVA
also change model, training scale, prompt and generation cap. MIMIC evaluation references
remain P4 Findings+Impression for direct scoring, while F1 trains Findings only. These
are exposed dev cohorts; Indiana baseline exposure and pretraining overlap remain.
The1000 patient-cluster bootstrap draws are paired across models, with unadjusted
percentile95% intervals; secondary contrasts are exploratory, not confirmatory claims.

A4 controls: first50 MIMIC + first50 Indiana, scores all0 and a seed42 shuffle across100.
No GT was used to define or prompt these controls. Direction concordance uses only
nonzero changes in CLEAR scores and definite generated-label changes; zero-shot scores
are uncalibrated proxies and this is not clinical causal validation.

{chr(10).join(intervention_lines) if intervention_lines else 'Intervention results unavailable / not executed.'}

## 14. Problems Encountered

Initial restored env lacked parser and dynamic libraries; repaired with matching public
assets before actual labeling. Original reports/run outputs were preserved. Any stop
reason, incomplete stage or collapsed arm is explicit above; no synthetic numbers or
post-hoc threshold/epoch selection fill missing results. Storage quota remains unknown.

## 15. Conclusion

{execution.get('status','UNKNOWN')}: {classification}. The experiment does not establish
clinical usefulness, accuracy on an independent test set or a benchmark improvement.

## 16. Next Experiment

Any additional arm, epoch, test evaluation or changed collapse rule needs a new scoped
work order and budget. Do not infer permission from this dev report or checkpoint.

## 17. Reproducibility Checklist

Private protocol lock pins scientific code/config; public config/aggregate hashes are
retained. CPU suite14/14 and root catalog2/2 passed before launch, as did syntax/import,
synthetic labeler and pinned WordNet checks. Runtime data/case outputs never enter Git.
Commit staging must be inspected and scanned for case identifiers and report text;
only public artifacts are committed on the current branch, with the requested coauthor
footer. No push.

## 18. Short Lab Update

Cohort30k and labeler gate prepared under WO_F1. Execution status={execution.get('status','UNKNOWN')};
GPU={fmt(total/3600,3)}h of14h. Read the arm table and fixed-dev metrics above, with
collapse/coverage/protocol limits preserved. Research only; no clinical claim.
'''


def main():
    status=read(BASE/'execution_summary.json').get('status')
    if status not in ('COMPLETE','STOP'):
        raise RuntimeError('Wait for a terminal execution state before writing report')
    path=ROOT/'reports/2026-11-13_F1_clear_tokens_qwen_30k.md'
    if path.exists():
        raise FileExistsError('Preserve existing F1 report')
    with path.open('x') as stream:
        stream.write(render())
    print('Created aggregate F1 report; no clinical rows emitted')


if __name__ == '__main__':
    main()
