# P4 progress

Google Docs `VLM Medical` decision tab `t.ugcbxnenhjxv` and progress subtab `t.og7ix2n9u1zd` now contain the final P4 scientific interpretation, full-cohort metrics, clinical F1 sensitivity, text-based error analysis and P5 HOLD recommendation (synced 2026-10-25). The progress tab includes `reports/figures/p4_f1_sensitivity_20261025.png`; earlier 474-pair PARTIAL material is retained for provenance. Sync evidence is in `runs/P4_projection_pilot_20261023/google_docs_sync.json`.

## P4 final scientific closeout — 2026-10-25

**Execution PASS; lexical improvement supported; clinical improvement unproven.** CPU-only final analysis used existing artifacts. P4.0 scientific approval remains APPROVED. P4.1: 20 optimizer updates, 71.499 s cumulative, CUDA peak reserved 7.953 GiB (allocated 7.568); gradients, frozen parameters, optimizer updates and reload passed. P4.2: 2000 unique official MIMIC train studies, 2000 updates, one epoch, 4944.690 s, original pretrained projection initialization, final adapter reload/hash verified. First/last 100 logged losses averaged 2.969/1.961; 388 targets were truncated to the approved 128-token cap. P4.3: 500 unique MIMIC validation and 500 unique Indiana studies, 2,000 valid M0/M1 generations, 15243.812/18,000 cumulative GPU s. The CPU integrity audit checked manifest order, study/image/reference linkage, X00 saved output joins, no missing or duplicate records, labeler text alignment and SHA-256. Both BLIP-2 arms used the same approved greedy prompt, preprocessing, EOS=2 and max_new_tokens=128. P4.3 remains a metric-complete research evaluation, not clinical validation.

### Matched results (500 studies per cohort)

| Cohort | Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR | Hist micro-F1 | Alt micro-F1 | Hist definite labels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MIMIC | M0 | 0.0893 | 0.0202 | 0.0076 | 0.0000 | 0.0899 | 0.0688 | 0.2167 | 0.0340 | 107/3500 |
| MIMIC | M1 | 0.1756 | 0.0935 | 0.0554 | 0.0346 | 0.1663 | 0.1399 | 0.4351 | 0.1372 | 353/3500 |
| INDIANA | M0 | 0.0643 | 0.0222 | 0.0101 | 0.0031 | 0.0852 | 0.0885 | 0.1839 | 0.0556 | 79/3500 |
| INDIANA | M1 | 0.2288 | 0.1291 | 0.0820 | 0.0515 | 0.2095 | 0.2124 | 0.1045 | 0.0488 | 463/3500 |
| INDIANA | M2 | 0.2954 | 0.1488 | 0.0824 | 0.0448 | 0.2084 | 0.2365 | 0.5448 | 0.3883 | 867/3500 |

The Indiana M2 row is saved historical CXR-LLaVA X00 output on the same 500 study/image/reference IDs, but its model protocol differs; it is descriptive. Primary lexical comparisons are M1−M0 within each cohort. Patient-cluster bootstrap for MIMIC (121 subjects) and study-cluster bootstrap for Indiana (patient IDs unavailable) used 1,000 paired resamples, seed 42:

- MIMIC: Δ BLEU-4 M1−M0 = 0.034638, paired 95% CI [0.027476, 0.045860]; historical Δ micro-F1 = 0.218480 (CI [-0.0023342532400833617, 0.4186826642901172]); alternative Δ micro-F1 = 0.103216 (exploratory, no confirmatory claim).
- INDIANA: Δ BLEU-4 M1−M0 = 0.048423, paired 95% CI [0.043004, 0.054389]; historical Δ micro-F1 = -0.079430 (CI [-0.20871354049508872, 0.043963621389866744]); alternative Δ micro-F1 = -0.006775 (exploratory, no confirmatory claim).

The MIMIC and Indiana BLEU-4 intervals exclude zero. This supports better lexical overlap on both fixed source-order cohorts, including Indiana external to the current fine-tuning. It does not establish generalization to unrelated patients, sites or pretraining-unseen images. All 1,000 M0 outputs reached the 128-token cap; M1 did so on 113/500 MIMIC and 94/500 Indiana. Exact report-text diversity and latency are in `P4.3_final_integrity_audit.json`.

### Clinical policy and per-class evidence

Primary historical policy excludes GT or prediction blank/uncertain labels; the alternative retains GT-definite 0/1 cases and maps prediction blank/uncertain to negative. GT labels never change. The historical masks differ sharply by arm (MIMIC 107 vs 353 jointly definite labels; Indiana 79 vs 463); alternative coverage is identical across arms within each cohort (MIMIC 1,439/3,500, Indiana 1,482/3,500 class-cases). MIMIC alternative micro-F1 is M0 0.0340, M1 0.1372; Indiana is M0 0.0556, M1 0.0488, M2 0.3883. Thus M1's automated clinical F1 gain is not consistent across datasets or policies. Undefined F1 remains N/A. Column `covered/500` is per-class study coverage; blank/uncertain columns give two percentages of the 500 reports. Exact counts, rates and confusion for both policies are in `P4.3_final_scientific_analysis.json`. **Measurement correction:** the original full `metrics.json` compared labeler strings to literal `-1` when counting uncertain labels, while the CSV stores `-1.0`; its `reference_uncertain`/`generated_uncertain` counts are therefore understated. The corrected counts/rates in this table and the final analysis JSON were recalculated from immutable labeler output. Historical confusion and F1 were independently reproduced exactly because their label parser already converts values numerically. The original metrics file is preserved unchanged.

| Cohort | Arm | Class | Hist TP/FP/FN/TN | Hist P/R/F1 | Hist support; covered/500 | Alt TP/FP/FN/TN; F1 | GT blank/uncertain % | Pred blank/uncertain % |
|---|---|---|---|---|---|---|---|---|
| MIMIC | M0 | Cardiomegaly | 2/1/0/0 | 0.6667/1.0000/0.8000 | 2; 3/500 | 2/1/153/62; 0.0253 | 52.4/4.0 | 97.0/0.0 |
| MIMIC | M0 | Consolidation | 0/0/0/0 | N/A/N/A/N/A | 0; 0/500 | 0/0/33/103; 0.0000 | 68.8/4.0 | 100.0/0.0 |
| MIMIC | M0 | Edema | 0/0/0/0 | N/A/N/A/N/A | 0; 0/500 | 0/0/72/111; 0.0000 | 58.6/4.8 | 100.0/0.0 |
| MIMIC | M0 | Lung Opacity | 0/0/0/0 | N/A/N/A/N/A | 0; 0/500 | 0/0/169/10; 0.0000 | 62.4/1.8 | 100.0/0.0 |
| MIMIC | M0 | Pleural Effusion | 1/2/0/0 | 0.3333/1.0000/0.5000 | 1; 3/500 | 1/2/169/182; 0.0116 | 26.8/2.4 | 99.4/0.0 |
| MIMIC | M0 | Pneumonia | 1/0/0/0 | 1.0000/1.0000/1.0000 | 1; 1/500 | 1/0/39/46; 0.0488 | 73.4/9.4 | 98.2/0.0 |
| MIMIC | M0 | Pneumothorax | 9/91/0/0 | 0.0900/1.0000/0.1651 | 9; 100/500 | 9/91/10/173; 0.1513 | 42.8/0.6 | 48.0/12.6 |
| MIMIC | M1 | Cardiomegaly | 31/12/22/13 | 0.7209/0.5849/0.6458 | 53; 78/500 | 31/12/124/51; 0.3131 | 52.4/4.0 | 64.6/0.6 |
| MIMIC | M1 | Consolidation | 0/0/0/0 | N/A/N/A/N/A | 0; 0/500 | 0/0/33/103; 0.0000 | 68.8/4.0 | 100.0/0.0 |
| MIMIC | M1 | Edema | 1/0/4/8 | 1.0000/0.2000/0.3333 | 5; 13/500 | 1/0/71/111; 0.0274 | 58.6/4.8 | 92.0/0.2 |
| MIMIC | M1 | Lung Opacity | 0/0/0/0 | N/A/N/A/N/A | 0; 0/500 | 0/0/169/10; 0.0000 | 62.4/1.8 | 100.0/0.0 |
| MIMIC | M1 | Pleural Effusion | 19/34/43/67 | 0.3585/0.3065/0.3304 | 62; 163/500 | 19/34/151/150; 0.1704 | 26.8/2.4 | 58.0/0.0 |
| MIMIC | M1 | Pneumonia | 1/1/7/6 | 0.5000/0.1250/0.2000 | 8; 15/500 | 1/1/39/45; 0.0476 | 73.4/9.4 | 88.4/0.0 |
| MIMIC | M1 | Pneumothorax | 0/1/11/72 | 0.0000/0.0000/0.0000 | 11; 84/500 | 0/1/19/263; 0.0000 | 42.8/0.6 | 73.4/0.0 |
| INDIANA | M0 | Cardiomegaly | 5/7/0/0 | 0.4167/1.0000/0.5882 | 5; 12/500 | 5/7/71/243; 0.1136 | 33.2/1.6 | 95.6/0.0 |
| INDIANA | M0 | Consolidation | 0/0/0/0 | N/A/N/A/N/A | 0; 0/500 | 0/0/13/146; 0.0000 | 67.8/0.4 | 100.0/0.0 |
| INDIANA | M0 | Edema | 0/0/0/0 | N/A/N/A/N/A | 0; 0/500 | 0/0/3/41; 0.0000 | 89.4/1.8 | 100.0/0.0 |
| INDIANA | M0 | Lung Opacity | 0/0/0/0 | N/A/N/A/N/A | 0; 0/500 | 0/0/83/123; 0.0000 | 56.6/2.2 | 100.0/0.0 |
| INDIANA | M0 | Pleural Effusion | 0/10/0/0 | 0.0000/N/A/0.0000 | 0; 10/500 | 0/10/17/347; 0.0000 | 23.4/1.8 | 97.6/0.0 |
| INDIANA | M0 | Pneumonia | 2/1/0/0 | 0.6667/1.0000/0.8000 | 2; 3/500 | 2/1/9/20; 0.2857 | 91.8/1.8 | 97.0/0.0 |
| INDIANA | M0 | Pneumothorax | 1/53/0/0 | 0.0185/1.0000/0.0364 | 1; 54/500 | 1/53/5/282; 0.0333 | 30.8/1.0 | 83.0/4.0 |
| INDIANA | M1 | Cardiomegaly | 6/9/34/107 | 0.4000/0.1500/0.2182 | 40; 156/500 | 6/9/70/241; 0.1319 | 33.2/1.6 | 53.6/0.2 |
| INDIANA | M1 | Consolidation | 0/0/0/0 | N/A/N/A/N/A | 0; 0/500 | 0/0/13/146; 0.0000 | 67.8/0.4 | 100.0/0.0 |
| INDIANA | M1 | Edema | 0/1/0/2 | 0.0000/N/A/0.0000 | 0; 3/500 | 0/1/3/40; 0.0000 | 89.4/1.8 | 90.4/0.0 |
| INDIANA | M1 | Lung Opacity | 0/0/0/0 | N/A/N/A/N/A | 0; 0/500 | 0/0/83/123; 0.0000 | 56.6/2.2 | 100.0/0.0 |
| INDIANA | M1 | Pleural Effusion | 1/60/11/127 | 0.0164/0.0833/0.0274 | 12; 199/500 | 1/60/16/297; 0.0256 | 23.4/1.8 | 44.0/0.0 |
| INDIANA | M1 | Pneumonia | 0/0/3/3 | N/A/0.0000/0.0000 | 3; 6/500 | 0/0/11/21; 0.0000 | 91.8/1.8 | 88.6/0.0 |
| INDIANA | M1 | Pneumothorax | 0/1/1/97 | 0.0000/0.0000/0.0000 | 1; 99/500 | 0/1/6/334; 0.0000 | 30.8/1.0 | 67.8/0.0 |
| INDIANA | M2 | Cardiomegaly | 38/42/18/93 | 0.4750/0.6786/0.5588 | 56; 191/500 | 38/42/38/208; 0.4872 | 33.2/1.6 | 40.4/0.6 |
| INDIANA | M2 | Consolidation | 1/12/1/41 | 0.0769/0.5000/0.1333 | 2; 55/500 | 1/12/12/134; 0.0769 | 67.8/0.4 | 69.4/0.4 |
| INDIANA | M2 | Edema | 1/1/0/8 | 0.5000/1.0000/0.6667 | 1; 10/500 | 1/1/2/40; 0.4000 | 89.4/1.8 | 65.0/2.2 |
| INDIANA | M2 | Lung Opacity | 19/6/2/6 | 0.7600/0.9048/0.8261 | 21; 33/500 | 19/6/64/117; 0.3519 | 56.6/2.2 | 85.8/0.6 |
| INDIANA | M2 | Pleural Effusion | 11/24/3/284 | 0.3143/0.7857/0.4490 | 14; 322/500 | 11/24/6/333; 0.4231 | 23.4/1.8 | 12.6/1.0 |
| INDIANA | M2 | Pneumonia | 2/1/1/6 | 0.6667/0.6667/0.6667 | 3; 10/500 | 2/1/9/20; 0.2857 | 91.8/1.8 | 66.2/3.2 |
| INDIANA | M2 | Pneumothorax | 1/8/3/234 | 0.1111/0.2500/0.1538 | 4; 246/500 | 1/8/5/327; 0.1333 | 30.8/1.0 | 30.6/0.2 |

### Text-based error candidates and limitations

| Cohort | Proxy | M0 cases | M1 cases | M1 introduced / resolved / persistent |
|---|---|---:|---:|---|
| MIMIC | hallucination | 94 | 47 | 38 / 85 / 9 |
| MIMIC | omission | 334 | 322 | 5 / 17 / 317 |
| MIMIC | incorrect_negation | 0 | 34 | 34 / 0 / 0 |
| MIMIC | generic | 380 | 265 | 64 / 179 / 201 |
| INDIANA | hallucination | 71 | 69 | 56 / 58 / 13 |
| INDIANA | omission | 150 | 149 | 4 / 5 / 145 |
| INDIANA | incorrect_negation | 0 | 8 | 8 / 0 / 0 |
| INDIANA | generic | 436 | 273 | 39 / 202 / 234 |

These are automated-label/text-heuristic candidate counts on identical studies, not radiologist-confirmed hallucination, omission or negation errors. MIMIC hallucination proxies fell 94→47 and omission proxies 334→322; Indiana changed 71→69 and 150→149. Meanwhile the incorrect-negation proxy was introduced in 34 MIMIC and 8 Indiana M1 cases. A same-pathology positive-plus-negative cue occurred within 22/500 M1 reports on each cohort, versus 0/500 M0; the cue rule may misread scope or temporal comparisons. M2 Indiana descriptive proxy counts and transitions by pathology are in `P4.3_final_error_pairing.json`. The deterministic review sample contains 157 disagreement/control candidates plus 110 paired-transition/intra-report candidates with case index and hashes only; no patient ID or report text was exported. Certainty, laterality, severity, genericity and report repetition were screened; the regex candidates require blinded image/report adjudication.

Scientific limits: MIMIC validation is a development cohort; both pretrained models may have seen MIMIC/Indiana related material. Cohorts were selected in source order, not randomly. CheXpert/NegBio labels are imperfect and had nine nonfatal parse-tree warnings. Many generated labels are blank, making historical F1 sparse and model masks incomparable. M0's systematic 128-token censoring and repetitive QA text bias the lexical comparison. Secondary intervals and error categories were not multiplicity-adjusted. No image review or radiologist adjudication occurred. No claim of reduced clinical hallucinations or improved patient safety is supported.

Evidence: `runs/P4_projection_pilot_20261023/P4.3_evaluation_full/metrics.json` SHA-256 ed6be843294d14deafa1fd9c9366bf302a53c08f12f6804c530e680393fb9be5; `P4.3_final_scientific_analysis.json` SHA-256 ab30c506b6ccf7d10d8cec3d3a36c4ccd15167e7924b1a51381c33a4ff24799e; `P4.3_final_error_pairing.json`; `P4.3_final_integrity_audit.json`; config SHA-256 d010d2cf32cb6cc9b96b834e5eb0a5df1d8bd13985ec31cf5faa447260471eb9; manifest index SHA-256 74965a79b31394a65af10dcb7987db0881e5083a8bb0da888bccbe120bd03175; P4.2 checkpoint SHA-256 7f833eea69fb6d2743b19bbc808be949bb1a5e014bf43a81f08090f9a5f0d8af. Earlier PARTIAL/RUNNING snapshots below are historical and preserved. P5 execution remains HOLD pending separate approval.


## P4.3 full-cohort continuation — 2026-10-25

RUNNING under the user's instruction to complete P4.3. An auditable resource amendment raises the cumulative one-K80 inference cap from 7,200 to 18,000 seconds, retaining all 7,141.085 seconds already charged. The approved scientific config, frozen manifests, checkpoint and 474 saved paired outputs retain their original hashes. One resumed inference job continues the exact remaining prefix (31 MIMIC validation and 495 Indiana studies); GPU0 PID 1170123, attempt `cf3a908da5ca4fdb90aa142f6f4bb0dd`, watchdog 10,820 seconds plus 20 seconds kill grace. The full CPU evaluation handoff waits for PASS on exactly 500+500 paired studies and writes to `P4.3_evaluation_full`, preserving the earlier partial metrics. An independent status-sync watcher will update this report and the tracker after the evaluation reaches a terminal status. Do not read this launch as a completed experiment.

Evidence: `runs/P4_projection_pilot_20261023/P4.3_budget_amendment_20261025.json`, `P4.3_resume_20261024T214726Z_launch.json`, `P4.3_resume_full_handoff_status.json`, `P4.3_full_status_sync_status.json`.

Google Docs: partial P4.3 results, 17-row native metrics table and aggregate figure were synced to the `VLM Medical` child tab `t.og7ix2n9u1zd` under `Report 2/11` for 19/10–02/11. Full-cohort Docs update remains pending; sync target and trigger are recorded in `runs/P4_projection_pilot_20261023/google_docs_sync.json`. The supplied meeting tab `t.0vafplan` was absent; the latest available `Meeting Note & Log` entry dated 19/10 was reviewed.

## P4.3 final bounded evaluation — 2026-10-24

P4.0 APPROVED, P4.1 PASS and P4.2 PASS remain verified. P4.3 is PARTIAL after the fixed 7,200 s cumulative GPU inference budget: 474/1,000 valid paired studies (469/500 MIMIC validation, 5/500 Indiana), 948/2,000 M0/M1 generations. Actual GPU time was 7,141.085 s. The 10-study output-validity gate passed. No further GPU inference was launched.

CPU evaluation completed successfully on the saved paired intersection after a routine config-Path read fix. `P4.3_evaluation/metrics.json` SHA-256 `04ebf1c0f670985dbd6d7f70f8e9651849e1a0d1d40f7656156c4074bec06628`; `P4.3_retry2_evaluation_status.json` is the recovery status. MIMIC corpus BLEU-4 was 0 (M0) and 0.033834 (M1), paired patient-cluster bootstrap delta 95% CI [0.026367, 0.045306], 1,000 replicates. Indiana has only five paired cases; no benchmark conclusion. Historical clinical micro-F1 on MIMIC was 0.2162 (M0; 99 jointly definite labels, 12 positive support) and 0.4424 (M1; 322 labels, 127 support). Differing masks/support prevent a clean clinical-improvement claim. Labeler logged seven nonfatal parse-tree warnings. M0 hit the 128-token cap on all 474 outputs; frozen source order caused strong completion bias toward MIMIC. Hallucination/omission proxy counts are available per class, but no separate contradiction analysis was produced. Google Docs sync remains pending.

The overall P4 status is PARTIAL. Full 500+500 evaluation would require a new approved GPU budget; P5/P6 execution remains unapproved. Earlier RUNNING/FAIL entries below are historical snapshots, not the current stage status.

## P4.3 compatibility retry — 2026-10-24

RUNNING; ten-study/two-model validity gate PASS (20 nonempty generations). Scoped single-sequence EOS stopping preserves approved EOS2 and greedy output; actual tiny OPT CPU equivalence checks cover EOS2, checkpoint-default EOS, maximum length, restoration and BOS continuation counting. Four focused CPU tests passed. No environment or frozen scientific config changed. Mean gate generation latency 6.635 s; forecast 2,000 generations 13,269.38 s exceeds the fixed 7,200 s cumulative budget. Continue only within remaining budget and score the completed paired intersection as PARTIAL if incomplete. Preserve frozen cohort order and report per-dataset sample counts; partial coverage may be very uneven. No clinical quality result yet.

Retry evidence: `P4.3_retry_launch.json`, `P4.3_retry_cpu_tests.log`, `P4.3_status.json` and existing cumulative `P4.3_budget.json` under the run directory. Original failed logs remain preserved.


## P4 verified resume — 2026-10-24

- P4.0 SCIENTIFIC_APPROVAL=APPROVED. Config/index/manifest hashes unchanged; 18 CPU tests passed before execution. Prior pending-approval entries are historical.
- P4.1 PASS: 20 updates, 71.50/900 s; allocated peak 7.568 GiB, reserved 7.953 GiB. Gradients, updates, frozen parameters and reload checks passed. Conservative 2× maximum-step forecast: 10,035 s for 2,000 updates, below four hours.
- P4.2 PASS: 2,000 studies and updates, one epoch, fresh original pretrained projection; 4,944.69/14,400 s (82.41 minutes). Final checkpoint reload verified; actual SHA-256 `7f833eea69fb6d2743b19bbc808be949bb1a5e014bf43a81f08090f9a5f0d8af` matches `P4.2_last.pt`. Existing executed results verified this session, no training rerun.
- P4.3 PARTIAL: CUDA error at greedy EOS reduction before any generations. Validity gate PENDING. Charged 43.04/7,200 s; remaining 7,156.96 s. Previous processes exited. Scoped equivalent compatibility repair and CPU tests precede any retry; preserve ledger and old logs.
- No evaluation metrics or clinical benefit established. GOOGLE_DOCS_SYNC_PENDING.

Evidence: `runs/P4_projection_pilot_20261023/` stage status, budget, update logs and checkpoint. P0–P3 historical statuses unchanged.


## Historical snapshot (2026-10-23)

- Run ID: `P4_projection_pilot_20261023`.
- **P4.0 scientific approval: APPROVED.** Config SHA-256 `d010d2cf32cb6cc9b96b834e5eb0a5df1d8bd13985ec31cf5faa447260471eb9`; manifest index SHA-256 `74965a79b31394a65af10dcb7987db0881e5083a8bb0da888bccbe120bd03175`. Frozen source-order cohorts: 2,000 official MIMIC train, 500 official MIMIC validation and 500 Indiana matched to X00. See `runs/P4_projection_pilot_20261023/P4.0_execution_freeze.json` and `manifest_index.approved.json`.
- **P4.1 PASS:** 20/20 updates, 71.50/900 GPU seconds charged, peak CUDA allocated 7.568 GiB and reserved 7.953 GiB. Gradient, frozen-parameter and adapter reload checks passed. Mean 2.167 s/update, max 2.509 s/update. A 2× max-step 2,000-update estimate is 10,035 s, leaving 4,365 s for data/checkpoints under the 14,400 s cap. See `P4.1_verdict.json` and `P4.1_budget.json`.
- **P4.2 RUNNING:** one K80 GPU job, fresh original pretrained projection, 2,000 unique train studies in one seeded epoch. Cumulative 4-hour ledger, resume checkpoints and independent timeout active. Last verified checkpoint: 200/2,000 updates. Consult `P4.2_status.json` and `P4.2_budget.json` for live state. No final training result claimed.
- **P4.3 ARMED/WAITING:** deterministic handoff verifies P4.2 supervisor exit, 2,000 updates, one epoch, final reload/hash and budget before paired inference. Maximum 2,000 M0/M1 generations and two GPU hours; CPU evaluation follows. M2 reuses saved X00 Indiana outputs. Consult `P4.3_handoff_status.json`. No P4.3 GPU job is active at this snapshot.
- 18 targeted P4 CPU tests passed. P0–P3 claims unchanged. This source-order convenience pilot is not a representative sample or clinical validation. Google Docs sync remains pending.

`runs/P4_projection_pilot_20261023/status.json` is the tracking snapshot. Stage-specific JSON and budget files are the live records for detached execution.

Detached continuation verified: inference worker PID 1059865 under watchdog PID 1059863; CPU evaluation handoff `P4.3_retry_handoff_status.json` is WAITING_FOR_INFERENCE. It requires terminal PASS/PARTIAL, validity PASS, matching config/checkpoint hashes and pair counts, and refuses overwriting metrics. Stage JSON files are live authority; root status/report is a session snapshot. Evaluation has not completed.

## P4.3 bounded inference ended; CPU evaluation recovery

Inference PARTIAL: 474 unique paired studies (469 MIMIC validation, 5 Indiana), 948 completed generations, no empty M0/M1 output. Watchdog stopped retry; cumulative GPU wall time 7,141.085/7,200 s, remaining 58.915 s. No new GPU run. Initial CPU evaluation failed before labeling because a config Path was indexed as a dict; ordinary fix reads the approved contract. CPU retry is labeling 1,427 reference/generated texts with the pinned existing CheXpert/NegBio environment. Metrics pending; source of execution evidence: `P4.3_retry2_evaluate_stdout.log`, `P4.3_retry2_evaluate_stderr.log`, `P4.3_evaluation/labeler.log`.

Descriptive output limitations: all 474 M0 reports hit the 128-token cap; M1 hit the cap on 109/469 MIMIC and 1/5 Indiana reports. Exact unique text counts across completed outputs: M0 57/474 (largest repeated group 120), M1 294/474 (largest repeated group 78). Record IDs are unique. Content repetition is distinct from duplicate records. Indiana n=5 is exploratory; source-order budget truncation creates uneven cohort coverage. Parser warnings and final missing/uncertain coverage must accompany any clinical metrics.

## P4.3 full-cohort terminal sync — 2026-10-25

P4.0 APPROVED; P4.1/P4.2 PASS; P4.3 inference and full CPU evaluation PASS on 1,000/1,000 ordered paired studies (500 MIMIC validation, 500 Indiana), with 15243.812/18,000 GPU seconds charged. Full metrics: `runs/P4_projection_pilot_20261023/P4.3_evaluation_full/metrics.json` SHA-256 `ed6be843294d14deafa1fd9c9366bf302a53c08f12f6804c530e680393fb9be5`. The earlier 474-pair partial metrics remain preserved.

MIMIC 500/500 valid pairs: BLEU-4 M0=0.0000 M1=0.0346, M1−M0=0.0346 (paired cluster 95% CI [0.0275, 0.0459]); historical labeler micro-F1 M0=0.2167 (107 definite labels, 13 positive support), M1=0.4351 (353 definite labels, 139 positive support).

INDIANA 500/500 valid pairs: BLEU-4 M0=0.0031 M1=0.0515, M1−M0=0.0484 (paired cluster 95% CI [0.0430, 0.0544]); historical labeler micro-F1 M0=0.1839 (79 definite labels, 8 positive support), M1=0.1045 (463 definite labels, 56 positive support).

These are research metrics on selected source-order cohorts. CheXpert/NegBio labels are automated; blank and uncertain labels are excluded from historical F1, and the definite-label masks/support differ by arm. Lexical overlap does not establish clinical correctness. Indiana M2 uses saved historical X00 outputs under a different model protocol. Report repetition and 128-token truncation limit interpretation; this is not clinical validation. Google Docs sync remains pending.
