# Experiment tracker — P0 inventory only

| Date | Phase | ID | Dataset / split / manifest | Model / checkpoint / repo | Env / device / run | Status | Metrics / CI / runtime / VRAM | Blocker / next action |
|---|---|---|---|---|---|---|---|---|
| 2026-10-22 | P1-A | X00 | Indiana reconstructed frontal cohort; 3,667; SHA in run manifest | CXR-LLaVA saved outputs; historical checkpoint revision in manifest | cxr-clear-cpu; CPU; X00_indiana_rescore_20261022T062016Z | PARTIAL | BLEU1-4 0.2879/0.1468/0.0818/0.0450; ROUGE-L 0.2079; historical macro-F1 0.4457; study CI in metrics.json; METEOR N/A | WordNet unavailable; patient key absent; Claude review before B |
| 2026-10-22 | P0 | X01 | Indiana;MIMIC; split/SHA UNKNOWN | CXR-LLaVA; checkpoint/SHA UNKNOWN | UNKNOWN; no run | PLANNED | N/A | P1 linkage/overlap and P3 asset audit; GPU/training authorization where applicable |
| 2026-10-22 | P0 | X02 | Indiana;MIMIC; split/SHA UNKNOWN | BLIP-2; checkpoint/SHA UNKNOWN | UNKNOWN; no run | PLANNED | N/A | P1 linkage/overlap and P3 asset audit; GPU/training authorization where applicable |
| 2026-10-22 | P0 | X03 | Indiana;MIMIC; split/SHA UNKNOWN | ClinicalBLIP; checkpoint/SHA UNKNOWN | UNKNOWN; no run | PLANNED | N/A | P1 linkage/overlap and P3 asset audit; GPU/training authorization where applicable |
| 2026-10-22 | P0 | X04 | Indiana;MIMIC; split/SHA UNKNOWN | R2-LLM; checkpoint/SHA UNKNOWN | UNKNOWN; no run | PLANNED | N/A | P1 linkage/overlap and P3 asset audit; GPU/training authorization where applicable |
| 2026-10-22 | P0 | X05 | Indiana;MIMIC;VinDr; split/SHA UNKNOWN | META-CXR; checkpoint/SHA UNKNOWN | UNKNOWN; no run | PLANNED | N/A | P1 linkage/overlap and P3 asset audit; GPU/training authorization where applicable |
| 2026-10-22 | P0 | X06 | Indiana;MIMIC; split/SHA UNKNOWN | RRG-LLM; checkpoint/SHA UNKNOWN | UNKNOWN; no run | PLANNED | N/A | P1 linkage/overlap and P3 asset audit; GPU/training authorization where applicable |
| 2026-10-22 | P0 | X07 | Indiana; split/SHA UNKNOWN | Swin-Qwen3; checkpoint/SHA UNKNOWN | UNKNOWN; no run | PLANNED | N/A | P1 linkage/overlap and P3 asset audit; GPU/training authorization where applicable |
| 2026-10-22 | P0 | X08 | Indiana;MIMIC; split/SHA UNKNOWN | CLEAR Q-Former proposal; checkpoint/SHA UNKNOWN | UNKNOWN; no run | PLANNED | N/A | P1 linkage/overlap and P3 asset audit; GPU/training authorization where applicable |
| 2026-10-22 | P0 | X09 | Indiana;MIMIC; split/SHA UNKNOWN | CXR Q-Former proposal; checkpoint/SHA UNKNOWN | UNKNOWN; no run | PLANNED | N/A | P1 linkage/overlap and P3 asset audit; GPU/training authorization where applicable |
| 2026-10-22 | P0 | X10 | Indiana; split/SHA UNKNOWN | CLEAR evidence prompts; checkpoint/SHA UNKNOWN | UNKNOWN; no run | PLANNED | N/A | P1 linkage/overlap and P3 asset audit; GPU/training authorization where applicable |
| 2026-10-22 | P0 | X11 | VinDr;CheXpert;NIH; split/SHA UNKNOWN | Pathology benchmarking; checkpoint/SHA UNKNOWN | UNKNOWN; no run | PLANNED | N/A | P1 linkage/overlap and P3 asset audit; GPU/training authorization where applicable |

## P1 part checkpoints

| Part | Date | Status | Evidence | Blocker / next |
|---|---|---|---|---|
| B MIMIC | 2026-10-22 | PARTIAL | 377,110 metadata rows; 50/50 sampled reports; 48/50 sampled JPGs; repository path testset mostly official train | Version/coverage and pretraining exposure unresolved; proceed to C |
| C VinDr | 2026-10-22 | PARTIAL | 15,000 train images, 3,000 test, 28 label classes, no overlap; evaluator dry-run/tests PASS | Continuous scores and validation thresholds absent; proceed to D |
| D NIH/CheXpert | 2026-10-22 | PASS | NIH local 4,999 exploratory images; CheXpert 668 locked test images / 500 studies | No CheXpert train/val; no test tuning; proceed to E |
| E BLIP-2 candidate | 2026-10-22 | PASS metadata/dry-run | Four Salesforce HF API metadata checks; OPT 2.7B selected files 13.9551 GiB; no transfer | Candidate approval and later runtime feasibility remain pending; no download |
| 2026-10-22 | P2.1 | E4 B1 Indiana 24 studies / 264 saved reports | C0 + 10 paired arms | CPU, 1000 study bootstrap, seed 42 | PASS exploratory | C0 BLEU4 .0371; historical macro-F1 .8063; METEOR pending P2.5 | runs/P2_e4_paired/metrics.json |
| 2026-10-22 | P2.2 | MIMIC official validate 2,991 / test 5,159 image rows | Data coverage audit | CPU, read-only, bounded resume | PASS | JPG 2,809/2,991 validate; 4,904/5,159 test; all 5,077 study reports found; patient overlap 0 | manifests/mimic_official_coverage.json |
| 2026-10-22 | P2.3 | Pinned BLIP-2 OPT 2.7B local checkpoint | Config/processor only | CPU, offline, no weights instantiated | PARTIAL | Both shard SHA-256 PASS; 3.745B FP32 parameters; fast tokenizer fails, slow passes | manifests/blip2_preflight.json |
| 2026-10-22 | P2.5 | X00 Indiana 3,667 + E4 B1 24 studies | METEOR, NLTK WordNet | CPU, 1000 study bootstrap, seed 42 | PASS | X00 .236041; E4 C0 .211113; paired CIs saved | runs/X00_indiana_meteor_P2_20261022/metrics.json; runs/P2_e4_paired/metrics.json |
| 2026-10-22 | P2.6 | R2-LLM and META-CXR source repositories | Source audit only | CPU, no scripts run | PARTIAL | Pinned SHAs; R2 MIT, META root license UNKNOWN; checkpoint access/K80 UNKNOWN | reports/P2_repo_audit.md |
| 2026-10-22 | P2.4-prep | X00 Indiana deterministic 10/3,667 frontal sample | BLIP-2 OPT 2.7B offline smoke | Dry-run CPU only | PASS prep; GPU NOT_STARTED | 10/10 paths found; empirical FP16 probe added, GPU job NOT_STARTED | configs/experiments/X02_smoke.yaml; runs/X02_smoke_dry_run/status.json |
| 2026-10-22 | X02 smoke QA | Indiana 10-case sample | BLIP-2 OPT 2.7B v1 instruction prompt | Claude GPU0 run, peak 7.56 GB | FAIL output validity | 10/10 empty; root status corrected, files retained | runs/X02_smoke_20261022T0800Z/status.json |
| 2026-10-22 | X02 v2 repair | Indiana fixed 10-case sample | BLIP-2 QA prompt v2 | CPU dry-run only | PASS prep; GPU NOT_STARTED | Empty text now FAIL; evaluator gated after 10 valid cases | configs/experiments/X02_smoke.yaml; runs/X02_smoke_dry_run_v2/status.json |
| 2026-10-22 | X02 v2 final | Indiana fixed 10-case sample | BLIP-2 QA prompt v2 | Claude GPU0 generation; Codex CPU-only reevaluation | PARTIAL | 10/10 nonempty; 7.57 GB peak; 1.7406 s/case; BLEU4 0, ROUGE-L .070539, METEOR .036880; clinical F1 N/A (0 paired positive support); v1 FAIL retained | runs/X02_smoke_v2_20261022T0810Z/evaluation_v2/metrics.json |
| 2026-10-22 | P2.6 checkout completion | R2-LLM source SHA 2167adf5 | Source only | bounded git restore, no GPU | PASS checkout | 147 tracked files, clean worktree; weights/access still UNKNOWN | third_party/R2-LLM; reports/P2_repo_audit.md |

## P3.1 paper evidence

Six models classified with source-linked published metrics in `manifests/papers.csv` and `reports/03_paper_evidence_matrix.md`. Indiana/MIMIC pretraining or test overlap is unresolved; no published metric is a local benchmark. P3.2 asset/licensing decision pending.

| 2026-10-22 | P3.2 | X04/X05 asset audit | Indiana/MIMIC cohorts unresolved | R2/META metadata only | CPU/network GET | BLOCKED | No new metrics or GPU | HF R2 API 401; META Drive files/terms UNKNOWN; see P3_weights_licensing.md |

| 2026-10-22 | P3.3 | X06 projection/Q-Former/OPT-LoRA | MIMIC official split sample check + X00 external path | local BLIP-2 | CPU synthetic dry-run, no weights | REIMPLEMENTED_SCRIPT_READY; SCRIPT_READY_NOT_RUN | 3/3 dry-runs PASS; 3/3 unit tests PASS; VRAM/time heuristic only | Training approval and GPU smoke separate; see P3_implementation_audit.md |
| 2026-10-22 | P3.3 | X03/X04/X05/X07/X08/X09 | UNKNOWN or feature adapter unverified | official assets missing / local encoders | no model job | SKELETON_ONLY/BLOCKED | no metrics | X04 weights NOT_PUBLICLY_RELEASED; X08/X09 feature interface pending |

| 2026-10-22 | P3.4 | Legacy R2/META environment | N/A | official R2 source, no public official weights | import-only CPU | PLAN_ONLY/BLOCKED | no metrics | current env missing omegaconf; proposed install/rollback not executed; see P3_legacy_env_plan.md |

## P4.0 — configuration freeze preparation, 2026-10-23

| Phase | Status | Evidence | Next action |
|---|---|---|---|
| P4.0 | PARTIAL; BLOCKED pending scientific approval | Candidate studies 2000 train / 500 official validate / 500 Indiana; selected patient overlap 0; 14/14 CPU tests PASS; runs/P4_projection_pilot_20261023/status.json | Approve the single bundle in reports/P4_progress.md, then finish implementation and freeze before GPU |
| P4.1–P4.3 | PLANNED, conditional gates | No new GPU job; all cumulative budgets unspent | No launch before P4.0 PASS |

## P4 verified resume — 2026-10-24

- P4.0 SCIENTIFIC_APPROVAL=APPROVED. Config/index/manifest hashes unchanged; 18 CPU tests passed before execution. Prior pending-approval entries are historical.
- P4.1 PASS: 20 updates, 71.50/900 s; allocated peak 7.568 GiB, reserved 7.953 GiB. Gradients, updates, frozen parameters and reload checks passed. Conservative 2× maximum-step forecast: 10,035 s for 2,000 updates, below four hours.
- P4.2 PASS: 2,000 studies and updates, one epoch, fresh original pretrained projection; 4,944.69/14,400 s (82.41 minutes). Final checkpoint reload verified; actual SHA-256 `7f833eea69fb6d2743b19bbc808be949bb1a5e014bf43a81f08090f9a5f0d8af` matches `P4.2_last.pt`. Existing executed results verified this session, no training rerun.
- P4.3 PARTIAL: CUDA error at greedy EOS reduction before any generations. Validity gate PENDING. Charged 43.04/7,200 s; remaining 7,156.96 s. Previous processes exited. Scoped equivalent compatibility repair and CPU tests precede any retry; preserve ledger and old logs.
- No evaluation metrics or clinical benefit established. GOOGLE_DOCS_SYNC_PENDING.

Evidence: `runs/P4_projection_pilot_20261023/` stage status, budget, update logs and checkpoint. P0–P3 historical statuses unchanged.

## P4.3 compatibility retry — 2026-10-24

RUNNING; ten-study/two-model validity gate PASS (20 nonempty generations). Scoped single-sequence EOS stopping preserves approved EOS2 and greedy output; actual tiny OPT CPU equivalence checks cover EOS2, checkpoint-default EOS, maximum length, restoration and BOS continuation counting. Four focused CPU tests passed. No environment or frozen scientific config changed. Mean gate generation latency 6.635 s; forecast 2,000 generations 13,269.38 s exceeds the fixed 7,200 s cumulative budget. Continue only within remaining budget and score the completed paired intersection as PARTIAL if incomplete. Preserve frozen cohort order and report per-dataset sample counts; partial coverage may be very uneven. No clinical quality result yet.

Retry evidence: `P4.3_retry_launch.json`, `P4.3_retry_cpu_tests.log`, `P4.3_status.json` and existing cumulative `P4.3_budget.json` under the run directory. Original failed logs remain preserved.

## P4.3 bounded evaluation closeout — 2026-10-24

| Run | P4.0 | P4.1 | P4.2 | P4.3 | Evidence |
|---|---|---|---|---|---|
| P4_projection_pilot_20261023 | APPROVED | PASS: 20 updates, 71.499 s, peak reserved 7.953 GiB | PASS: 2,000 studies/updates, 4,944.690 s, final checkpoint verified | PARTIAL: 474/1,000 paired studies, 948/2,000 generations, 7,141.085/7,200 GPU s; CPU metrics completed | `runs/P4_projection_pilot_20261023/P4.3_retry2_evaluation_status.json`; `P4.3_evaluation/metrics.json` SHA-256 `04ebf1c0f670985dbd6d7f70f8e9651849e1a0d1d40f7656156c4074bec06628` |

Completed cohorts: MIMIC 469/500, Indiana 5/500. Existing M2 Indiana outputs were reused. Source-order truncation and labeler coverage limit interpretation; no full benchmark or clinical validation claim.

## P4.3 full-cohort terminal sync — 2026-10-25

P4.0 APPROVED; P4.1/P4.2 PASS; P4.3 inference and full CPU evaluation PASS on 1,000/1,000 ordered paired studies (500 MIMIC validation, 500 Indiana), with 15243.812/18,000 GPU seconds charged. Full metrics: `runs/P4_projection_pilot_20261023/P4.3_evaluation_full/metrics.json` SHA-256 `ed6be843294d14deafa1fd9c9366bf302a53c08f12f6804c530e680393fb9be5`. The earlier 474-pair partial metrics remain preserved.

MIMIC 500/500 valid pairs: BLEU-4 M0=0.0000 M1=0.0346, M1−M0=0.0346 (paired cluster 95% CI [0.0275, 0.0459]); historical labeler micro-F1 M0=0.2167 (107 definite labels, 13 positive support), M1=0.4351 (353 definite labels, 139 positive support).

INDIANA 500/500 valid pairs: BLEU-4 M0=0.0031 M1=0.0515, M1−M0=0.0484 (paired cluster 95% CI [0.0430, 0.0544]); historical labeler micro-F1 M0=0.1839 (79 definite labels, 8 positive support), M1=0.1045 (463 definite labels, 56 positive support).

These are research metrics on selected source-order cohorts. CheXpert/NegBio labels are automated; blank and uncertain labels are excluded from historical F1, and the definite-label masks/support differ by arm. Lexical overlap does not establish clinical correctness. Indiana M2 uses saved historical X00 outputs under a different model protocol. Report repetition and 128-token truncation limit interpretation; this is not clinical validation. Google Docs sync remains pending.

## P4 final scientific decision — 2026-10-25

| Run | Execution | Matched cohort and primary metric | Clinical sensitivity | Error review | Decision |
|---|---|---|---|---|---|
| P4_projection_pilot_20261023 | P4.0 APPROVED; P4.1/P4.2/P4.3 PASS; CPU closeout PASS | 500 MIMIC + 500 Indiana; BLEU-4 Δ M1−M0 +0.034638 CI [0.027476,0.045860] / +0.048423 CI [0.043004,0.054389], paired 1,000× seed 42 | Historical micro-F1 MIMIC 0.2167→0.4351, Indiana 0.1839→0.1045; GT-definite prediction-missing-negative 0.0340→0.1372 / 0.0556→0.0488; clinical benefit unproven | 267 deidentified text/labeler review candidates across two reproducible selections; no image adjudication | P4 technical PASS; lexical support only. P5 GPU HOLD; proposal ready for separate GO. |

Source artifacts: `runs/P4_projection_pilot_20261023/P4.3_evaluation_full/metrics.json`, `P4.3_final_scientific_analysis.json`, `P4.3_final_error_pairing.json`, `P4.3_final_integrity_audit.json`. Full per-class TP/FP/FN/TN, support, coverage, missing/uncertain rates and F1 are in the analysis JSON. Historical P4.3 PARTIAL entries above remain valid snapshots, not the terminal status.

## P5.0 CPU design and readiness — 2026-10-25

| Phase | Scope | Status | Verified evidence | Decision |
|---|---|---|---|---|
| P5.0 | BLIP-2 trainable Q-Former/query/projection design; matched P4 manifests; static CPU validation and resource/evaluation planning | PASS CPU DESIGN ONLY | P4 config/index/three cohort hashes, 2,000/500/500 unique studies/images and zero selected MIMIC train/validation patient overlap; checkpoint metadata count 107,130,880 trainables; small CPU wrapper gradient path PASS; legacy 100-step X06 trainer fails P4 parity | P5.1 GPU HOLD until new P5 executable contract, CPU tests and separate approval. P5.2/P5.3 also HOLD. |

Versioned design artifacts: `reports/P5_0_20261025/P5_architecture_spec.md`, `P5_matched_protocol.md`, `P5_cpu_static_validation.md`, `P5_resource_budget.md`, `P5_evaluation_plan.md`, `P5_approval_request.md`. No P5 model execution, inference, training, new predictions or clinical result. P4 artifacts unchanged.

## P5 execution launch — 2026-10-25

`P5_qformer_20261025`: CPU remediation/readiness PASS (6 P5 tests, 3 P4 parity tests, 4 dry-runs); P5.1 **RUNNING** in detached `tmux` supervisor PID 1223796 at launch. Config SHA-256 `841753a63743c62fe15be5cc2f9f2308a40d593668a891050da18f8e213475ce`. Exact stage status, cumulative budgets and logs are under `runs/P5_qformer_20261025/`. P5.2/P5.3 PENDING conditional gates; no P5 metric yet. This launch supersedes only the P5.0 GPU HOLD after the user's explicit P5 master authorization; P5.0 history remains unchanged.


## P5_qformer_20261025 final BLOCKED

Run: `P5_qformer_20261025`; scientific configuration SHA-256 `841753a63743c62fe15be5cc2f9f2308a40d593668a891050da18f8e213475ce`.
Status: **BLOCKED**. P4 artifacts remain unchanged.
- P5.1: PASS; updates/generations=20; peak reserved GiB=9.939453125
- P5.2: PASS; updates/generations=2000; peak reserved GiB=9.943359375
- P5.3: PASS; updates/generations=1000; peak reserved GiB=7.6953125
Stop reason: RuntimeError: CPU_ANALYSIS process exit 1; inspect /storage/student4/hanhpm/CXR_LLaVA_Improvement/research/blip2_cxr_v4/runs/P5_qformer_20261025/CPU_ANALYSIS.stderr.log


## P5_qformer_20261025 final COMPLETE

Run: `P5_qformer_20261025`; scientific configuration SHA-256 `841753a63743c62fe15be5cc2f9f2308a40d593668a891050da18f8e213475ce`.
Status: **COMPLETE**. P4 artifacts remain unchanged.
- P5.1: PASS; updates/generations=20; peak reserved GiB=9.939453125
- P5.2: PASS; updates/generations=2000; peak reserved GiB=9.943359375
- P5.3: PASS; updates/generations=1000; peak reserved GiB=7.6953125
indiana: paired M3−M1 BLEU-4 -0.05151685011825502, 95% CI [-0.0574076372468186, -0.04620050574398297]; clinical proxies require caution.
mimic: paired M3−M1 BLEU-4 -0.0326666025429177, 95% CI [-0.04494628927985667, -0.026138784306282362]; clinical proxies require caution.


## P5 scientific closeout — 2026-10-26

Execution **COMPLETE**: P5.1 20/20 updates, 193.16 s, peak reserved 9.939 GiB; P5.2 2,000/2,000 updates, one epoch, 9,849.69 s cumulative, peak 9.943 GiB, verified checkpoint; P5.3 1,000/1,000 paired new generations, 9,546.18 s, peak 7.695 GiB. The earlier storage interruption was charged conservatively and resumed from the last verified step. The CPU analysis used saved CheXpert output and the pinned P4 METEOR scorer/WordNet; no P4 predictions were regenerated.

| Cohort (500 each) | P4 M1 BLEU-4 | P5 M3 BLEU-4 | Paired Δ M3−M1; 95% CI |
|---|---:|---:|---:|
| MIMIC validation | 0.034638 | 0.001971 | −0.032667 [−0.044946, −0.026139] |
| Indiana | 0.051517 | 0.000000 | −0.051517 [−0.057408, −0.046201] |

P5 M3 produced only **two distinct report texts per cohort**; all 500 outputs per cohort reached the 128-token generation limit. The two repetitive patterns are nonclinical letter lists. Historical joint-definite clinical micro-F1 is **N/A** for M3 because its labeler outputs have zero definite paired labels; it must not be reported as zero. Under the GT-definite/prediction-missing-negative sensitivity policy, micro-F1 fell from 0.1372 to 0 on MIMIC and from 0.0488 to 0 on Indiana. Text/labeler omission candidates rose from 606 to 658 on MIMIC and 202 to 209 on Indiana (class-case counts); these are not radiologist-confirmed errors. Zero unsupported-positive candidates for M3 reflect report collapse, not clinical improvement.

**Decision: HOLD further Q-Former training/inference with this configuration.** The execution gates passed, but the matched scientific comparison strongly disfavors P5 M3 versus P4 M1. Investigate collapse using saved artifacts and CPU-only diagnostics before proposing any new GPU run; any changed optimization or architecture requires a new protocol and approval. Full per-class counts, coverage, lexical metrics, bootstrap and error proxies: `runs/P5_qformer_20261025/P5.3_cpu_analysis/metrics.json`, `error_pairing.json`; machine-readable decision: `P5_final_decision.json`. No P4 artifact was changed.


Google Docs sync VERIFIED: VLM Medical → Report 2/11 → P4 progress 19/10–02/11, mục “P5 — Q-Former: kết quả cuối (26/10/2026)”; đã đọc lại bảng BLEU-4 và hình. Tab ID `t.og7ix2n9u1zd`; biểu đồ cục bộ `runs/P5_qformer_20261025/P5.3_cpu_analysis/bleu4_comparison.png`.
