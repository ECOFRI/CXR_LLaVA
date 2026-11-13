# P0 results and decisions

Classification: inventory only; no smoke test, metric-complete evaluation, or benchmark claim. No model load, GPU inference, training, or metrics were run.

Decisions: stop GPU work at P0 gate; preserve existing Indiana outputs for later matching; treat VinDr as label-only until report text is verified; verify MIMIC report/JPG linkage and coverage before report metrics; resolve env probe/no-space blocker. Historical scores in the plan are not newly verified here.

## P1 Part A — X00 Indiana, 2026-10-22

Status PARTIAL: CPU re-score COMPLETE on 3,667/3,667 valid saved reports; METEOR N/A because WordNet is unavailable offline. Historical CheXpert joint-definite macro-F1 0.4457087416797642 reproduced exactly, including all seven per-class confusion counts. New corpus BLEU1-4 0.28794/0.14676/0.08183/0.04495 and mean ROUGE-L 0.20789. Clinical micro-F1 0.53423. Study-level 1,000-rep seed-42 CIs and sensitivity policies are in `runs/X00_indiana_rescore_20261022T062016Z/metrics.json`. No AUROC, inference, GPU work or clinical claim. Decision: retain frozen selection cohort as reference; investigate auxiliary candidate-file mismatch later; await Claude review before Part B.

## P1 Part B — MIMIC

PARTIAL: paired reports exist in 50/50 sampled studies; JPGs in 48/50 across two local roots. Official split counts are 368,960 train, 2,991 validate, 5,159 test images. The repository 3,000-row “testset” maps 2,943 rows to official train; decision: reject it as a locked test cohort. MIMIC report evaluation is conditional on full linkage, coherent versions and contamination checks. Details: `reports/P1_mimic_audit.md`, `manifests/mimic_p1_summary.json`.

## P1 Part C — VinDr

PASS inventory: 15,000 training images with three reader label rows each; disjoint 3,000-image consolidated test; 28 binary classes and box annotations. Decision: VinDr is classification/localization only unless report text is verified. AUROC/AUPRC remain N/A until continuous scores exist; fixed validation thresholds are required for F1. Evaluator prepared and dry-run/tests passed; no metric run. Details: `reports/P1_vindr_audit.md`.

## P1 Part D — NIH / CheXpert

PASS targeted location: NIH local CXR8 cohort 4,999 rows with prior patient-disjoint train/val/test 3,783/344/872; exploratory due possible CXR pretraining exposure. CheXpert local 668-image/500-study test is locked; train/val NOT_FOUND in targeted roots. Decision: no test-set tuning or new metric run. Details: `reports/P1_nih_chexpert_locations.md`.

## P1 Part E — BLIP-2 candidate comparison

PASS metadata/dry-run, no checkpoint download. Recommend official Salesforce BLIP-2 OPT 2.7B at pinned SHA `59a1ef6c1e5117b3f65523d1c6066825bcf315e3`: 13.9551 GiB of safetensors plus support files under the 20 GiB cap. Single-K80 FP16 fit and report quality UNKNOWN. Decision: await Claude candidate approval before any transfer; do not treat import-only compatibility as runnable. ClinicalBLIP, RRG-LLM, and Swin-Qwen3 official runnable assets remain UNVERIFIED; R2-LLM and META-CXR code is located, external checkpoint access unverified. Details: `reports/P1_blip2_candidates.md`, `reports/03_paper_evidence_matrix.md`, `manifests/papers.csv`.

## 2026-10-22 — P1 Part E download (Claude QA)
- Salesforce/blip2-opt-2.7b @59a1ef6 downloaded to `.cache/models/blip2-opt-2.7b_59a1ef6` (13.9551 GiB, 2/2 safetensors size+SHA-256 verified). Approval: `approvals/P1_download_blip2_opt_2_7b.json`. Log: `runs/P1_blip2_download/`.
- Not loaded; no GPU inference. X02 smoke (10 Indiana cases, 1 K80, FP16) awaits separate owner approval.

## P2.1 — E4 paired evaluation

PASS exploratory: frozen 264 reports across 24 identical studies and 11 arms; study-paired CIs and four clinical policies recorded in `runs/P2_e4_paired/metrics.json`. C0 BLEU-4 0.0371, historical macro-F1 0.8063. No arm is promoted on this small cohort without a paired delta CI excluding zero. METEOR to be filled after P2.5. CPU only.

## P2.2 — MIMIC official split coverage

PASS complete file coverage check: 2,809/2,991 validate and 4,904/5,159 test JPGs found across both local roots; 1,808/1,808 validate and 3,269/3,269 test reports found. Train/validate/test patient overlap zero. Missing local JPGs prevent complete image-report evaluation here. CXR-LLaVA MIMIC training exposure makes this unsuitable as an independent holdout without a contamination analysis; CLEAR exposure UNKNOWN. See `manifests/mimic_official_coverage.json`.

## P2.3 — BLIP-2 preflight

PARTIAL: downloaded shard SHA-256 values match the approved manifest, and config plus processor load offline with `use_fast=False`. The default fast tokenizer fails in the installed tokenizers parser. Decision: X02 uses slow GPT2Tokenizer and remains dry-run only this batch. No model weights were instantiated; K80 feasibility UNKNOWN. See `manifests/blip2_preflight.json`.

## P2.5 — METEOR

PASS: X00 METEOR 0.236041 across 3,667 saved reports; E4 C0 METEOR 0.211113 across 24 studies. Study bootstrap CIs saved. Official NLTK 3.6.7 scorer source plus NLTK 3.3 WordNet are recorded in `manifests/meteor_resource.json`; no package installed. Original X00 run remains untouched. This adds a text metric, not clinical validation.

## P2.6 — R2-LLM and META-CXR source audit

PARTIAL: official source locations and entrypoints verified at pinned SHAs; R2 root MIT, META root license UNKNOWN. Both require additional checkpoints and dependencies that are not verified on K80/cu102. Decision: no execution or benchmark claim; resolve licensing/assets and compatibility before any model job. R2 source worktree is partial after a timed-out checkout; audited files were restored from the pinned commit. See `reports/P2_repo_audit.md`.

## P2.4-prep — X02 BLIP-2 smoke

PASS preparation and dry-run on the fixed 10-case Indiana sample; no model or GPU job started. The launcher gates stage 2 on stage 1 PASS and records per-case and resource outputs. Claude/QA verified K80 GPU0 passes the seven-operation FP16 probe; the launcher now repeats that probe before model loading and records it in status.json. Actual model fit remains UNKNOWN. Decision: Claude controls any launch; no inference result exists. See `configs/experiments/X02_smoke.yaml` and `runs/X02_smoke_dry_run/status.json`.

## X02 smoke output validity correction

FAIL: Claude's GPU0 smoke run completed 10 cases but generated empty text for all 10. The original stage PASS statuses reflected execution only; the root status is corrected to FAIL, with zero valid outputs. Decision: exclude this run from report metrics, retain its files, switch to the fixed QA prompt, and require nonempty outputs before evaluation. The one-image prompt diagnostic is evidence for prompt sensitivity, not a benchmark result.

## X02 v2 output gate and exploratory evaluation

PASS code/dry-run only: the fixed QA prompt replaces the v1 instruction prompt, and any empty generation now fails its stage with `n_valid` recorded. Pixel values are explicitly cast to model dtype. The launcher runs CPU evaluation only after ten valid outputs, using BLEU-1..4, ROUGE-L, METEOR and historical clinical F1, labeled exploratory n=10. The prior all-empty run remains FAIL and is excluded from metrics; no new GPU run has been launched.

## X02 v2 final evaluation and R2 checkout

PARTIAL X02 v2: Claude's GPU0 run generated 10/10 nonempty reports at 7.57 GB peak and 1.7406 s/case mean. The original clinical 0.0 was invalid because no reference-positive class label had a paired generated definite label. The old evaluation is SUPERSEDED; a CPU rerun with X00-matched CheXpert/NegBio revisions reports historical micro/macro F1 N/A with reason. Exploratory n=10 BLEU1-4 .002221/.000744/0/0, ROUGE-L .070539, METEOR .036880; bootstrap CIs in `evaluation_v2/metrics.json`. Decision: retain technical generation success but no clinical performance claim; v1 remains FAIL. The R2-LLM source checkout is now clean at its pinned SHA; checkpoint/access compatibility remains UNKNOWN. No new GPU job by Codex.

## P3.1 paper evidence

Six models classified with source-linked published metrics in `manifests/papers.csv` and `reports/03_paper_evidence_matrix.md`. Indiana/MIMIC pretraining or test overlap is unresolved; no published metric is a local benchmark. P3.2 asset/licensing decision pending.

## P3.2 weights and licensing

Decision: X04 R2 and X05 META remain BLOCKED for checkpoint execution. R2 HF model API returned HTTP 401; Claude/QA subsequently verified zero public author models and identified the README URL as an anonymized-review placeholder. Official R2 weights are NOT_PUBLICLY_RELEASED; file names/sizes and weight terms UNKNOWN. META Drive page yielded no verified `.pth` filename/size; root license and weight terms UNKNOWN. Vicuna/LLaMA base terms require separate verification; K80 sm37 lacks supported bitsandbytes LLM.int8 (≥sm75), and FP16 7B/13B base weights alone exceed one K80. See `reports/P3_weights_licensing.md`. P3.3-P3.7 not started.

## P3.3 reimplemented scripts

Three local BLIP-2 bridge arms have passing CPU-only dry-runs, classified REIMPLEMENTED_SCRIPT_READY and SCRIPT_READY_NOT_RUN; no paper model reproduction or training is claimed. X03/X04/X05/X07 are SKELETON_ONLY/BLOCKED; X08/X09 local encoder bridge configs are BLOCKED until feature adapters/caches are verified. Official R2 weights are NOT_PUBLICLY_RELEASED per Claude/QA HF API check. See `reports/P3_implementation_audit.md`.

## P3.4 legacy environment plan

PLAN_ONLY: proposed isolated py3.9/torch1.12.1/cu102 R2 recipe and guarded install/rollback scripts exist but were not run. Current K80 env R2 import-only probe fails at missing `omegaconf`; official R2 weights remain NOT_PUBLICLY_RELEASED, META rights/assets unresolved, and sm37 excludes supported bitsandbytes quantization. No inference or training approval follows from this plan. See `reports/P3_legacy_env_plan.md`.

## P4.0 — configuration freeze preparation, 2026-10-23

P4.0 preparation PARTIAL; P4 BLOCKED at scientific freeze. One GPT-6 Sol medium worker implemented generic execution safety; 14/14 CPU checks passed. Proposed manifests contain 2000 unique MIMIC train, 500 official validation and 500 matched Indiana studies, with zero selected MIMIC patient overlap. These are source-order convenience candidates, NOT approved frozen cohorts. Tokenizer EOS 2 conflicts with checkpoint generation EOS 50118 (newline); prompt-conditioned report loss, token limits, optimizer defaults/module modes and candidate cohort policy await one bundled approval in `reports/P4_progress.md`. Proposed config and SHA identities: `runs/P4_projection_pilot_20261023/freeze_proposal.json`. GPU entrypoints still require integration. No new GPU training, inference, metrics, or clinical claim; prior HOLD/BLOCKED/dry-run evidence unchanged.

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

## P4.3 partial evaluation decision — 2026-10-24

P4.0 APPROVED; P4.1 and P4.2 PASS. P4.3 PARTIAL: fixed inference budget yielded 469 MIMIC and 5 Indiana valid matched cases, 474/1,000 planned. CPU scoring completed. On MIMIC, corpus BLEU-4 M0=0 and M1=0.033834, paired 1,000-replicate patient-cluster delta 95% CI [0.026367, 0.045306]. This is evidence of better lexical overlap on the completed, source-order prefix only. Historical clinical micro-F1 M0=0.2162 and M1=0.4424 uses very different jointly definite label counts (99 versus 322) and positive support (12 versus 127); do not claim clinical improvement. Indiana n=5, M2 point metrics are exploratory only. All M0 outputs reached the 128-token cap, and labeler recorded seven parse-tree warnings. Hallucination/omission proxies are per-class FP/FN; separate contradiction analysis is absent.

Decision: record P4 as PARTIAL and preserve the fixed budget. A full-cohort comparison requires a separately approved GPU inference budget. P5/P6 training remains unapproved. Metrics: `runs/P4_projection_pilot_20261023/P4.3_evaluation/metrics.json` SHA-256 `04ebf1c0f670985dbd6d7f70f8e9651849e1a0d1d40f7656156c4074bec06628`. Google Docs sync pending.

## P4.3 full-cohort terminal sync — 2026-10-25

P4.0 APPROVED; P4.1/P4.2 PASS; P4.3 inference and full CPU evaluation PASS on 1,000/1,000 ordered paired studies (500 MIMIC validation, 500 Indiana), with 15243.812/18,000 GPU seconds charged. Full metrics: `runs/P4_projection_pilot_20261023/P4.3_evaluation_full/metrics.json` SHA-256 `ed6be843294d14deafa1fd9c9366bf302a53c08f12f6804c530e680393fb9be5`. The earlier 474-pair partial metrics remain preserved.

MIMIC 500/500 valid pairs: BLEU-4 M0=0.0000 M1=0.0346, M1−M0=0.0346 (paired cluster 95% CI [0.0275, 0.0459]); historical labeler micro-F1 M0=0.2167 (107 definite labels, 13 positive support), M1=0.4351 (353 definite labels, 139 positive support).

INDIANA 500/500 valid pairs: BLEU-4 M0=0.0031 M1=0.0515, M1−M0=0.0484 (paired cluster 95% CI [0.0430, 0.0544]); historical labeler micro-F1 M0=0.1839 (79 definite labels, 8 positive support), M1=0.1045 (463 definite labels, 56 positive support).

These are research metrics on selected source-order cohorts. CheXpert/NegBio labels are automated; blank and uncertain labels are excluded from historical F1, and the definite-label masks/support differ by arm. Lexical overlap does not establish clinical correctness. Indiana M2 uses saved historical X00 outputs under a different model protocol. Report repetition and 128-token truncation limit interpretation; this is not clinical validation. Google Docs sync remains pending.

## P4 final scientific conclusion and P5 proposal — 2026-10-25

P4 technical execution is PASS. On the fixed paired cohorts, projection-only adaptation increased corpus BLEU-4 from 0 to 0.034638 on 500 MIMIC validation studies (Δ 0.034638; patient-cluster 95% CI [0.027476, 0.045860]) and from 0.003094 to 0.051517 on 500 Indiana studies (Δ 0.048423; study-cluster CI [0.043004, 0.054389]). Both paired intervals used 1,000 seed-42 bootstrap resamples and exclude zero. The Indiana change shows lexical transfer outside the current MIMIC fine-tuning cohort; source-order selection and possible pretraining overlap limit broader generalization. M2/X00 BLEU-4 on those Indiana IDs was 0.044782 under a different model protocol, so it is descriptive.

The evidence does not show consistent clinical improvement. Historical CheXpert/NegBio micro-F1 was MIMIC M0/M1 0.2167/0.4351 and Indiana 0.1839/0.1045. Its jointly definite masks covered only 107/353 MIMIC and 79/463 Indiana class-cases out of 3,500 per arm. When GT-definite cases are retained and missing/uncertain predictions are negative, micro-F1 was MIMIC 0.0340/0.1372 and Indiana 0.0556/0.0488. Indiana M2 was 0.5448 historical and 0.3883 alternative, with different protocol. The historical MIMIC micro-F1 delta CI [-0.0023, 0.4187] includes zero; Indiana delta CI [-0.2087, 0.0440] includes zero. Full per-class confusion, precision/recall/F1, support, coverage and missing rates are preserved in the final analysis JSON; undefined classes remain N/A. The original `metrics.json` uncertain-label count fields undercount because they match literal `-1` against CSV `-1.0`; the post hoc analysis corrects these rates from immutable labeler rows, while reproducing the original F1/confusion exactly. The original evaluation artifact remains unchanged.

Text/labeler proxy cases do not establish whether unsupported positive findings or omissions decreased. M1 versus M0 hallucination proxies were 47 vs 94 MIMIC and 69 vs 71 Indiana; omission proxies were 322 vs 334 and 149 vs 150. M1 introduced 34 MIMIC and 8 Indiana incorrect-negation cue candidates; same-pathology positive/negative cues appeared within 22 M1 reports in each cohort. Scope, location, severity and report repetition require manual review, ideally blinded radiologist adjudication against images. All M0 outputs hit the 128-token cap, which also weakens report-quality inference. No radiologist-confirmed error rates, patient-safety claim or scientific clinical-success claim follows from P4.

P5 recommendation: GO for CPU-only P5.0 proposal/design; HOLD all P5 GPU execution pending a separate decision. The lexical gain is enough to motivate testing whether a trainable Q-Former changes the outcome, but not enough to predict clinical benefit. A higher-capacity model can add failure modes and overfit. Prioritize a small adjudicated error subset and freeze the P5 protocol before requesting GPU GO. Do not use remaining P4 time for P5.

Proposed X06_qformer ablation (not approved): initialize the same pinned original BLIP-2 OPT-2.7B revision 59a1ef6; train the pretrained Q-Former, query tokens and pretrained language projection; freeze vision encoder and OPT. Safetensors shape inventory gives 105,137,664 Q-Former + 24,576 query-token + 1,968,640 projection = 107,130,880 proposed trainable parameters, versus 1,968,640 in P4. This is a local reimplementation, not a paper-model reproduction. The existing P3 `X06_qformer.yaml` and generic bridge trainer have `max_steps=100` and a different loss/target path, so they cannot run the P5 comparison as-is.

P5.0 proposal: reuse the approved P4 train manifest (2,000 official MIMIC studies), validation/Indiana manifests (500 each), original checkpoint, image/view policy, QA prompt, FINDINGS+IMPRESSION target-only/EOS loss, tokenizer, 128/256-token caps, 32 visual tokens, greedy EOS2 generation and evaluation ontology/policies. Freeze exact Q-Former/query-token/projection trainable list, optimizer/scheduler/LR, modes/dtypes, seed, bootstrap and all hashes in a new P5-specific scientific approval record. P4 AdamW settings are a candidate, not an automatic authorization for the larger trainable module. No architecture/hyperparameter changes during a run.

P5.1 proposed feasibility gate, conditional on P5.0 approval: one K80, 20 real optimizer updates, at most 15 cumulative minutes, peak reserved VRAM at most 10.5 GiB (11,441 MiB installed), finite loss/gradients, verified updates, unchanged frozen parameters and checkpoint reload. P4 peak was 7.953 GiB. The 105.16M newly trainable parameters require about 1.57 GiB for FP32 parameter/gradient/Adam states at 16 bytes each, plus roughly 0.20 GiB for FP16→FP32 parameter expansion, before activation and allocator overhead. Therefore ~9.72 GiB is only a lower-bound planning estimate; actual Q-Former VRAM and throughput are unknown until the gate. Stop if the gate or forecast fails.

P5.2 proposed conditional budget: one K80, same 2,000-study manifest, one epoch, fresh pretrained initialization and at most 4 cumulative GPU hours with resume and hash/position/RNG checks. Proceed only if measured P5.1 throughput predicts completion inside the cap with margin. P5.3 proposed conditional evaluation: generate only the new Q-Former arm on the frozen 500+500 cohorts (up to 1,000 new reports, provisionally at most 3 cumulative K80 hours), reuse P4 M0/M1 and saved Indiana X00 predictions, check a ten-case validity gate, then compute paired corpus BLEU-4 and identical clinical/coverage policies with 1,000 cluster bootstrap resamples, seed 42. GPU limits and the full P5 configuration require explicit approval; none has been launched.

Evidence: `runs/P4_projection_pilot_20261023/P4.3_evaluation_full/metrics.json`, `P4.3_final_scientific_analysis.json`, `P4.3_final_error_pairing.json`, `P4.3_final_integrity_audit.json`; proposed architecture source `configs/experiments/X06_qformer.yaml` and `scripts/train/train_blip2_bridges.py`. P4 config SHA-256 d010d2cf32cb6cc9b96b834e5eb0a5df1d8bd13985ec31cf5faa447260471eb9; manifest index SHA-256 74965a79b31394a65af10dcb7987db0881e5083a8bb0da888bccbe120bd03175; final adapter SHA-256 7f833eea69fb6d2743b19bbc808be949bb1a5e014bf43a81f08090f9a5f0d8af.

## P5.0 CPU design decision — 2026-10-25

P5.0 design/static validation is complete; **P5.1/P5.2/P5.3 GPU HOLD**. Metadata-only inspection of the pinned original BLIP-2 checkpoint verifies 105,137,664 Q-Former, 24,576 query-token and 1,968,640 projection parameters (107,130,880 proposed trainables). A small CPU Q-Former surrogate verified wrapper output shapes and nonzero finite gradients to all three modules; no complete-model load, inference or GPU work was done. P4 config/index/cohort and final adapter/metrics hashes match their recorded values; all 2,000/500/500 manifest studies/images are unique and selected MIMIC train/validation subject overlap remains zero. These checks establish design consistency, not P5 executable readiness.

The current generic `X06_qformer.yaml`/`train_blip2_bridges.py` path has a 100-update cap, different target masking and selection, frozen-model mode/resume mismatches and an 11.5 GiB free-memory preflight incompatible with the one-K80 proposal. It must not run as P5.1 without a new P5-specific implementation and CPU tests. Using the P4 7.953 GiB measured peak plus incremental FP32 training state gives ~9.324 GiB as a planning comparison before extra Q-Former activations; peak may exceed the proposed 10.5 GiB gate. P5 runtime is unmeasured. Within a 4-hour/2,000-update proposal, 7.2 s/update is the zero-overhead ceiling; a 1,800 s overhead reserve gives a 6.3 s/update planning target.

The pre-registered primary comparison is future M3 Q-Former minus saved P4 M1 corpus BLEU-4 separately on identical 500 MIMIC validation and 500 Indiana studies, with paired 1,000-replicate seed-42 cluster bootstrap. Clinical F1 uses both the historical joint-definite policy and GT-definite/missing-prediction-negative sensitivity with full support/coverage disclosure; text/labeler error candidates are not clinical adjudications. M0 and Indiana M2/X00 remain secondary saved controls. This ablation does not validate CLEAR concepts. The matched P4 AdamW settings are a proposed candidate, not a P5 scientific approval.

Decision: **GO only for completed P5.0 CPU design; HOLD P5.1** until a separate scientific/resource approval reviews the new executable contract, CPU tests, P5.1 output identity, 20-update/900 s/10.5 GiB gate and stop rules. P5.2 (2,000 updates/4 hours) and P5.3 (≤1,000 generations/3 hours) remain future conditional proposals. Versioned design files: `reports/P5_0_20261025/P5_architecture_spec.md`, `P5_matched_protocol.md`, `P5_cpu_static_validation.md`, `P5_resource_budget.md`, `P5_evaluation_plan.md`, `P5_approval_request.md`. P4 source artifacts are unchanged.

## P5 execution authorization and launch — 2026-10-25

The user's subsequent P5 master directive authorized P5.1→P5.2→P5.3 within explicit hard limits, superseding the P5.0 proposal's GPU HOLD. The new frozen P5 config has SHA-256 `841753a63743c62fe15be5cc2f9f2308a40d593668a891050da18f8e213475ce`, one K80, 10.0 GiB peak reserved ceiling, and separate 900/14,400/10,800-second stage caps. The repaired trainer uses P4 target-only/EOS masking, P4 manifests, fresh original pretrained initialization for P5.2, full cursor/RNG/optimizer resume, and frozen ViT/OPT eval modes. Six P5 CPU tests, three P4 parity tests and four dry-runs passed. Detached supervisor PID 1223796 launched, P5.1 status RUNNING; no P5 training or evaluation result is claimed until gates complete. Machine status and evidence: `runs/P5_qformer_20261025/`; authoritative launch status: `reports/P5_progress.md`.


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
