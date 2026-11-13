# Experiment Analysis Log — E0B / E2 direct contrastive

## 1. Experiment Information

E0B and E2 run `20261013T214811_E2_direct_contrastive`, completed 2026-10-13 (Asia/Ho_Chi_Minh). Codex execution agent, project CXR_LLaVA_Improvement, branch master, project commit `f6535c6d9d7eb23c2e9a2ac24f48771a1d276efc`. Repository https://github.com/hanhpm/CXR_LLaVA_Improvement. Isolated scripts: research/e2_direct.py, e2_finalize.py, e2_analysis_report.py.

## 2. Objective

Validate the official CLEAR text pathway, then evaluate exactly one prespecified direct image/text pass on locked CheXpert 518 images against existing CXR-LLaVA Yes/No predictions. No model or threshold training.

## 3. Survey / Technical Motivation

Official CLEAR commit `57d2eae57ddabf8e56c8013ec675336d71b3d8e7`, DINO commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Official README/examples/quickstart.py use normalized image/text vectors and unscaled paired cosine softmax. examples/concepts/exp_zeroshot.py lines 357–358 use lowercase positive/no prompts; zero_shot.run_softmax_eval specifies positive-first paired softmax. File hashes are in the frozen protocol. CLEAR paper DOI 10.1038/s41551-026-01741-4 describes full concept projection, which this direct pathway does not reproduce. Pipeline ranking is the question; isolated encoder H1 and concept-utility H2 remain separate.

## 4. Environment

Reused Conda prefix `/storage/student4/hanhpm/CXR_LLaVA_Improvement/research/envs/cxr-clear-cpu`; Python 3.10.22, torch 2.3.1+cpu, CPU only, FP32, four intraop threads and one interop thread. CUDA was not initialized. Exact versions in environment_manifest.json and pip_freeze.txt. Current pip freeze exactly matches prior CPU environment setup. Live available RAM was approximately 41.2 GiB; job cap 8 GiB. Shared NFS free space 2.178 TB; account quota remains unverified. Saved inventory reused.

## 5. Dependency Setup

No packages installed or changed. Offline strict cached checkpoint load with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1. xFormers-unavailable warnings reflect the expected CPU implementation and did not prevent inference.

## 6. Code / Notebook Changes

Only isolated research modules and new artifacts created. Official CLEAR worktree is clean. An initial pre-score harness KeyError for absent parsed_labels entries was corrected to .get(l), matching the existing missing-response policy. No test scores were produced in that failed attempt. Offline finalizer changes figures and adds paired CIs; it never reruns either encoder.

## 7. Experiment Configuration

Locked CheXpert: 518 frontal images, 500 studies, 500 patients; five common findings. Official 448×448 RGB square bicubic resize and released normalization. Prompts are lowercase finding and no + finding. Score is softmax of positive/negative cosine similarities without logit scaling. Fixed score >=0.5 is an untuned exploratory operating point, not a calibrated probability. Batch one, seed 42, 1,000 patient-cluster bootstrap replicates with percentile 95% CIs. Exactly 50 compatible E0A projected globals reused and 468 new globals extracted. Protocol frozen before scores: `d74c956edb5402c2efb16efb5845107e191c18abc2d4a71cb473f8979836ae6a`.

## 8. Code Executed

```python
text = normalize(model.encode_text(tokenize(prompts)), dim=-1)
image = normalize(model.encode_image(native_input), dim=-1)
positive_score = (image @ text.T).reshape(5, 2).softmax(-1)[:, 0]
```

Exact command in reproduce.sh. Offline figure/paired-CI command in offline_reproduce.sh. Reproduction is an explicit new run, not automatic test repetition.

## 9. Output

Tokenizer [10,77], normalized text [10,768], projected image features [518,768], 2,590 finite scores. E0B 1→10 passed; repeated official image/cache endpoint differences and score differences were zero. Checkpoint and source identity checks passed. Logs, per-image timings, continuous scores and exact baseline case hashes are saved.

## 10. Output Analysis

All 518 source identities/hashes align with frozen baseline records. Cached features match checkpoint and native preprocessing provenance. No numerical, shape or resource failures. Full concept vocabulary and both embedding matrices are absent in the bounded local repo/cache/HF search; no assets downloaded. Direct pathway completed; full concept pathway remains blocked.

## 11. Quantitative Results

|Finding|AUROC (95% cluster CI)|AUPRC (95% cluster CI)|CLEAR F1|CXR parsed F1|
|---|---|---|---|---|
|Atelectasis|0.857648 (0.824694–0.892279)|0.658237 (0.580674–0.751828)|0.711485|0.527536|
|Cardiomegaly|0.894552 (0.863871–0.923017)|0.773596 (0.697555–0.843763)|0.730878|0.240000|
|Consolidation|0.882589 (0.832160–0.927293)|0.330906 (0.197590–0.520360)|0.197952|0.139535|
|Edema|0.899801 (0.868654–0.927530)|0.587351 (0.476295–0.704486)|0.453488|0.070796|
|Pleural Effusion|0.929482 (0.902645–0.951044)|0.751669 (0.659937–0.836740)|0.504808|0.265734|

CLEAR macro AUROC 0.892814; macro AP 0.620352. AP uses sklearn average_precision_score, not trapezoidal PR integration. Full-518 CLEAR fixed-0.5 macro F1 0.519722. On exactly the CXR parsed mask, CLEAR macro F1 0.521036 versus CXR 0.248720; CXR missing-as-negative sensitivity 0.247987.

Warmed forward mean 0.882717 s; p95 0.934947 s over 468 new forwards. Mean excludes preprocessing, warmups and cache IO. Model load 3.984 s; process wall 549.458 s including identity checks and evaluation, excluding later offline evidence refinement. Peak RSS 1498669056 bytes (1.395744 GiB). Zero GPU hours. Global cache 1,624,984 bytes including IDs.

## 12. Qualitative Results

Ten distinct technical disagreement examples, two per finding, selected deterministically without changing the inference protocol. Contact sheet links GT, both outputs, source hash and original local image via image_evidence.csv. The 10% margin crop applies only to visualization, preserves interior intensities and aspect ratio, and records source crop boxes. Cropping is not a privacy guarantee. Images stayed local; no expert clinical review performed. Visible posture/equipment text is an unverified annotation, not a pathology conclusion.

## 13. Comparison with Baseline

This contrasts CLEAR direct image/text scoring with saved CXR-LLaVA autoregressive Yes/No outputs, not isolated vision encoders. CLEAR ranking is reported without inventing CXR continuous AUROC. CLEAR full coverage and CXR parsed coverage differ; the CSV includes CLEAR on the same CXR mask. Paired F1 differences and 1,000 patient-cluster CIs are in paired_f1_differences.csv. CXR coverage: Atelectasis 517/518, Cardiomegaly 518/518, Consolidation 518/518, Edema 513/518, Pleural Effusion 507/518. The Consolidation paired difference CI includes zero. Fixed-0.5 recall is 1.0 for Consolidation/Pleural Effusion, with precision 0.1098/0.3376: the operating-point tradeoff remains substantial. No CPU/GPU speed ranking or clinical superiority claim.

## 14. Problems Encountered

Initial missing-key error fixed in the isolated harness before scores. Full concept assets remain unavailable and require a separate resource estimate; no substitution of text similarity for full concepts. Exact pretraining image membership and account storage quota remain uncertain. DINO attention source matches the prior E0A hash; official CLEAR worktree is clean.

## 15. Conclusion

E0B official text gate 1→10 and the fixed E2 direct contrastive 518-image pass completed. Macro AUROC is 0.892814 and AP 0.620352; fixed exploratory classification gives same-mask macro F1 0.521036 versus saved CXR 0.248720. These are pipeline results under a particular threshold and cohort, not clinical validation or isolated encoder-quality proof. Full concept CLEAR and H2 were not tested.

## 16. Next Experiment

Review E1 independently under NIH pretraining-exposure limitations. Review frozen E2 methods before interpreting results. Future prompt/threshold choices require independent development data; never tune on CheXpert 518. Full concept work requires verified assets, row alignment, semantic text encoder and an explicit resource estimate. No additional test inference or scheduling initiated.

## 17. Reproducibility Checklist

Repository, branch and commits recorded; runner/source/checkpoint hashes and environment versions recorded; seed, native preprocessing, fixed prompts/threshold and cluster bootstrap recorded. All 518 image and baseline case hashes are stored in alignment_manifest.csv and rechecked in preservation_postflight.json. Logs, manifests, scores, metric policies and figures saved; successful run artifact SHAs verified. Source files stayed intact and current environment versions match the prior snapshot. Source paths in local manifests are restricted research metadata, not public release artifacts.

## 18. Short Lab Update

E0B 1→10 passed. E2 direct contrastive completed 518/518 with fixed official prompts, CPU only and 50 reused globals: macro AUROC 0.892814, AP 0.620352, same-mask F1 0.521036 vs CXR 0.248720. Wall 549.458 s; peak RSS approximately 1.396 GiB. Ten distinct technical disagreement images and patient-cluster CIs are linked locally. No test tuning, clinical review or full-concept reproduction; concept assets remain the blocker.
