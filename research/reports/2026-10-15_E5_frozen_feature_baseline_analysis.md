# E5 frozen-feature baseline analysis

## 1. Experiment Information

E5 COMPLETE. Run `research/runs/20261015T010818_V3_E5_BASELINE/E5`; measured process 2026-10-15T01:12:47.695524+07:00 to 2026-10-15T01:32:51.381626+07:00. Explicit E5-only authorization after independent E1.3/profile. Decision HOLD_COMPLEX_FUSION. Exploratory metric-complete NIH development benchmark, not external/clinical/report-generation validation.

## 2. Objective

Test whether independently standardized concatenation improves over the best single native patch-mean encoder under matched patient-grouped TRAIN-CV. Macro sklearn AP is primary; AUROC/F1 secondary. No downstream experiment.

## 3. Survey / Technical Motivation

Active V3 plus latest authorization. Original E1/E2 test scores excluded from model/prompt/architecture choice. E1.3 geometry differences never used as a fusion performance gate. Existing feature caches reused; no source-image/encoder inference.

## 4. Environment

Existing cxr-clear-cpu; Python3.10.22, NumPy1.26.4, SciPy1.13.1, sklearn1.5.2, matplotlib3.9.2. Source/version hashes and CPU-only module assertions in manifest. GPUhours0. NumPy ndarrays/CPU liblinear only.

## 5. Dependency Setup

No dependency installation or Conda mutation. Fresh RAM41.275GiB/24CPU affinity, cache/label/fold checks and128MiBallocationpassed. Existing estimate43.590min<120. No full inventory repeated. Numeric quota unverified; allocation probe is not allocation guarantee.

## 6. Code / Notebook Changes

Original research/v3_cached.py and its E5 modeling implementation preserved byte-for-byte. New thin execute/guard wrappers and sys.setprofile fit observer; no model/data/parameter overrides. Synthetic check confirmed identical coefficients/intercept with/without fit observer. Resource watchdog7200s/8GiB/2GiB and address-space7GiB fail-safe. No original checkpoint/source writes.

## 7. Experiment Configuration

NIH exploratory; potential CXR encoder pretraining overlap; not independent external test. TRAIN3783images/1222patients, outer3patient folds1261images each/407,407,408patients. OriginalVALIDATION344/57 used once as secondaryDEV.13fixedpathologies; Hernia/supplemental/NoFindingexcluded. A=CXRpatchmean1024,B=CLEARpatchmean768,A+B=1792independently standardized. Fusion has more parameters; no matched-capacity sensitivity study. Native preprocessing retained.

## 8. Code Executed

Unchanged original E5 code. Same fixed inner25%patient GroupShuffleSplit perouter(seed20261015+k), fulltrainseed20261115. Inner-fit-only scalers/labels; C[.001,.01,.1,1] chosen perlabel by innerAP, tie smallestC. L2liblinear/maxiter1000/class_weightNone/seed42. Threshold innerF1 grid.01:.99,tie closest.5thenlargest. Refit selectedC on outertraining/fullTRAIN with corresponding fitted scalers; transfer innerthresholds. No outerheldout/originalvalidation labels used for choices. Fullmodels/selection/protocolhashes written before once-only DEV score. Commands retained in reproduce_commands.md.

## 9. Output

780actualfits=624candidate+156refit;260perarm. All3outerfolds/all3arms+fulltrain models completed.0fitfailures,0convergencewarnings,maxiterations20.160953unique finite dev prediction rows,195classmetric rows; no originaltest predictions. Raw logs/fit audit/selection/unlock manifest preserved.

## 10. Output Analysis

Best single for pooledmacroAP is CLEAR(B), never just CXR. Comparator max(AP_A,AP_B) recomputed inside each matchedpatientbootstrapdraw; perlabel comparator separately takes labelmax. Pooled OOF and foldwise rankings are different aggregations; score distributions come from different foldmodels. Report both, apply no posthoc calibration or change of primarymetric. AUROC improvement in pooledOOF does not override prespecified AP outcome.

## 11. Quantitative Results

NIH exploratory; potential CXR encoder pretraining overlap; not independent external test.

| Partition / arm | Images / patients | Macro AP | AUROC | F1 |
|---|---:|---:|---:|---:|
| train_OOF / A | 3783/1222 | 0.079991 | 0.606881 | 0.125048 |
| train_OOF / B | 3783/1222 | 0.178787 | 0.709392 | 0.251887 |
| train_OOF / A+B | 3783/1222 | 0.177186 | 0.728990 | 0.250796 |
| validation_DEV / A | 344/57 | 0.147303 | 0.639778 | 0.128361 |
| validation_DEV / B | 344/57 | 0.256568 | 0.748612 | 0.243131 |
| validation_DEV / A+B | 344/57 | 0.251516 | 0.738461 | 0.250574 |

Primary fusion−bestmacroAP -0.001601506,95%patientCI[-0.013699334,+0.009033026],validdraws500/500(seed20261015). Conditional on fixedOOFmodels, no model/refit/hyperparameter uncertainty. FoldAPdeltas [-0.00235461696821887, -0.0013664308648849988, -0.012141416542552436]; secondaryDEVdelta -0.005051424.

NIH exploratory; potential CXR encoder pretraining overlap; not independent external test.

| Finding | CXR AP | CLEAR AP | Fusion AP | Fusion−per-label best AP [95%CI] | Positive images / patients |
|---|---:|---:|---:|---:|---:|
| Atelectasis | 0.168701 | 0.292130 | 0.316735 | +0.024606 [-0.001030, +0.056461] | 312/169 |
| Cardiomegaly | 0.042884 | 0.321888 | 0.298518 | -0.023370 [-0.113181, +0.049194] | 98/72 |
| Consolidation | 0.070388 | 0.111092 | 0.115914 | +0.004822 [-0.028599, +0.049849] | 124/63 |
| Edema | 0.080968 | 0.087879 | 0.124165 | +0.036286 [-0.044990, +0.098260] | 48/19 |
| Effusion | 0.172616 | 0.419505 | 0.432579 | +0.013074 [-0.007292, +0.031067] | 296/136 |
| Emphysema | 0.023721 | 0.151704 | 0.118067 | -0.033636 [-0.090785, +0.015924] | 76/49 |
| Fibrosis | 0.073212 | 0.182189 | 0.189318 | +0.007129 [-0.039824, +0.047717] | 128/91 |
| Infiltration | 0.244095 | 0.306864 | 0.320399 | +0.013535 [-0.011415, +0.038379] | 562/278 |
| Mass | 0.035890 | 0.150364 | 0.129377 | -0.020986 [-0.048031, +0.008369] | 114/65 |
| Nodule | 0.050041 | 0.081928 | 0.068123 | -0.013805 [-0.033224, +0.009512] | 155/104 |
| Pleural Thickening | 0.036209 | 0.084012 | 0.072902 | -0.011110 [-0.056379, +0.031951] | 91/75 |
| Pneumonia | 0.013084 | 0.018224 | 0.019870 | +0.001646 [-0.018350, +0.012123] | 40/34 |
| Pneumothorax | 0.028069 | 0.116453 | 0.097444 | -0.019009 [-0.061628, +0.012780] | 72/37 |

Resource: fullguardedE5wall 1203.686117s (20.061435min),peakRSS 0.586483GiB; CPUuser 1375.499474s/system 4.140036s.0GPUh. Runtime includes loading/tuning/refits/predictions/bootstrap/originalplot; later cached-output reporting/evidence link verification excluded.

## 12. Qualitative Results

e5_comparison.png and supplemental e5_paired_incremental_benefit.png visually inspected; all13labels shown, not cherry-picked. Twelve predeclared fusion/CLEAR score-gap cases linked to sourceSHA, bothcache paths, all3scores/thresholds anddatasetrelative TP/FP/FN/TN in e5_feature_evidence_linked.csv. Illustrative score gaps are not population sampling or verified pathology. No pixels opened; expert review NOTPERFORMED.

## 13. Comparison with Baseline

Fusion AP is below CLEAR in pooledOOF, eachouterfold andsecondaryDEV. MacroΔCIincludeszero, so evidence does not establish reliable gain or definite degradation. This is matchedCV/DEV, not a new NIHtest/CheXpert comparison. All prior artifacts preserved. Capacity/native-preprocessing differences and weaklabels limit mechanism attribution.

## 14. Problems Encountered

No execution/resource/integrity failures. Primary negative result preserved without changing folds/Cs/labels/arms or rerunning for a better outcome. Patient-bootstrapCIconditional; localconvenience cohort/pretrainingexposure/scorepooling/classsupport issues restrict generalization.

## 15. Conclusion

HOLD_COMPLEX_FUSION. Consistent improvement criterion not met. PointAP is lower but CIincludeszero, preventing a reliable-worse NO_GO conclusion under originaldecisionimplementation. No report-generation/clinical advantage demonstrated; geometry differences did not translate into reliable aggregate AP improvement in this baseline.

## 16. Next Experiment

STOP. Propose only: review label/support quality and paired image-report data provenance; keep CLEAR-only as strong control. Any independent cohort, capacity-control/fusion variant or frozen-LLM pilot needs separate scope/compute/approval. No adaptive gating/spatial fusion/downstream training launched.

## 17. Reproducibility Checklist

Cache/source/protocol hashes, exact ID/sourceSHA/splits, outer/inner fold assignments, fit input sizes/widths/Ccounts andselectedrefits verified. Finalscaler means/samplecounts matchTRAIN only. Model/selection/protocolhashes verify pre-DEV unlock. Unique finite score rows, scripts/resources/uncertainty policies/evidence/figures retained. No original test evaluation, downloads, envmutation, checkpointmodification or scheduler.

## 18. Short Lab Update

E5completed780fits/3folds in20.06min/0.586GiB CPU. OOFAP A0.07999/B0.17879/fusion0.17719; Δfusion−best−0.00160,CI[−0.01370,+0.00903]; secondaryDEV0.25152vsB0.25657. HOLD_COMPLEX_FUSION; stop with no downstreamlaunch.
