# Experiment Analysis — E4-A3A automated reference-proxy evaluation

## 1. Experiment Information

E4-A3A, 17/10/2026, `CXR_LLaVA_Improvement`, local Git HEAD `f6535c6d9d7eb23c2e9a2ac24f48771a1d276efc` (working tree contains uncommitted/untracked research files). Script: `research/e4_a3a.py`. Run: `research/runs/20261017_E4_A3A_automated_proxy_eval_v1/`. Status: **E4_A3A_INCONCLUSIVE_INSUFFICIENT_COVERAGE**. This is a **post-hoc exploratory automated reference-proxy analysis**, not a clinical benchmark or pre-registered study.

## 2. Objective

Explain the repeated short C2a/C3 reports, measure report stability across the four frozen arms and quantify five-finding agreement with existing Indiana reference-report labels where support permits.

## 3. Survey / Technical Motivation

E4-R4 generated 80 reports from 20 frontal Indiana images. E4-A1 found C2a/C3 text identical in 17/20 images and severe labeler missingness. E4-A2 prepared a blinded human review package, but has zero qualified annotations. A3A uses saved artifacts only; the arm mapping and headline result were already known before this protocol was written.

## 4. Environment

CPU-only Python 3 standard-library analysis on the existing Linux server. `TMPDIR`, `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME` and the recorded pytest cache path point under `/storage/student4/hanhpm/`; exact paths are in the run manifest. No GPU or model was loaded. E4-R4's frozen generation configuration, including four K80 devices, is historical input rather than an A3A resource use.

## 5. Dependency Setup

No packages installed or upgraded. The default Python did not include `pytest`; tests ran with standard-library `unittest`. The saved CheXpert rule-based outputs were reused; the labeler was not rerun.

## 6. Code / Notebook Changes

Added `research/e4_a3a.py` and `tests/test_e4_a3a.py`. The original model, prompts, reports, labels, E4-R4 and E4-A2 package were preserved. A3A artifacts were written only under its new run directory; synthetic validation is isolated in `synthetic/`.

## 7. Experiment Configuration

One report per image × arm, 20×4. Prespecified primary C2a versus C3; secondary C2a versus C0 and C1. Five cached findings: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion. Other CheXpert observations are `UNAVAILABLE`. Label states remain Positive, Negative, Uncertain, Not mentioned and Extraction failure. An agreement rate requires at least five images jointly definite in both compared arms and reference for a given finding. Protocol SHA-256: `6bfcb55d2f528f3303dcdc38026dfc7bbfcf94a1e2f233f20f38dfa8ebf530e7`, recorded before comparative metric calculation.

## 8. Code Executed

From repo root, with the five cache variables set under `/storage/student4/hanhpm/.a3a_cache/` and `PYTHONDONTWRITEBYTECODE=1`: `python3 -m unittest discover -s tests -p test_e4_a3a.py -q`; `python3 research/e4_a3a.py --synthetic`; `python3 research/e4_a3a.py`. The SHA-256 input audit was completed first. No inference, training or download command was run.

## 9. Output

The A3A run directory contains the required manifest, frozen protocol, generation diagnosis, report stability CSV, finding coverage CSV, reference-proxy CSV, paired comparison CSV, case analysis, statistical limitations, research decision and artifact index. Extra audit files include input integrity, prompt differences, image-level reference-proxy pairs and isolated synthetic results.

## 10. Output Analysis

SHA-256 matched all manifest-listed input files: 392 R4 sources, 228 R4 outputs, 56 A2 sources and 39 A2 outputs. Both the R4 20×4 report grid and A2 blinded package are complete. Exact C2a/C3 saved chats and all five numeric evidence rows differ for every image; prompt/input hashes differ. The same original system message, user instruction and single `<image>` marker are retained. C0 has 98 input tokens; C1/C2a/C3 have 498. Native generation uses temperature 0.2 and `do_sample=True`, one draw per arm, so sampling noise can affect differences.

## 11. Quantitative Results

| Arm pair | Exact identical reports /20 | Normalized identical /20 |
|---|---:|---:|
| C0–C1 | 0 | 0 |
| C0–C2a | 0 | 0 |
| C0–C3 | 0 | 0 |
| C1–C2a | 3 | 3 |
| C1–C3 | 3 | 3 |
| C2a–C3 | **17** | **17** |

C2a–C3 proxy paired outcome by **image**: improved 1, reduced 0, tied 0, non-evaluable 19. Only two jointly definite image–finding cells in the primary comparison, both from one image; no finding reaches the required five-image threshold. The sole apparent improvement is a reference-text proxy observation, without independent image confirmation. All raw numerators, denominators and unknown states are retained in the CSVs. No confidence interval or sign-test p-value was calculated.

## 12. Qualitative Results

The common short sentence `No acute cardiopulmonary process.` repeats across different evidence inputs. `E4_A3A_case_analysis.md` quotes every C2a/C3 report and any C0/C1 output under 60 characters. CXR2891, CXR486 and CXR981 are reviewed specifically. On CXR981, C2a denies pleural effusion while C3 asserts it and refers to a CT exam; neither is adjudicated as correct.

## 13. Comparison with Baseline

C0 is the original prompt. Equal-text counts and sparse reference-proxy comparisons are descriptive; support differs across arms. There is no defensible ranking of clinical quality. The original E4-R4 macro-F1 values used different jointly definite denominators and are not reused as a claim of CLEAR benefit.

## 14. Problems Encountered

The 20-image cohort has only four reference positives in the five-finding E4-R4 labels, all Cardiomegaly. C2a/C3 often omit finding-specific statements. Report references may cover lateral views while model input was frontal. The exact rendered prompt strings were not persisted; frozen chat content and rendered-prompt/input-ID hashes support comparison. The cause of short generation cannot be isolated from padding, context wording, decoding and model preference without a new experiment.

## 15. Conclusion

**VERIFIED:** input integrity, complete paired grid, materially different C2a/C3 evidence on 20/20 images and identical outputs on 17/20. **AUTOMATED PROXY:** sparse report-reference agreement; primary comparison lacks coverage. **UNVERIFIED:** image-grounded correctness, omission/hallucination rates, clinical benefit and causal use of CLEAR evidence. Decision: `E4_A3A_INCONCLUSIVE_INSUFFICIENT_COVERAGE`; retain `HOLD_MORE_EVIDENCE` and `HOLD_COMPLEX_FUSION`.

## 16. Next Experiment

Recommendation only: define a separate E4-B development cohort and test a shorter prompt without neutral `x` padding, controlled greedy decoding or repeated seeds, and the shuffled control. Each target finding needs at least five reference-definite images with definite outputs in both compared arms just to meet the minimal proxy threshold; a paper-grade clinical study needs more and qualified image-based review. At the historical ~24 seconds/report on four K80 devices, four arms ×100 images imply ~2.7 hours of generation before setup/evaluation; two parallel four-device workers on eight visible devices remain an unverified capacity estimate. No E4-B run was started.

## 17. Reproducibility Checklist

Protocol hash frozen before metrics; inputs hashed and complete; per-image/arm keys checked; cached labels linked; five-state mapping tested; denominators and sparse statuses saved; synthetic dry run isolated; code and test source saved; output hashes indexed. No independent clinical annotations were fabricated. The Git HEAD identifies a base revision; source SHA-256 in the manifest identifies the local A3A script.

## 18. Short Lab Update

A3A finished as a CPU-only post-hoc exploratory audit. Different CLEAR evidence reached the frozen prompts in all 20 cases, but C2a/C3 outputs matched in 17 and the proxy evaluation was evaluable for only one image. Keep the research and fusion holds; review prompt format and design a separate, adequately supported study before further claims.
