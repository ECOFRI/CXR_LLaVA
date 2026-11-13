# P1 official asset evidence matrix

Statuses mean verified public location, not local executability. No paper result or clinical claim was reproduced. Access, exact weight license, and local compatibility remain UNKNOWN where indicated.

| Model | Primary paper | Official code | Official checkpoint | Classification / decision |
|---|---|---|---|---|
| ClinicalBLIP | [JMIR paper](https://doi.org/10.2196/32690) | UNVERIFIED | UNVERIFIED | Paper describes an InstructBLIP-based method; no official runnable assets verified, so no script. |
| R2-LLM | [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/download/29826/31434) | [Official repository](https://github.com/synlp/R2-LLM), includes `train.py` and `generate_reports.py` | README external links; access/license UNVERIFIED | Official code verified, exact runnable checkpoint unverified; no script or download. |
| META-CXR | paper URL UNKNOWN | [Official repository](https://github.com/DasithEdirisinghe/META-CXR), includes `inference.py` | README Google Drive link; access/license UNVERIFIED | Official inference code verified; training entrypoint UNKNOWN; no script or download. |
| RRG-LLM | [Proceedings DOI](https://doi.org/10.3233/SHTI250976) | UNVERIFIED | UNVERIFIED | Treat future implementation as reconstruction only if authorized. |
| Swin-Qwen3 | [Frontiers paper](https://www.frontiersin.org/journals/radiology/articles/10.3389/fradi.2026.1875905/full) | UNVERIFIED | UNVERIFIED | Local Qwen3-0.6B does not verify the combined model; reconstruction only if authorized. |
| BLIP-2 OPT 2.7B | [BLIP-2 paper](https://arxiv.org/abs/2301.12597) | [LAVIS](https://github.com/salesforce/LAVIS) | [Official Salesforce model](https://huggingface.co/Salesforce/blip2-opt-2.7b) | Official candidate; metadata-only dry-run completed. |

The status is bounded to the listed official sources and targeted local inventory. UNVERIFIED does not prove absence anywhere. No code or checkpoint for these paper models was downloaded or run. See `manifests/papers.csv` and `reports/P1_blip2_candidates.md`.

## P2.6 source audit update

R2-LLM source pinned at `2167adf5bf3a204ad85e7065aad74fa1ebb9640e` (MIT), META-CXR at `e97d70900dd92c2349b98e60decb5c27e6b1e121` (root license UNKNOWN). R2 has train/infer and IU/MIMIC JSON adapters; META has train/infer and MIMIC/Indiana data adapters. Checkpoint links exist but exact asset access/license and K80 compatibility remain UNKNOWN. See `reports/P2_repo_audit.md`. No weights were fetched.

## P3.1 evidence matrix — 2026-10-22

| Model | Q-Former class | Official code / weights | License | K80 |
|---|---|---|---|---|
| ClinicalBLIP | MODIFIED_QFORMER | UNKNOWN / UNKNOWN | code UNKNOWN; weights UNKNOWN | BLOCKED: no official weights/code verified; V100-specific run not reproduced |
| RRG-LLM | ORIGINAL_QFORMER | UNKNOWN / UNKNOWN | code UNKNOWN; weights UNKNOWN | BLOCKED: code/weights unverified and Vicuna base exceeds single K80 fp16 |
| Swin-Qwen3 | QFORMER_STYLE_ATTENTION | UNKNOWN / UNKNOWN | code UNKNOWN; weights UNKNOWN | BLOCKED: combined checkpoint/code unverified; Qwen3 runtime on cu102 UNKNOWN |
| R2-LLM | ORIGINAL_QFORMER | https://github.com/synlp/R2-LLM / NOT_PUBLICLY_RELEASED (QA HF author listing) | code MIT (third_party/R2-LLM/LICENSE:1); weights UNKNOWN; Vicuna/LLaMA base terms additional | BLOCKED: weights inaccessible; 13B fp16 ~26GB base alone; bnb int8 needs sm75 |
| META-CXR | MODIFIED_QFORMER | https://github.com/DasithEdirisinghe/META-CXR / LINK_ONLY_FILES_UNKNOWN | code UNKNOWN; vendored LAVIS LICENSE does not cover repo; weights UNKNOWN; Vicuna/LLaMA base terms additional | BLOCKED: asset filename/license unknown; 7B fp16 ~14GB base alone |
| BLIP-2 | ORIGINAL_QFORMER | https://github.com/salesforce/LAVIS / LOCAL_VERIFIED_P1; X02_SMOKE_P2 | code LAVIS license UNKNOWN in this audit; weights HF model card terms UNKNOWN in this audit | X02 smoke technical fit 7.57GB GPU0; 10/10 valid; no clinical claim |

Exact metrics, datasets/splits, stages, versions, and evidence locations are in `manifests/papers.csv`. R2 paper [Tables 2-3](https://ojs.aaai.org/index.php/AAAI/article/download/29826/31434) reports IU/MIMIC test results; no cross-paper score is directly comparable with X00. ClinicalBLIP [Tables 1-2](https://formative.jmir.org/2024/1/e32690/) and Swin-Qwen3 [Tables 1-2](https://www.frontiersin.org/journals/radiology/articles/10.3389/fradi.2026.1875905/full) use different test definitions. RRG [Table 1](https://journals.sagepub.com/doi/10.3233/SHTI250976) reports ROUGE-L/METEOR only. META IEEE full text was inaccessible (HTTP 202); its [README Results](https://github.com/DasithEdirisinghe/META-CXR#-results) supplies limited values and no verified paper table. BLIP-2 [paper](https://arxiv.org/abs/2301.12597) has no CXR table.

### P3.2 asset verdict

R2 and META checkpoint chains are BLOCKED pending exact files, access and license evidence. See `reports/P3_weights_licensing.md` and `manifests/checkpoints.csv`; no weights fetched.

### P3.3 implementation distinction

R2 official checkpoint is **NOT_PUBLICLY_RELEASED** per Claude/QA HF author-list audit. X06 local BLIP-2 bridge arms are REIMPLEMENTED, never REPRODUCED R2/RRG models; see `reports/P3_implementation_audit.md`. ClinicalBLIP, META, and Swin-Qwen3 remain skeleton/blocker-only.

### P3.4 environment feasibility

Existing K80 env fails R2 import-only probe on missing `omegaconf`; no model was loaded. Proposed cu102 isolated environment is PLAN_ONLY, not solver verified. See `reports/P3_legacy_env_plan.md`.
