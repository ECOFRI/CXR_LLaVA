# VLM MEDICAL — CLAUDE ORCHESTRATION / CODEX GPT-6-SOL-LIGHT MASTER PLAN V4

**Ngày lập:** 2026-10-22  
**Điều phối:** Claude (orchestrator). **Thực thi mã:** Codex GPT-6-sol-light (single coding worker).  
**Mục tiêu:** Kiểm kê máy Linux + dữ liệu thật, tái sử dụng hệ sinh thái hiện tại, benchmark mọi nhánh BLIP-2/Q-Former cho CXR report generation khả thi và sinh script để chạy mọi nhánh training; lưu log, kết quả, tiến độ và bằng chứng nguồn.

> Tài liệu này là lệnh triển khai cho tác nhân trên **server có quyền truy cập**. Những đường dẫn, Conda environments, GPU, trọng số và trạng thái dataset trên server **chưa được xác minh** trong phiên soạn plan. Không biến giả định thành kết quả hoàn thành. Dự án là nghiên cứu, không phải hệ thống sử dụng lâm sàng.

## 0. Quy tắc điều phối và ngân sách tác nhân — bắt buộc

1. **Claude làm PM/QA và giữ một coding worker duy nhất là Codex GPT-6-sol-light.** Không tạo agent song song, multi-agent debate, recursive delegation, hoặc lặp hỏi đáp vô ích.
2. Giao **một work order ngắn mỗi phase**, bao gồm scope, artifact phải sinh, acceptance criteria, danh sách file cần đọc. Codex chỉ trả về tóm tắt <= 15 dòng cùng paths, status, lỗi chặn, không paste log dài.
3. Codex chạy **audit đọc dữ liệu, unit tests, CPU-based evaluation** khi an toàn. GPU inference chỉ chạy lần lượt và theo ngân sách được Claude khóa trước khi thực thi. **Training/fine-tuning/full-dataset inference: viết đầy đủ script/config, chỉ chạy sau phê duyệt explicit.** Không khởi chạy jobs ngầm chỉ vì script đã tạo.
4. Không giữ Claude/Codex online để chờ training. Nếu job đã được phê duyệt, dùng `sbatch` khi có scheduler hoặc `nohup`/`tmux` dạng detached cùng job manifest/PID/log; **không polling bằng LLM**, không spam trạng thái, không chạy nhiều GPU jobs cùng lúc nếu chưa có quota.
5. Ưu tiên tái sử dụng code/checkpoint/Conda/cache, không tải dữ liệu/weights, không cập nhật Conda/PyTorch/CUDA hàng loạt, không sửa source/repo cũ. Nếu thiếu dependency: chụp compatibility matrix trước, đề xuất bản env **mới** theo prefix thay vì phá env cũ.
6. Tuyệt đối không chạy `sudo`, `rm -rf`, `conda env remove`, bulk `pip install -U`, chỉnh `/storage/shared`, tự ý push code, tự ý upload/re-share DUA datasets hoặc xóa checkpoints. Tất cả dataset là read-only.
7. Không dùng nhãn **REPRODUCED** nếu chưa khớp input, split, checkpoint, decoding, preprocessing và metric protocol paper. Dùng **REIMPLEMENTED**, **PARTIAL**, **BLOCKED**, **SKIPPED_WITH_REASON** khi phù hợp.
8. Không sử dụng nhãn test/ground-truth report để tạo prompt evidence hay chọn threshold, model, epoch, adapter. Không kết luận chỉ từ BLEU hay chỉ từ F1.
9. Không ghi clinical report nguyên văn, patient IDs hay ảnh bệnh nhân vào báo cáo công khai. Báo cáo nội bộ chứa case identifier đã kiểm soát, chỉ lưu theo chính sách quyền truy cập dataset.
10. Mọi run phải **idempotent**: có config snapshot, deterministic seed, hash/commit, data manifest, run_id, resume, stdout/stderr log, metrics JSON/CSV, report `.md`, `status.json`.

## 1. Research questions và baseline đã được ghi nhận

**RQ-A:** CXR-LLaVA tạo báo cáo gần ground truth đến mức nào (BLEU-1/2/3/4, ROUGE-L, METEOR, clinical F1)?  
**RQ-B:** Các hệ BLIP-2/Q-Former có vượt baseline CXR-LLaVA **trên cùng cohort và metric implementation** không?  
**RQ-C:** Frozen vision + trainable projection/Q-Former so với fine-tuning LoRA có lợi ích đáng kể không?  
**RQ-D:** CLEAR feature hoặc finding/concept evidence có cải thiện report fidelity mà không tăng hallucination không?  
**RQ-E:** Kết quả có tái lập trên dữ liệu độc lập, không contamination và với runtime hợp lý không?

**Historical context (KHÔNG phải run mới):** Indiana local CXR-LLaVA đã sinh/đánh giá **3,667/3,667** báo cáo (macro-F1 local **0.4457**, nhạy với missing/uncertain); CLEAR NIH exploratory AUROC **0.7671** vs CXR-LLaVA **0.6312**; E5 DEV Macro AP fusion **0.2515** vs CLEAR-only **0.2566**; E4 pilot có **264** báo cáo từ **24** ảnh; E4-A2 blind rating vẫn chưa hoàn thành. Các kết quả thuộc cohort/protocol khác nhau nên không gom vào một ranking trực tiếp.

## 2. Pha P0 — Server, Git, GPU, Conda, path audit (đọc, không sửa)

### 2.1. Câu hỏi phải trả lời

- User, hostname, kernel, working directory, nhóm quyền; vị trí project root thật, Git remotes/branches/commits, disk free/quota; scheduler có hay không.
- GPUs: model, số lượng, VRAM, compute capability, driver; K80 được báo cáo trong các run cũ nhưng phải **verify lại**. CUDA toolkit, driver, PyTorch, available BF16/FP16, torch distributed compatibility.
- Mọi env Conda: `conda env list`, python, torch/transformers/peft/accelerate/bitsandbytes/deepspeed nếu được cài; identify env chạy được CXR-LLaVA và CLEAR. `conda run -n NAME` hoặc `conda run -p PREFIX` thay vì thay đổi môi trường global.
- Dataset roots thực tế tại `/storage/shared/` và project-local data. Scan aliases: `mimic`, `mimic-cxr`, `mimic-cxr-jpg`, **`minic`**, `vindr`, `vindxr`, `indiana`, `iu`, `chexpert`, `nih`, `padchest` (case-insensitive). Khẳng định alias sau khi inspect metadata, không suy từ tên thư mục.
- Các repo/checkpoints/output đã có: CXR-LLaVA, CLEAR, project E0/E1/E2/E4/E5, report JSONL/CSV, labeler, IU references; checkpoint provenance SHA/license.
- Trước khi làm gì: không dùng `du -sh` toàn `/storage/shared` khi có nguy cơ I/O lớn; listing độ sâu thấp và sample metadata trước.

### 2.2. Lệnh audit mẫu (chỉ chạy trên server đúng quyền)

```bash
pwd; whoami; hostname; uname -a
command -v conda || true; conda env list 2>/dev/null || true
nvidia-smi -L 2>/dev/null || true; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv 2>/dev/null || true
df -h /storage/shared "$HOME" 2>/dev/null || true
find /storage/shared -maxdepth 3 -type d 2>/dev/null | head -n 300
find /storage/shared -maxdepth 3 -type f \( -iname '*metadata*.csv*' -o -iname '*split*.csv*' -o -iname '*report*.csv*' -o -iname '*report*.txt' -o -iname '*annotation*.csv*' \) 2>/dev/null | head -n 250
```

Không log secrets/env variables nhạy cảm; không `find / -type f` toàn filesystem. Nếu thiếu quyền, ghi `NO_ACCESS` và command output tóm tắt; không tự xin root.

### 2.3. P0 deliverables

- `reports/00_system_inventory.md`: thật/đã kiểm chứng vs chưa biết.
- `manifests/conda_envs.csv`, `manifests/gpu_inventory.json`, `manifests/code_artifacts.csv`, `manifests/checkpoints.csv`.
- `configs/paths.local.yaml` (không commit absolute private paths nếu có thông tin hạn chế), `configs/paths.example.yaml`.
- `reports/01_dataset_inventory.md` và `manifests/datasets.csv` với columns: name/version/root/num_images/num_studies/num_patients/report_available/labels_available/view_types/splits/permissions/leakage_risk.
- Bảng `reuse vs missing vs incompatible` và đề xuất env cho mỗi paper.

**Gate P0:** Không mở rộng experiments khi chưa có paths thực sự và env xác minh.

## 3. Pha P1 — Dataset integrity, provenance và cohort manifests

### 3.1. Tác vụ theo dataset

- **Indiana/IU:** Audit mapping ảnh ↔ study ↔ reference report, `Findings`/`Impression`, ảnh frontal/lateral, trùng study; xác minh 3,667 local run với dataset version; lấy cohort khác khi cần paper split, không tự map 1 ảnh thành multi-view study.
- **MIMIC-CXR:** Phân biệt ảnh `MIMIC-CXR-JPG` và **bản có free-text reports** của MIMIC-CXR; JPG+CheXpert labels không chứng minh report text đã có trên server. Xác minh `subject_id`, `study_id`, `dicom_id`, split, report sections, permissions; dùng patient/study-grouped split.
- **VinDr-CXR:** Có labels và boxes phục vụ classification/localization, **không mặc định có free-text ground-truth reports** nên BLEU = N/A trừ khi xác minh được phiên bản có reports được cấp phép. Giữ nguyên ontology mapping và test set 3,000 nếu đúng phiên bản.
- **CheXpert/NIH:** Classification/probe; không mặc định report-BLEU. NIH/CheXpert/MIMIC/VinDr có thể có **pretraining exposure** với checkpoint gốc. Tách **within-experiment leakage** và **pretraining contamination**.
- Datasets có hạn chế truy cập: chỉ dùng khi quyền/license phù hợp, không sao chép vào public repo, không tự download.

### 3.2. Output và integrity tests

- `scripts/data/audit_datasets.py --paths configs/paths.local.yaml --output ... --sample-only`
- `scripts/data/build_study_manifest.py --dataset indiana|mimic|vindr ...`
- `scripts/data/audit_overlap.py --manifest ... --known-pretrain-manifest ...`
- `scripts/data/validate_splits.py --group-by subject_id --strict`
- `tests/test_study_linkage.py`, `tests/test_split_disjoint.py`, `tests/test_missing_reports.py`, `tests/test_label_mapping.py`.
- `manifests/{dataset}_{version}_{split}.parquet/csv` chứa **relative paths**, IDs cần thiết, dataset source/version, labels/report paths (không duplicate raw image).

**Gate P1:** Primary report benchmark chỉ dùng cohort có reference reports ghép đúng và split hợp lệ; VinDr ưu tiên F1/AUROC/AUPRC thay vì BLEU.

## 4. Pha P2 — Reuse output hiện có; chuẩn hóa evaluation suite

**Ưu tiên cao nhất về chi phí thấp:** không generate lại 3,667 Indiana reports hay 264 E4 pilot reports nếu artifacts còn đầy đủ.

### 4.1. Common output contract

`predictions.jsonl` cho mỗi study gồm `run_id`, `dataset`, `split`, `study_id`, `image_relpaths`, `reference_sections`, `generated_sections`, `model_name`, `checkpoint_revision`, `prompt_version`, `decoding`, `seed`, `inference_sec`, `evidence_arm`; pathology scores lưu **riêng** `scores.parquet` dạng continuous `[study_id, finding, score]`.

### 4.2. Metrics

- **Report:** corpus BLEU-1/2/3/4 (fix tokenizer/normalization, lưu corpus + optional sentence BLEU riêng), ROUGE-L, METEOR; RadGraph-F1/RadCliQ nếu dependencies và format phù hợp; phân tích clinically important omissions/hallucination/laterality/severity và E4-A2 blind ratings khi đủ reviewer.
- **Clinical labels:** CheXpert/NegBio/CheXbert theo 1 version cố định, per-class precision/recall/F1, micro/macro F1, support, coverage. Đánh giá `blank`/`uncertain` sensitivity **ngoài** primary policy.
- **Classification:** AUROC và AUPRC chỉ nếu có **continuous scores** cho cả dương và âm; positive/negative support; F1 tại ngưỡng fixed trên validation. **Không suy diễn AUROC từ generated reports hoặc binary labels.**
- Patient-level paired bootstrap 95% CI; raw metric và delta với baseline cùng protocol. Một model không chạy trên cùng cohort không được xếp hạng trực tiếp với model khác.
- Cả lexical và clinical scores phải có thêm QC cases. BLEU cao nhưng F1 giảm phải báo rõ.

### 4.3. Scripts bắt buộc

- `scripts/eval/match_existing_reports.py`
- `scripts/eval/evaluate_reports.py --predictions ... --metrics bleu1,bleu2,bleu3,bleu4,rouge_l,meteor`
- `scripts/eval/evaluate_clinical.py --predictions ... --labeler chexpert --missing-policy fixed`
- `scripts/eval/evaluate_classification.py --scores ... --labels ... --metrics auroc,auprc,f1`
- `scripts/eval/paired_bootstrap.py --group study_or_patient --seed 42`
- `scripts/eval/compare_runs.py --baseline ... --candidate ... --strict-cohort`
- `tests/test_bleu_known_examples.py`, `tests/test_auc_requires_scores.py`, `tests/test_same_cohort.py`, `tests/test_report_sections.py`.

**P2 output:** `reports/02_historical_baselines.md`; `results/benchmarks/{dataset}/baseline_metrics.csv`, `baseline_metrics.json`; summary comparison table và error gallery danh tính an toàn.

## 5. Pha P3 — Paper catalog (verified references) và coverage tracking

Phân loại rõ Q-Former **chính thức** (BLIP-2/InstructBLIP), **kế thừa** (MiniGPT-4), và **Q-Former-inspired/custom**. BLIP-2 gốc là general-image model, không mặc định là pretrained CXR model. Với mọi paper: paper PDF/DOI, code repo/status, licensed checkpoints, trainable/frozen modules, pretrained data membership, dataset/split/metrics, train costs, native pipeline commands, reproducibility gaps. Không mô tả custom adaptation là paper reproduction.

| ID | Paper và nguồn | Q-Former status | Code/weight status phải audit | Thí nghiệm tối đa theo điều kiện |
|---|---|---|---|---|
| B2 | **BLIP-2 (2023)** — https://arxiv.org/abs/2301.12597 ; https://github.com/salesforce/LAVIS/tree/main/projects/blip2 | Q-Former gốc | LAVIS; verify pretrained model requirements | Frozen zero-shot smoke; fixed queries, resize, projection; adapter training scripts nếu hợp lệ |
| CB | **ClinicalBLIP (2024)** — https://doi.org/10.2196/32690 | InstructBLIP + Q-Former, LoRA multi-stage | Chưa xác minh official code/weights; audit trước | Pretrained InstructBLIP inference; stage 1 visual LoRA, stage 2 joint LoRA, prior-prompt ablation; script-only nếu thiếu resources |
| R2 | **R2-LLM (AAAI 2024)** — https://ojs.aaai.org/index.php/AAAI/article/view/29826 ; https://github.com/synlp/R2-LLM | MiniGPT-4 ViT + Q-Former (kế thừa) | Official repo có `train.py`, `generate_reports.py`, `environment.yaml`, configs; verify weights | Official inference; stage 1 alignment/pretraining; stage 2 fine-tuning; coarse-to-fine, I3 ablations |
| MC | **META-CXR (IEEE Access 2025)** — https://doi.org/10.1109/ACCESS.2025.3606961 ; https://github.com/DasithEdirisinghe/META-CXR | META-Former, modified Q-Former + multi-encoder | Repo hiện có `inference.py`, model/pretraining modules; verify published checkpoint and training entrypoint | Inference checkpoint; multi-encoder vs single; MHCAC/evidence off/on; scripts for missing training steps |
| RR | **RRG-LLM (2025/2026 indexed)** — https://doi.org/10.3233/SHTI250976 | MiniGPT-4/R2-LLM Q-Former frozen; projection + LoRA | Official public implementation chưa xác minh | Method reimplementation using pinned R2-LLM; projection-only vs projection+LoRA; semantic ranking ablation |
| SQ | **Swin-Qwen3 (2026)** — https://doi.org/10.3389/fradi.2026.1875905 | Q-Former-**style** cross-attention, không phải BLIP-2 original | Official code/checkpoint chưa xác minh | 1-stage/2-stage/3-stage offline report generation; Qwen3-0.6B adapter/LoRA script if environment permits |
| XL | **XrayGPT / PCLmed (optional)** — https://aclanthology.org/2024.bionlp-1.35/ ; https://ceur-ws.org/Vol-3497/paper-146.pdf | Kiểm tra **thực tế** Q-Former qua source; PCLmed không phải benchmark CXR chuyên biệt | Verify code | Chỉ thêm vào matrix nếu đủ evidence architecture + report/CXR relevance; nếu không survey-only |

**Coverage goal:** Tạo `reports/03_paper_evidence_matrix.md` và `manifests/papers.csv` với từng phương pháp, official/open code/revision, metric tables original, dataset, supported run arms, train requirements, `PASS/PARTIAL/BLOCKED/SKIPPED_WITH_REASON`. Mọi paper trong danh sách phải có **một kết luận khả thi/không khả thi bằng chứng**, không bắt buộc brute-force train khi thiếu weights/compute.

## 6. Pha P4 — Experiment matrix: lần lượt, giá trị cao trước, toàn bộ khả thi

### 6.1. Thứ tự thực hiện

| EXP ID | Đầu vào | Thí nghiệm / đối chứng | Training? | Expected output | Điều kiện thực thi |
|---|---|---|---|---|---|
| X00 | Indiana existing | Re-score BLEU/clinical F1, E4 264 reports; kiểm tra missing/uncertain | Không | Baseline locked | Chạy CPU sau P1 |
| X01 | IU/MIMIC | CXR-LLaVA released checkpoint matched inference | Không | Reference baseline | GPU inference *authorized* |
| X02 | IU/MIMIC | BLIP-2/InstructBLIP released zero-shot image→report prompt | Không | General VLM control | Checkpoint, env, licensing, authorized |
| X03 | IU/MIMIC | ClinicalBLIP native checkpoint nếu có, rồi InstructBLIP LoRA/prompt reproduction | Có ở stages | ClinicalBLIP method comparison | P3 code/weights verified; training script prepared |
| X04 | IU/MIMIC | R2-LLM official checkpoint inference; draft vs refine | Có ở stages | R2-LLM controlled baseline | repo + weights present; scripts prepared |
| X05 | IU/MIMIC; VinDr classification | META-CXR pretrained model inference; single/multi-encoder; MHCAC yes/no | Có ở stages | META-CXR controlled ablations | actual checkpoint compatibility; scripts prepared |
| X06 | IU/MIMIC | RRG-LLM projection-only vs LoRA adaptation; semantic ranking off/on | Có | RRG-LLM method reconstruction | stage script prepared; not exact paper reproduction |
| X07 | IU paired-view | Swin-Qwen3 draft, verify, refine 1/2/3-stage, overhead | Có (adapter/LoRA) | QFormer-style control | matching views + available framework; prepare scripts |
| X08 | matched cohort | CLEAR encoder + linear bridge vs same encoder + Q-Former | Có | Q-Former incremental effect | valid weights, paired design; scripts first |
| X09 | matched cohort | CXR-LLaVA encoder + linear bridge vs same encoder + Q-Former | Có | bridge-vs-encoder ablation | matched data/config; scripts first |
| X10 | matched cohort | CLEAR prompt evidence arms C0 original, C1 neutral, C2a findings, C3 shuffled | Không | Does evidence help output? | locked prompts, no GT leakage; existing artifacts reuse |
| X11 | VinDr/eligible classification | CLEAR/CXR/META finding scores AUROC/AUPRC/F1 | Có nếu probe | Pathology benchmark | verify ontology and contamination |

### 6.2. Chế độ chạy bắt buộc theo 3 mức

- **Tier A — Offline reuse (automatic):** Data audit, generate manifests, re-score saved reports, tests, no training/GPU.
- **Tier B — Bounded inference (after Claude checks explicit resource permission):** 10 cases smoke → <=100 case pilot → 500 case evaluation → full approved split only if needed. All jobs sequential, compare output validity and memory first.
- **Tier C — Training/finetuning (scripts-only until owner APPROVES each train job):** projector training, LoRA, stage 1+2, custom Q-Former, clinical prior, META-CXR multi-encoder. For every training arm create `train_*.py`, `train_*.sh`, config YAML, dry-run validation mode, checkpoint saving/resume; **no actual optimization starts** by default.

Một experiment khả thi nếu đủ dữ liệu, license, repo/weights, environment, metric protocol và ngân sách. Nếu không đủ, ghi `BLOCKED` hoặc `SCRIPT_READY_NOT_RUN` cùng bằng chứng, không cố chạy tốn GPU/LLM token.

## 7. Pha P5 — Linux repository structure (bắt buộc, không phá repo hiện có)

**Đầu tiên tìm project root hiện có**, rồi tạo một thư mục nghiên cứu versioned dưới đó (không đoán `/home/...`):

```text
<VERIFIED_PROJECT_ROOT>/
  research/blip2_cxr_v4/
    README.md
    pyproject.toml               # chỉ nếu không ảnh hưởng source sẵn có
    configs/
      paths.example.yaml
      paths.local.yaml            # .gitignore if private
      benchmark.yaml
      papers/{blip2,clinicalblip,r2llm,meta_cxr,rrg_llm,swin_qwen3}.yaml
      experiments/{X00,...,X11}.yaml
    src/vlm_medical/
      data/{datasets.py,manifests.py,preprocess.py,ontology.py}
      models/{blip2.py,clinicalblip.py,r2llm.py,meta_cxr.py,rrg_llm.py,swin_qwen3.py,clear_adapter.py}
      eval/{text_metrics.py,clinical_metrics.py,classification_metrics.py,bootstrap.py}
      utils/{config.py,provenance.py,logging.py,compute.py}
    scripts/
      audit/{system_audit.sh,env_audit.sh,dataset_audit.sh}
      data/{audit_datasets.py,build_study_manifest.py,validate_splits.py,audit_overlap.py}
      eval/{match_existing_reports.py,evaluate_reports.py,evaluate_clinical.py,evaluate_classification.py,compare_runs.py,paired_bootstrap.py}
      experiments/{infer_blip2.py,infer_clinicalblip.py,infer_r2llm.py,infer_meta_cxr.py,infer_rrg_llm.py,infer_swin_qwen3.py}
      train/{train_clinicalblip.py,train_r2llm_stage1.py,train_r2llm_stage2.py,train_meta_cxr.py,train_rrg_llm.py,train_swin_qwen3.py,train_clear_bridge.py,train_qformer_adapter.py}
      run/{dry_run.sh,submit_approved.sh,collect_results.py,make_report.py}
    third_party/                 # pinned external repos, no edits to upstream working copy
    manifests/
    results/benchmarks/
    runs/<RUN_ID>/{config.yaml,run_manifest.json,status.json,stdout.log,stderr.log,predictions.jsonl,scores.parquet,metrics.json,report.md}
    reports/{00_system_inventory.md,01_dataset_inventory.md,02_historical_baselines.md,03_paper_evidence_matrix.md,04_experiment_tracker.md,05_results_and_decisions.md}
    tests/
```

**Code rules:** Không đổi tên biến/đường dẫn đang hoạt động trong repo gốc; modular adapters; all scripts có `--help`, `--config`, `--output-dir`, `--dry-run`, `--resume`; kiểm tra path, missing file, CUDA support và invalid metric early. `--dry-run` không load model lớn/training. Multi-view phải được thể hiện trong manifest; không lặng lẽ đổi single-view thành dual-view.

**Example usage — chỉ là CLI contract cần Codex triển khai; không giả vờ script đã có:**

```bash
export PROJECT_ROOT="<VERIFIED_PROJECT_ROOT>"
cd "$PROJECT_ROOT/research/blip2_cxr_v4"
conda run -n <VERIFIED_COMPATIBLE_ENV> python scripts/data/audit_datasets.py \
  --paths configs/paths.local.yaml --output reports/01_dataset_inventory.md --sample-only
conda run -n <VERIFIED_COMPATIBLE_ENV> python scripts/eval/evaluate_reports.py \
  --predictions runs/<BASELINE_RUN>/predictions.jsonl \
  --output-dir results/benchmarks/indiana --metrics bleu1,bleu2,bleu3,bleu4,rouge_l,meteor
# only after explicit train approval:
bash scripts/run/submit_approved.sh --experiment X08 --approval-file approvals/X08.yaml
```

## 8. Pha P6 — Environment strategy và training scripts

**Conda priority:** reuse working env if compatible → conda clone/prefix isolated if needed → create new env via locked YAML only if unavoidable; offline pip wheels first, no network installs without approval. Per model `environment_report.md` documents package versions, driver/GPU constraints, known unsupported features (K80 has no BF16; check FP16/bitsandbytes/CUDA toolkit before enabled). `nvidia-smi` driver alone != nvcc; check torch CUDA compatibility inside each env. Do not globally `conda activate` in unattended shell; use `conda run` or absolute env interpreter.

**Per-paper training scripts must implement:**

1. CLI + YAML configs, seed, split manifest, image preprocessing matching paper, reference section, tokenizer/model revision, parameters `freeze_vision`, `freeze_qformer`, `train_projector`, `lora_r/alpha`, gradient accumulation, bf16/fp16 guard.
2. Preflight: weights available, license, data permission, device/dtype, free VRAM, estimated parameter count, disk, sample count, learning rate and effective batch; fail fast if unknown.
3. `--dry-run` builds parsed config, checks paths and shape expectations using small synthetic input where possible, without optimizer steps or huge model loading.
4. Actual `--train` requires explicit approval file containing signed-by-person identifier/text, model, dataset, script, max walltime, GPU quota, memory budget and output dir. Codex must NOT generate fake approval.
5. Resume from interrupted checkpoint with optimizer/scheduler state, graceful save on SIGTERM, periodic metrics checkpoint and `train_history.csv`.
6. A script wrapper may submit detached jobs **only after** approval and record PID/Slurm job ID. Agent ends turn immediately, does not wait for completion; subsequent separate user/Claude invocation gathers results.
7. Provide exact start command in `reports/04_experiment_tracker.md` even for blocked scripts; never claim it ran.

**Per-paper stage mapping:**

- ClinicalBLIP: pretrained InstructBLIP/QFormer inference → visual adaptation stage script → joint LoRA/QFormer and clinical-prior ablation script; separate `no-prior`, `predicted-prior`, `GT-prior-TRAIN-ONLY` controls to prevent label leakage.
- R2-LLM: official inference, stage1 alignment/projection, stage2 in-domain adaptation; draft/refine off/on; estimate Vicuna license/weights feasibility and skip if unavailable.
- META-CXR: official inference if available; single vs multi-encoder; META-Former vs simple projection if code supports; MHCAC off/on; prepare training entrypoints if upstream lacks published script, label as method recreation.
- RRG-LLM: freeze ViT/QFormer, train projection + Vicuna LoRA; semantic-retrieval/contrastive ranking ablations; requires verified R2 baseline.
- Swin-Qwen3: frozen Swin, train QFormer-style+Qwen3-0.6B LoRA; one-/two-/three-stage inference ablations; respect paired-view test split and 14.8× reported inference overhead as motivation to measure runtime, not as measured local value.
- CLEAR bridge study: CLEAR patch features + linear projector **versus** same CLEAR features + Q-Former; identical frozen LLM, same training cohort and compute budget; optional CXR encoder matched study; no direct assumption concept bank exists.

## 9. Progress, reporting và user-facing handoff

**Two layers of reporting:**

1. **Always local source of truth** `reports/04_experiment_tracker.md` + `reports/05_results_and_decisions.md` + `runs/<id>/metrics.json`.
2. **Google Docs** `VLM Medical` → parent tab `Report 2/11` → proposed subtab `BLIP-2 / Q-Former Benchmark Progress` **only after verifying Docs write access and when Claude has connected Google Docs credentials/explicit authorization**. Do not let coding agent invent Google credentials. Do not edit historical tabs. Local reports remain authoritative; sync concise milestone status/results to Google Docs no more than once per completed phase.

**Tracker fields:** Date, phase, experiment_id, dataset+split+manifest sha, model/checkpoint, repo SHA, Conda env/prefix, device, run_id, status (`PLANNED/RUNNING/COMPLETE/PARTIAL/BLOCKED/SCRIPT_READY_NOT_RUN/FAILED`), methods summary, metrics (BLEU1-4/ROUGE/METEOR/clinical F1/AUROC where valid), confidence intervals, runtime, VRAM, artifacts links/paths, blocker, next action. Keep `RUNNING` truthful when Claude detaches; update to COMPLETE only after collector verifies exit status + files.

**Report after each phase (compact):**

```text
Phase / date / result: [COMPLETE|PARTIAL|BLOCKED]
Verified assets: dataset paths, env and checkpoint refs
Work done: scripts/tests/experiments actually executed
Numerical results: metric + cohort + N + 95% CI + missing policy
Limitations: overlap, missing labels/weights, unsupported metrics
Artifacts: exact absolute or project-relative paths
Next step: one concrete work order; training approval required? YES/NO
```

**Acceptance criteria:** Results are recalculable from manifests and scripts; all claimed scores have saved raw prediction artifacts; every attempted paper has status and blockers; no destroyed existing runs; smoke/run status auditable; no fabricated benchmark comparison.

## 10. Stopping / decision rules

- P0 audit fails → stop GPU work; write blocker report.
- Corpus BLEU cannot run because no matched reference → fix linkage first; do not score label-only dataset.
- AUROC unavailable because only binary/generated labels → mark N/A, do not invent scores.
- BLIP2 paper lacks code/weights → `PARTIAL` (method reimplementation) or `BLOCKED` (no weights), not `REPRODUCED`.
- Training resource unknown or too large → `SCRIPT_READY_NOT_RUN`; never attempt job to learn cost empirically without approval.
- Existing full Indiana baseline already available → re-score, do not regenerate by default.
- Hold complex fusion and full LLM fine-tuning until matched baseline+controls and data-provenance gates pass.
- Proposals to claim clinical safety must await clinical evaluation; this plan benchmarks research models only.

## 11. Deliverables / checklist Claude must enforce

- [ ] P0: server + Conda + GPU + checkpoints + `/storage/shared/` inventory with verified absolute paths.
- [ ] P1: Indiana/MIMIC/VinDr manifests, report availability, permissions, train/test/pretraining-overlap audits.
- [ ] P2: 3,667 Indiana baseline BLEU-1/2/3/4 + clinical F1 re-score (if artifacts genuinely available), E4 pilot paired metrics.
- [ ] P3: fully cited BLIP2/QFormer paper table, code/weight availability and reproduction feasibility statuses.
- [ ] P4: complete **feasible** nontraining pilots one by one, remaining arms marked BLOCKED or SCRIPT_READY_NOT_RUN with concrete reasons.
- [ ] P5/P6: maintainable model/data/eval code; executable **training scripts and run configurations** for all applicable BLIP2/QFormer variants, with safe guards.
- [ ] Reports: milestone `progress.md`, result tables with cohort and metric protocols, comparisons, images/case review if allowed, no unresolved missing-data ambiguity.
- [ ] Handoff: exact next command, required approval when GPU training, status summary <= 15 lines.

## 12. Verified public research links & data context

- BLIP-2: https://arxiv.org/abs/2301.12597 ; https://github.com/salesforce/LAVIS/tree/main/projects/blip2
- ClinicalBLIP: https://doi.org/10.2196/32690
- R2-LLM: https://ojs.aaai.org/index.php/AAAI/article/view/29826 ; https://github.com/synlp/R2-LLM
- META-CXR: https://doi.org/10.1109/ACCESS.2025.3606961 ; https://github.com/DasithEdirisinghe/META-CXR
- RRG-LLM: https://doi.org/10.3233/SHTI250976
- Swin-Qwen3: https://doi.org/10.3389/fradi.2026.1875905
- MIMIC-CXR (image + free-text report; credentialed): https://physionet.org/content/mimic-cxr/2.1.0/
- MIMIC-CXR-JPG (images, labels/splits; report text not presumed present in JPG-only copy): https://physionet.org/content/mimic-cxr-jpg/2.1.0/
- VinDr-CXR (labels/boxes, no standard paired free-text report): https://physionet.org/content/vindr-cxr/1.0.0/ ; https://www.nature.com/articles/s41597-022-01498-w

**Tình trạng hiện tại:** Đây là **plan**, không phải kết quả máy chủ. Không có thí nghiệm GPU, training hoặc lệnh truy cập `/storage/shared/` nào được thực thi từ phiên tạo tài liệu này.
