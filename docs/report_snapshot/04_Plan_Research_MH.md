<!-- Snapshot of Google Doc 'VLM Medical' tab 'Plan Research (MH)' (t.6ammpquceezr), exported 2026-11-13. Read-only copy; source of truth is the Google Doc. -->

# VLM MEDICAL — RESEARCH & EXPERIMENT MASTER PLAN V3
Ngày cập nhật: 21/10/2026Đề tài: Improving Chest X-ray Report Generation using Vision-Language ModelsMô hình nghiên cứu: CXR-LLaVA, CLEAR, BLIP-2/Q-FormerĐịnh hướng: Dataset Audit → Baseline Benchmark → Report Evaluation → Architecture Improvement

## ĐIỀU CHỈNH CÁCH THỰC THI P5 — ONE-SHOT SCRIPT ORCHESTRATION (25/10/2026)
## Chủ trương mới từ người dùng: không chia P5 thành nhiều vòng giao prompt/chờ quyết định thủ công; cho Codex tự nghiên cứu lỗi, sửa mã P5-specific, bổ sung CPU tests rồi thiết kế runner tách khỏi phiên agent để thực hiện tuần tự gate P5.1 → P5.2 → P5.3 → phân tích CPU. Mỗi gate bắt buộc PASS mới chuyển bước. Một script orchestration, nhật ký, PID, heartbeat, status/lock, checkpoint/resume idempotent; không chạy GPU trực tiếp trong phiên agent kéo dài.
## Phạm vi tài nguyên không tăng so với các mức đã dùng/được phép ở P4: một K80; P5.1 tối đa 20 optimizer updates, 900 giây, peak reserved không vượt 10,0 GiB (chọn giới hạn chặt hơn mức P5.0 đề xuất 10,5 GiB); P5.2 tối đa 2.000 updates/1 epoch, 4 giờ; P5.3 tối đa 1.000 generations mới, 3 giờ (thấp hơn trần inference P4.3 đã gia hạn 5 giờ). Các hạn mức này là trần riêng cho P5, không được cộng dồn ngân sách còn thừa của P4, không dùng 2 GPU, không tự kéo dài thời gian hay đổi cấu hình khoa học để vượt gate.
## Điều kiện tự tiến hành: preflight GPU rảnh và phù hợp, đúng manifest/hashes/seed, parity loss/target so P4, frozen modules ở eval mode, gradient 3 nhóm trainable, resume khôi phục data cursor/RNG/optimizer; CPU tests PASS; launcher có hard timeout/VRAM monitor, lock chống chạy trùng, checkpoint atomic và exit status đáng tin. Lỗi kỹ thuật sửa được trong ràng buộc thì tự sửa và test lại; OOM, không đủ giờ/VRAM, parity fail, dữ liệu lệch, lỗi an toàn hoặc cần ngân sách mới thì STOP/HOLD và báo cáo, không tự phê duyệt.
## Ngưỡng P5.2 phải đánh giá khả năng hoàn tất toàn bộ 2.000 updates trong 4 giờ dựa trên tốc độ P5.1 đo được (mục tiêu toàn bộ khoảng ≤7,2 giây/update kể cả overhead), không suy luận chỉ từ P5.1 PASS về VRAM. P5.3 sử dụng lại M0/M1 đã lưu, chỉ sinh Q-Former output trên cùng 500 MIMIC validation + 500 Indiana; clinical F1 và lỗi văn bản chỉ là proxy khi chưa có radiologist adjudication.
## Trạng thái tại thời điểm cập nhật kế hoạch: P5.0 PASS CPU; P5.1/2/3 CHƯA CHẠY. Đây là điều chỉnh phạm vi công việc và cơ chế gate, không phải báo cáo P5 execution PASS. Handoff cuối chỉ khi script chạy xong hoặc STOP/BLOCKED, phân biệt completed/partial/pending.
## CẬP NHẬT ƯU TIÊN — P4 CLOSEOUT / P5.0 CPU DESIGN (25/10/2026)
Trạng thái hiệu lực: phần cập nhật 25/10 này được ưu tiên khi các bảng trạng thái hoặc roadmap V3 ngày 21/10 bên dưới còn ghi P5/BLIP-2 là Proposed/HOLD. Giữ nguyên các snapshot lịch sử để truy vết; đây không phải phê duyệt huấn luyện.
P4.1 PASS: 20 optimizer updates / 71,499 giây / peak reserved 7,953 GiB. P4.2 PASS: 2.000 unique MIMIC train studies/updates, 1 epoch / 4.944,690 giây; checkpoint save/reload PASS.
P4.3 PASS về thực thi: 500 MIMIC official validation + 500 Indiana unique studies, đủ 2.000 output M0/M1 hợp lệ, không thiếu/trùng, match study/image/reference; GPU inference lũy kế 15.243,812/18.000 giây. Không cần chạy lại mô hình.
Primary lexical result (M1−M0, paired corpus BLEU-4): MIMIC 0,0000→0,0346, Δ +0,0346, CI 95% [0,0275; 0,0459]; Indiana 0,0031→0,0515, Δ +0,0484, CI [0,0430; 0,0544]. Bootstrap 1.000 lần; MIMIC grouped by patient, Indiana by study. Chỉ xác nhận lexical overlap trên cohort cố định.
Clinical F1 không cải thiện nhất quán: historical micro-F1 MIMIC 0,2167→0,4351; Indiana 0,1839→0,1045. Với GT-definite và missing/uncertain prediction = negative, MIMIC 0,0340→0,1372; Indiana 0,0556→0,0488. Coverage khác nhau mạnh; không suy ra tăng độ chính xác hay an toàn lâm sàng.
Rủi ro khoa học: M0 bị giới hạn 128 token ở 1.000/1.000 output; report repetition, source-order selection, pretraining contamination chưa loại trừ. Hallucination/omission/negation chỉ là ứng viên text/labeler, chưa đối chiếu ảnh bởi bác sĩ. metrics.json gốc và SHA được giữ nguyên; count uncertain được hiệu chỉnh riêng từ CSV -1.0.
QUYẾT ĐỊNH: GO duy nhất P5.0 thiết kế/audit trên CPU; HOLD tuyệt đối mọi GPU P5.1–P5.3, Q-Former training, OPT LoRA, CLEAR integration, feature extraction, inference và tải weights cho đến khi có phê duyệt cấu hình + ngân sách riêng.
P5.0 task 1 — Reuse artifacts/cấu hình P4 bất biến; kiểm tra script X06_qformer và phần trainable: Q-Former + query tokens + projection, ViT/OPT frozen. 107.130.880 parameters hiện là số đề xuất cần đối chiếu bằng code/CPU trước khi đưa vào approval.
P5.0 task 2 — Soạn matched experiment specification: cùng 2.000 MIMIC train, 500 MIMIC validation + 500 Indiana eval, prompt/preprocessing/decoding/evaluator và study/image/reference manifests như P4. Giữ P4 projection M1 làm primary matched baseline, M0 pretrained và M2/X00 chỉ là control/historical comparator phù hợp.
P5.0 task 3 — Dự toán VRAM, runtime, storage, gradient flow/checkpoint, rủi ro OOM trên một K80; xác định acceptance/stop criteria, giới hạn uncertainty. P5.1 đề xuất (CHƯA DUYỆT): 20 updates, ≤15 phút, reserved ≤10,5 GiB. P5.2 đề xuất: 2.000 updates/≤4 giờ sau khi gate PASS. P5.3 đề xuất: tối đa 1.000 generations mới, ≤3 giờ và reuse P4 outputs.
P5.0 task 4 — Chốt evaluation & review plan trước training: paired BLEU-4 primary, BLEU-1/2/3, ROUGE-L, METEOR; per-class TP/FP/FN/TN, support/coverage/missing/uncertain; 2 policies F1, paired bootstrap; error candidate review thủ công và phân loại chưa clinical adjudication. Không sử dụng locked tests để chọn architecture/threshold.
P5.0 deliverables: versioned architecture_spec.md, matched_protocol.md, cpu_static_validation.md, resource_budget.md, P5_approval_request.md và tracker/decision update trong project hiện hữu. Cuối bước chỉ báo cáo GO/HOLD/BLOCKED và yêu cầu duyệt GPU riêng; không tự launch job.

## 1. Mục tiêu nghiên cứu
### 1.1. Mục tiêu chính
Nghiên cứu khả năng cải thiện quá trình tự động sinh báo cáo X-quang ngực bằng cách kết hợp biểu diễn hình ảnh và thông tin bệnh lý từ các mô hình Vision-Language.
Hệ thống nghiên cứu gồm:
CXR-LLaVA: Mô hình baseline thực hiện chest X-ray report generation.
CLEAR: Mô hình cung cấp visual representations và radiological concept evidence.
BLIP-2: Kiến trúc tham khảo cơ chế Q-Former để kết nối vision encoder và LLM hiệu quả hơn.
### 1.2. Research questions
RQ1 — Baseline quality: CXR-LLaVA hiện tạo báo cáo gần với ground truth đến mức nào, xét cả BLEU và chất lượng phát hiện bệnh?
RQ2 — Visual representation: CLEAR có biểu diễn hình ảnh tốt hơn vision encoder của CXR-LLaVA trong cùng benchmark và protocol không?
RQ3 — Concept grounding: CLEAR finding/concept evidence có cải thiện chất lượng nội dung báo cáo được tạo ra không?
RQ4 — Architecture: Thay linear projector bằng Q-Former, hoặc dùng CLEAR làm vision encoder, có đem lại cải thiện so với baseline đơn giản không?
RQ5 — Generalization: Các cải thiện có duy trì trên dữ liệu độc lập, ngoài tập development và không bị ảnh hưởng bởi pretraining contamination không?

## 2. Trạng thái nghiên cứu hiện tại
| Experiment | Trạng thái | Kết quả |
|---|---|---|
| E0 — CXR-LLaVA reproduction | Completed | Indiana 3.667/3.667 |
| Indiana baseline | Completed | Macro F1 = 0,4457 theo protocol local |
| E1 — Encoder comparison | Completed | CLEAR AUROC = 0,7671; CXR-LLaVA = 0,6312 trên NIH exploratory |
| E1.3 — Similarity analysis | Completed | Representation comparison và error complementarity |
| E2 — CLEAR zero-shot | Completed, direct pathway | CheXpert AUROC = 0,893; F1 = 0,520 |
| E5 — Global feature fusion | Completed | DEV Macro AP fusion = 0,2515; CLEAR-only = 0,2566 |
| E4 — Finding evidence pilot | Exploratory | 264 reports từ 24 Indiana images |
| E4-A2 — Blind review | Pending | 80 reports chuẩn bị, 0/400 lượt chấm |
| BLIP-2/Q-Former investigation | Proposed | Chưa có benchmark hoặc checkpoint được kiểm chứng |
| Full dual-encoder training | Not started | Chưa phê duyệt |

Kết luận hiện tại: Baseline và representation evaluation đã có kết quả. Chưa có bằng chứng rằng fusion cải thiện report generation. Ưu tiên mới là hoàn thành benchmark report generation trước khi thay đổi kiến trúc.

## 3. G0 — Dataset Inventory & Provenance Audit
Ưu tiên: P0 — Bắt buộc thực hiện đầu tiên.
### 3.1. Kiểm tra /storage/shared/
Kiểm tra toàn bộ các thư mục dữ liệu được chia sẻ trên server, ưu tiên:
MIMIC-CXR / MIMIC-CXR-JPG (tên MINIC cần xác minh).
VinDr-CXR / VinDXR / VinDR.
Indiana University Chest X-ray.
CheXpert.
NIH ChestX-ray14.
Các dataset CXR khác có sẵn.
Không tải lại dataset trước khi xác nhận tài nguyên hiện hữu.
### 3.2. Nội dung kiểm kê
Với mỗi dataset cần thống kê:
Dataset name, version, source và đường dẫn thực tế.
Số lượng ảnh, số lượng studies và số lượng patients.
Kiểu ảnh: DICOM, JPG, PNG.
Frontal/lateral views, resolution và preprocessing.
Ground-truth pathology labels.
Ground-truth radiology reports.
Patient ID, study ID và image ID.
Train/validation/test splits.
Tính đầy đủ và khả năng đọc của từng thành phần.
Nguy cơ trùng dữ liệu với quá trình huấn luyện của CLEAR và CXR-LLaVA.
### 3.3. Ma trận dataset dự kiến
| Dataset | F1 | AUROC | BLEU | Ghi chú |
|---|---|---|---|---|
| Indiana | Có | Có điều kiện | Có | Có image–report pairs; cần audit study linkage |
| MIMIC-CXR | Có | Có điều kiện | Có | Ưu tiên report benchmark, cần kiểm tra split |
| VinDr-CXR | Có | Có | Không mặc định | Có image-level labels và bounding boxes; không có free-text report chuẩn tương ứng |
| CheXpert test | Có | Có | Không | Không có ground-truth report cho phép đo này |
| NIH | Có | Có | Không | Classification benchmark; có nguy cơ pretraining overlap |

Quy tắc: BLEU chỉ được tính trên dữ liệu có ground-truth reports thực sự ghép được với output. AUROC chỉ được tính khi có continuous scores và ground-truth nhị phân hợp lệ.
### 3.4. Data leakage audit
Đối chiếu dataset với paper gốc, checkpoint metadata và danh sách ảnh đã sử dụng trong training.
Đặc biệt:
VinDr đã được dùng trong training vision encoder của CXR-LLaVA theo Table 1 của paper.
MIMIC-CXR đã được dùng trong quá trình phát triển CXR-LLaVA và pretraining CLEAR.
NIH cũng đã được sử dụng trong training CXR-LLaVA.
Do đó, các dataset này không được mặc định là external independent test của toàn bộ hệ thống.
Phải xác định chính xác mức độ overlap bằng identifiers hoặc bằng chứng provenance phù hợp. Nếu chưa xác minh được, gắn nhãn exploratory hoặc potentially contaminated trong báo cáo.
### 3.5. Deliverables
dataset_inventory.csv
dataset_provenance.md
dataset_eligibility_matrix.csv
dataset_overlap_audit.md
benchmark_cohort_manifest.csv
Gate G0: Chỉ tiếp tục benchmark chính thức trên những cohort có dữ liệu cần thiết, mapping chính xác và tình trạng provenance được ghi rõ.

## 4. G1 — Historical Report Generation Benchmark
Ưu tiên: P0 — Tận dụng output hiện có.
### 4.1. Đánh giá Indiana baseline
Sử dụng 3.667 generated reports đã lưu từ CXR-LLaVA.
Không chạy lại model nếu các output và ground-truth reports còn đầy đủ.
Thực hiện:
Lấy generated report và reference report.
Ghép chính xác theo study/image mapping.
Kiểm tra missing, duplicate và empty reports.
Chuẩn hóa văn bản theo cùng một protocol.
Đo BLEU-1, BLEU-2, BLEU-3, BLEU-4.
Bổ sung ROUGE-L và METEOR nếu công cụ có sẵn.
Trích xuất clinical findings từ cả hai loại report.
Đánh giá per-class F1, macro F1 và micro F1.
Phân tích false positives, false negatives và omissions.
Lưu kết quả để so sánh với các architecture cải tiến.
### 4.2. BLEU evaluation protocol
Primary report metric: Corpus BLEU-4.Secondary report metrics: BLEU-1/2/3, ROUGE-L, METEOR.
Cần cố định:
Phiên bản công cụ và tokenizer.
Cách xử lý viết hoa, dấu câu và khoảng trắng.
Cách xử lý placeholder ẩn danh.
Cách tách Findings và Impression.
Giới hạn độ dài báo cáo.
Quy tắc chọn reference khi có nhiều report.
Báo cáo riêng:
Full-report BLEU nếu có full-report ground truth phù hợp.
Findings-only BLEU nếu trích xuất được phần Findings đáng tin cậy.
Impression-only BLEU nếu được thiết lập thành benchmark riêng.
Không so sánh BLEU lấy từ các phần báo cáo khác nhau.
Không tự thay đổi reference để làm tăng BLEU. Lưu dữ liệu gốc và bản normalized riêng.
### 4.3. Clinical F1 evaluation
Sử dụng cùng một label ontology và cùng một labeler cho generated/reference reports.
Báo cáo:
Per-class precision/recall/F1.
Macro-F1 và micro-F1.
TP, FP, FN, TN.
Support của từng nhãn.
Tỷ lệ missing và uncertain.
Sensitivity analysis theo các chính sách xử lý missing.
Lưu ý: Không loại toàn bộ các trường hợp không được model nhắc đến rồi chỉ công bố F1 trên những ca còn lại mà không báo coverage.
### 4.4. AUROC evaluation
Đối với report-generation output, không tự động suy ra AUROC từ BLEU hoặc nhãn nhị phân của labeler.
Nếu muốn đánh giá AUROC cho CXR-LLaVA, phải xây dựng nhánh pathology scoring riêng có continuous scores, kiểm tra tính hợp lệ và cố định protocol.
CLEAR có thể được đánh giá bằng continuous direct-contrastive scores, nhưng cần ghi rõ đây là classification/pathology-scoring task, không phải report-generation metric.
### 4.5. Deliverables
indiana_bleu_baseline.csv
indiana_report_metrics.json
indiana_clinical_f1.csv
indiana_error_analysis.md
indiana_benchmark_summary.md
Gate G1: Có bảng BLEU và F1 tái lập được cho Indiana baseline, cùng các file output/reference và protocol cố định.

## 5. G2 — Expanded Benchmark trên MIMIC và VinDr
Ưu tiên: P0 sau G0/G1.
### 5.1. MIMIC-CXR: Report generation benchmark
Mục tiêu là đánh giá khả năng tạo báo cáo trên tập có ground-truth radiology reports.
Pipeline:
Image → CXR-LLaVA → Generated Report → Report Metrics
Ground Truth Report → Reference Text / Clinical Labels
Metrics:
BLEU-1/2/3/4.
ROUGE-L, METEOR.
CheXbert hoặc CheXpert-labeler-based F1.
RadGraph F1 nếu có đủ tài nguyên.
Hallucination, omissions và clinical errors.
AUROC chỉ với một nhánh scoring có continuous outputs hợp lệ.
Thứ tự thực hiện:
Smoke test: tối đa 100 studies hợp lệ.
Pilot: tối đa 500 studies hợp lệ.
Full evaluation: toàn bộ cohort được phê duyệt.
Đối với mỗi study, kiểm tra có bao nhiêu ảnh được sử dụng để viết reference report. Nếu chỉ đưa một ảnh frontal vào model nhưng reference dựa trên nhiều views, cần ghi nhận hạn chế này và định nghĩa một single-frontal benchmark riêng.
Các tập train/validation/test phải được bảo toàn theo protocol đã chọn.
Điều kiện: Không coi MIMIC test là hoàn toàn độc lập với CLEAR trước khi kiểm tra membership/provenance.
### 5.2. VinDr-CXR: Pathology recognition benchmark
Mục tiêu là so sánh khả năng phát hiện bất thường của visual encoders và classification heads.
Models:
CLEAR direct-contrastive.
CLEAR frozen-feature probe (nếu có train cohort hợp lệ).
CXR-LLaVA visual encoder + probe.
Fusion baseline đã có hoặc được tái đánh giá trên cohort phù hợp.
Metrics:
Per-class AUROC.
Macro AUROC.
Per-class AUPRC.
Macro/micro F1.
Sensitivity, specificity.
95% confidence intervals.
Grounding/localization metrics khi có thí nghiệm sử dụng bounding boxes.
Cần ánh xạ ontology giữa VinDr và CheXpert/CLEAR trước khi tính macro averages.
Không tính BLEU trên VinDr tiêu chuẩn nếu không xác minh được ground-truth free-text reports thực sự có trong phiên bản local.
### 5.3. Bảng benchmark tổng hợp
Kết quả cuối phải tổ chức theo dataset, model, task và cùng protocol.
| Dataset | Model | Task | AUROC | F1 | BLEU-4 |
|---|---|---|---|---|---|
| Indiana | CXR-LLaVA | Report generation | N/A* | Existing baseline | To measure |
| MIMIC-CXR | CXR-LLaVA | Report generation | N/A* | To evaluate | To measure |
| VinDr-CXR | CLEAR | Pathology scoring | To evaluate | To evaluate | N/A |
| VinDr-CXR | CXR-LLaVA encoder | Pathology classification | To evaluate | To evaluate | N/A |
| CheXpert | CLEAR | Classification | Existing result | Existing result | N/A |

*AUROC cần nhánh scoring độc lập với continuous outputs nếu muốn báo cáo.
Không so sánh các con số giữa hai dataset như thể đó là cùng một experiment.
Gate G2: Có ít nhất một benchmark report generation hoàn chỉnh và một benchmark pathology recognition với protocol, dataset eligibility và hạn chế được công bố rõ.

## 6. G3 — Literature Review & Benchmark Reproduction
Ưu tiên: P0/P1.
### 6.1. Paper cần nghiên cứu
Nhóm A — Existing models
CXR-LLaVA.
CLEAR.
BLIP-2.
Nhóm B — BLIP-based và report generation
ClinicalBLIP / InstructBLIP.
CheXagent / CheXbench.
R2GenGPT.
MAIRA-2.
Nhóm C — Evaluation
RadGraph và RadCliQ.
CheXbert / CheXpert Labeler.
Các nghiên cứu đối chiếu BLEU với chất lượng lâm sàng.
### 6.2. Paper review template
Với mỗi paper, bắt buộc lập bảng:
| Trường | Nội dung bắt buộc |
|---|---|
| Research question | Vấn đề paper giải quyết |
| Architecture | Encoder, projector/Q-Former, LLM, fusion |
| Training dataset | Dataset, số lượng, provenance |
| Evaluation dataset | Test sets, cohort size, split |
| Task | Classification / VQA / report generation |
| Report metrics | BLEU-1/2/3/4, ROUGE, METEOR... |
| Clinical metrics | F1, AUROC, RadGraph, CheXbert... |
| Main results | Bảng điểm từ paper |
| Limitations | Weaknesses và evaluation gaps |
| Code/checkpoint | Mức độ công khai và khả năng tái lập |
| Relevance | Thành phần phù hợp với VLM Medical |

### 6.3. Benchmark reference cần lưu ý
ClinicalBLIP (InstructBLIP-based):
| Dataset | BLEU-4 | BLEU-A | ROUGE-L |
|---|---|---|---|
| IU X-Ray | 0,254 | 0,296 | 0,534 |
| MIMIC-CXR | 0,115 | 0,162 | 0,313 |

Đây là số liệu tác giả công bố. BLEU-A được định nghĩa là trung bình BLEU-2/3/4, không phải BLEU-4.
Không dùng các số trên làm mục tiêu so sánh trực tiếp nếu chưa xác nhận dataset split, evaluation script, preprocessing và input protocol tương đương.
CXR-LLaVA: Tiếp tục dùng F1 paper gốc làm reference, nhưng bổ sung benchmark BLEU local từ outputs đã sinh.
CLEAR: Đánh giá classification và evidence retrieval. Không áp dụng trực tiếp BLEU cho CLEAR khi chưa có report generator.
BLIP-2: Paper gốc là phương pháp vision-language pretraining tổng quát. Cần ưu tiên nghiên cứu khả năng chuyển Q-Former sang CXR thay vì diễn giải kết quả trên ảnh tự nhiên thành hiệu năng báo cáo y khoa.
### 6.4. Deliverables
literature_review_matrix.csv
benchmark_comparison.md
blip2_qformer_architecture.md
evaluation_gap_analysis.md
paper_reproduction_feasibility.md
Gate G3: Mỗi paper được lựa chọn phải có một benchmark cụ thể có thể đối chiếu hoặc giải thích rõ vì sao không thể tái lập công bằng.

## 7. G4 — BLIP-2 / CLEAR Architecture Investigation
Ưu tiên: P1. Không tự động bắt đầu training.
### 7.1. Kiến trúc hiện tại
Chest X-ray → CXR-LLaVA ViT-L/16 → Linear Projector → LLaMA-2 → Generated Report
### 7.2. Các giả thuyết mới
H1 — CLEAR encoder replacement
Chest X-ray → CLEAR Encoder → Trainable Projector → LLaMA-2 → Generated Report
H2 — Q-Former adaptation
Chest X-ray → Frozen Vision Encoder → Q-Former → Projector → LLaMA-2 → Generated Report
H3 — CLEAR concept evidence
Chest X-ray → CLEAR Findings/Concepts + CXR-LLaVA → Generated Report
H4 — Hybrid representation and concept integration
Chest X-ray → CLEAR Visual Features + Structured Concepts → Learnable Fusion Module → LLaMA-2 → Generated Report
Các nhánh H2–H4 chỉ được thực hiện khi có data readiness và compute feasibility đầy đủ.
### 7.3. Ablation matrix
| Arm | Visual encoder | Bridge/Fusion | Training |
|---|---|---|---|
| A0 | CXR-LLaVA | Linear projector gốc | Frozen baseline |
| A1 | CLEAR | Linear projector | Train projector |
| A2 | CXR-LLaVA | Q-Former | Train bridge |
| A3 | CLEAR | Q-Former | Train bridge |
| A4 | CXR-LLaVA | CLEAR finding evidence | Inference-only |
| A5 | CLEAR + CXR-LLaVA | Advanced fusion | Chỉ xem xét sau các gate |

Chỉ so sánh các arm trên cùng test cohort, cùng policy giải mã và cùng metric implementation.
Nếu thay encoder thì phải kiểm tra embedding dimensions, preprocessing, projection alignment và tham số cần huấn luyện.
Spatial fusion phải kiểm tra geometric alignment; không được mặc định hai patch grids cùng chỉ số chứa cùng vùng giải phẫu.
### 7.4. Compute readiness
Ước lượng trước:
GPU model, VRAM khả dụng.
Batch size và sequence length.
RAM và storage.
Runtime/inference throughput.
Trainable parameter count.
Checkpoint memory.
Khả năng sử dụng frozen features.
Khả năng gradient accumulation.
Không huấn luyện hoặc fine-tune LLM trước khi có baseline và phương án adapter được phê duyệt.

## 8. G5 — Standardized Evaluation & Statistical Analysis
### 8.1. Bảng metric bắt buộc
| Metric | Mục đích | Trạng thái |
|---|---|---|
| BLEU-1 đến BLEU-4 | Lexical report similarity | Mandatory |
| ROUGE-L | Report sequence overlap | Recommended |
| METEOR | Text generation evaluation | Recommended |
| CheXpert-labeler F1 | Clinical label agreement | Mandatory |
| AUROC | Pathology discrimination | Mandatory khi có scores |
| AUPRC | Đánh giá bệnh lý mất cân bằng | Recommended |
| RadGraph F1 | Clinical entities và relations | Recommended |
| Hallucination / omission | Clinical error analysis | Mandatory |
| Clinical blind review | Kiểm tra chất lượng y khoa | Mandatory khi đủ reviewer |

### 8.2. Statistical protocol
So sánh paired predictions trên cùng cohort.
Confidence intervals bằng patient-level bootstrap khi có identifiers.
Chỉ chọn threshold trên train/validation.
Không chọn model dựa trên locked test.
Báo cáo số patient/study/image và positives từng class.
Báo cáo điểm tuyệt đối và delta so với baseline.
Không khẳng định cải thiện nếu kết quả không ổn định hoặc giới hạn thống kê còn lớn.
### 8.3. Evaluation report
Mỗi thí nghiệm phải có:
Experiment ID, date, git commit, model revision.
Dataset version, manifest và split.
Preprocessing và prompt/decoding.
Evaluation script/version.
Results và confidence intervals.
Failure cases.
Data leakage statement.
GO/HOLD decision.

## 9. Execution Roadmap
| Giai đoạn | Công việc | Priority | Output |
|---|---|---|---|
| Week 1 | Kiểm kê /storage/shared/, audit MIMIC/VinDr | P0 | Dataset inventory |
| Week 1 | Ghép 3.667 Indiana outputs với reference | P0 | Matched report pairs |
| Week 1–2 | BLEU/F1 evaluation trên Indiana | P0 | Complete baseline |
| Week 2 | Hoàn thiện E4-A2 blind review | P0 | Paired clinical evaluation |
| Week 2 | Literature review và benchmark matrix | P0 | Paper survey |
| Week 2–3 | MIMIC pilot 100→500 studies | P1 | Report benchmark |
| Week 3 | VinDr pathology benchmark | P1 | F1/AUROC/AUPRC |
| Week 3–4 | MIMIC expanded evaluation | P1 | Full cohort results nếu feasible |
| Week 4 | Q-Former/adapter design và compute estimation | P1 | Architecture proposal |
| After review | Conditional adapter/Q-Former training | P2 | New checkpoint and ablations |

Các thời lượng là kế hoạch đề xuất, phụ thuộc vào dữ liệu thực tế và khả năng cấp phát GPU.

## 10. Go / Hold Criteria
### GO — Tiếp tục thử architecture mới khi
Đã có benchmark BLEU/F1 tái lập được từ baseline.
Dataset manifests và splits rõ ràng.
Có dữ liệu training/development phù hợp.
Đã kiểm tra nguy cơ pretraining contamination.
Đã xác định ưu/nhược điểm của từng architecture dựa trên kết quả.
Có đủ compute budget và ablation controls.
Có phương pháp đánh giá chất lượng báo cáo beyond BLEU.
### HOLD — Chưa huấn luyện nếu
Chưa xác nhận datasets trên server.
Không có ground-truth report phù hợp để đo BLEU.
Chưa có baseline nhất quán.
Không kiểm soát được split leakage.
Chưa phân biệt được classification và report-generation metrics.
Fusion chưa vượt được single-encoder baseline đáng tin cậy.
Chưa có bằng chứng để biện minh chi phí của architecture mới.

## 11. Deliverables tổng kết
Bắt buộc trước khi báo cáo vòng tiếp theo:
Hoàn tất inventory /storage/shared/.
Xác minh MIMIC và VinDr có những thành phần dữ liệu nào.
Hoàn thành dataset/pretraining overlap audit.
Đo BLEU-1/2/3/4 cho Indiana reports đã sinh.
Tổng hợp BLEU và clinical F1 vào cùng báo cáo.
Xây dựng metric protocol có thể tái sử dụng.
Hoàn thiện blinded evaluation E4-A2.
Lập bảng benchmark của ít nhất 4–6 paper liên quan.
Chuẩn bị MIMIC report-generation pilot.
Chuẩn bị VinDr pathology classification benchmark.
Đánh giá tính khả thi của CLEAR + Q-Former.
Ra quyết định GO/HOLD về huấn luyện architecture mới.

## 12. Research Direction
Ưu tiên ngắn hạn: Hoàn thiện bằng chứng đánh giá report generation, đặc biệt BLEU và clinical F1 trên Indiana và MIMIC-CXR.
Ưu tiên trung hạn: Kiểm tra khả năng tổng quát hóa của visual encoders trên VinDr và các tập có nhãn đủ điều kiện, đồng thời đánh giá BLIP-2/Q-Former bằng ablation công bằng.
Định hướng dài hạn: Phát triển một chest X-ray report generator sử dụng thông tin bệnh lý được grounding từ CLEAR để giảm hallucination và omission, đồng thời duy trì hoặc cải thiện các chỉ số sinh văn bản.
Nguyên tắc kết luận: Một architecture chỉ được xem là cải thiện khi các chỉ số phù hợp với cùng tác vụ, cùng cohort và cùng protocol cho thấy lợi ích có thể tái lập, không chỉ khi điểm BLEU hoặc F1 đơn lẻ tăng.
## Tài liệu tham khảo chính
BLIP-2: https://arxiv.org/abs/2301.12597
CXR-LLaVA: https://github.com/ECOFRI/CXR_LLAVA
CLEAR: https://doi.org/10.1038/s41551-026-01741-4
ClinicalBLIP: https://formative.jmir.org/2024/1/e32690
CheXagent: https://github.com/Stanford-AIMI/CheXagent
MIMIC-CXR: https://physionet.org/content/mimic-cxr/2.1.0/
MIMIC-CXR-JPG: https://physionet.org/content/mimic-cxr-jpg/2.1.0/
VinDr-CXR: https://physionet.org/content/vindr-cxr/1.0.0/
R2GenGPT: https://github.com/wang-zhanyu/r2gengpt
Radiology Report Generation Evaluation: https://pmc.ncbi.nlm.nih.gov/articles/PMC10499844/

