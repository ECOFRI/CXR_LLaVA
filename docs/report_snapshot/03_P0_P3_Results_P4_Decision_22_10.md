<!-- Snapshot of Google Doc 'VLM Medical' tab 'P0–P3 Results & P4 Decision (22-10)' (t.ugcbxnenhjxv), exported 2026-11-13. Read-only copy; source of truth is the Google Doc. -->

VLM MEDICAL — TIẾN ĐỘ P0–P3 & QUYẾT ĐỊNH P4
Báo cáo chuẩn bị cho Report 2/11 | Cập nhật: 22/10/2026 | Dự án CXR-LLaVA × CLEAR × BLIP-2/Q-Former
Nguồn kết quả: báo cáo vận hành Claude/Codex do nhóm cung cấp. Các số đo server và GPU được giữ theo log của nhóm; phiên cập nhật Google Docs này chưa tự kiểm toán filesystem hoặc tái chạy thí nghiệm.
# I. TỔNG QUAN TRẠNG THÁI
P0: đã kiểm kê hạ tầng và dữ liệu ở mức PARTIAL; phát hiện /home đầy 100%, chuyển cache/temp và output sang /storage, dùng trực tiếp Python của Conda env có sẵn.
P1: COMPLETE; tái chấm Indiana X00 trên CPU, audit MIMIC/VinDr/NIH/CheXpert, và tải checkpoint BLIP-2 có kiểm tra SHA-256.
P2: COMPLETE; so sánh 264 E4 reports, kiểm kê official MIMIC splits, chấm METEOR, BLIP-2 X02 smoke test và audit source R2-LLM/META-CXR.
P3: COMPLETE; kiểm tra paper/code/weights/licensing, tạo scripts X03–X09, dry-run/test và đề xuất phương án training X06_projection. Chưa train các nhánh này.
# II. KẾT QUẢ THỰC NGHIỆM ĐÃ XÁC NHẬN THEO LOG
## 1. Indiana X00 — CXR-LLaVA baseline
3.667/3.667 generated reports được ghép reference và đánh giá. BLEU-1 = 0,2879; BLEU-2 = 0,1468; BLEU-3 = 0,0818; BLEU-4 = 0,0450; ROUGE-L = 0,2079. METEOR được bổ sung sau P1; giá trị tổng hợp chưa được cung cấp trong bản recap này, cần lấy từ artifact P2.
Historical clinical macro-F1 = 0,4457 theo local policy. Nhạy cảm với uncertain/missing: sensitivity analysis trước đó đạt 0,2766 theo chính sách thay thế. Luôn báo label support, coverage và chính sách mask; không so trực tiếp các F1 khác cohort/ontology.
## 2. E4 — CLEAR evidence prompting (exploratory)
264 reports từ 24 studies, 11 arms. So sánh paired theo study; C0 BLEU-4 = 0,0371, METEOR = 0,2111, macro-F1 historical = 0,8063 (6 nhãn). Con số F1 E4 KHÔNG so trực tiếp với X00: cohort 24 studies, mỗi lớp chỉ có khoảng 1–6 ca dương.
Một số CI của delta BLEU/METEOR không chứa 0 ở N1, F1-M, F1-S, F1-O, F2-O; macro-F1 ở F3-M. Tuy nhiên n = 24 và CI chưa hiệu chỉnh đa so sánh: chỉ coi là tín hiệu thăm dò, không khẳng định CLEAR cải thiện chất lượng lâm sàng. Đánh giá mù E4-A2 chưa được ghi nhận hoàn tất.
## 3. MIMIC, VinDr và dữ liệu đánh giá
MIMIC-CXR-JPG: 377.110 ảnh, 227.835 studies, 65.379 patients. Official validation: 2.809/2.991 JPG có sẵn; official test: 4.904/5.159 JPG có sẵn. Cả 5.077 studies thuộc hai split được kiểm kê đều có report; không có patient overlap giữa validation/test.
Quan trọng: MIMIC_CXR_DATASET_TESTSET.csv trong repo KHÔNG phải official test: 2.943/3.000 dòng thuộc official train. Chỉ dùng official split từ dataset; cảnh báo contamination vì CXR-LLaVA/CLEAR từng sử dụng MIMIC trong pretraining/training.
VinDr-CXR: 15.000 ảnh train và 3.000 ảnh test, không giao nhau; chuẩn bị script classification AUROC/AUPRC/F1 nhưng chưa có scores. Không tính BLEU trên bản VinDr tiêu chuẩn nếu không có reference reports.
NIH: /storage/student4/hanhpm/datasets/CXR8 (E1 cohort 4.999 ảnh, test locked). CheXpert: .../datasets/CheXpert_test_labels (668 ảnh / 500 studies, test locked). Không dùng hai locked sets để chọn model hoặc threshold.
## 4. X02 — BLIP-2 OPT-2.7B zero-shot smoke
Checkpoint Salesforce/blip2-opt-2.7b, revision 59a1ef6, 13,96 GiB; hai weight files đúng kích thước và SHA-256. Đã lưu dưới research/blip2_cxr_v4/.cache/models/blip2-opt-2.7b_59a1ef6/.
Inference FP16 trên 1 Tesla K80 chạy được, peak VRAM 7,57 GiB. Prompt dài lần 1 tạo 10/10 output rỗng (FAIL, đã giữ log). Lần 2 prompt Question: … Answer: cho 10/10 output hợp lệ nhưng BLEU-1 = 0,002; BLEU-4 = 0; ROUGE-L = 0,071; METEOR = 0,037.
Clinical F1 = N/A vì 10 references không có ca positive; có 2 pneumothorax dương tính giả trong generated reports. Kết luận: baseline đa lĩnh vực chạy được nhưng không đủ năng lực sinh báo cáo CXR đáng tin cậy; không gọi F1 = 0.
# III. P3 — LITERATURE, CODE VÀ TRAINING READINESS
## 1. Paper & quyền sử dụng
BLIP-2 OPT-2.7B: Original Q-Former; LAVIS/Hugging Face; checkpoint đã có và X02 chạy được.
R2-LLM: Original Q-Former (MiniGPT-4); official code MIT tại commit 2167adf, nhưng không xác minh được pretrained model weights công khai → BLOCKED. Mọi phương án tự train dựa trên Vicuna/LLaMA phải kiểm tra license riêng.
META-CXR: Modified Q-Former; official source commit e97d709, weights chỉ qua link Drive chưa xác minh; không có license chung rõ ở repository root → BLOCKED đến khi có quyền sử dụng.
ClinicalBLIP: InstructBLIP/modified Q-Former; RRG-LLM: Original Q-Former; Swin-Qwen3: Q-Former-style attention. Theo audit P3 chưa xác nhận official code và weights phù hợp → BLOCKED hoặc chỉ tái dựng rõ nhãn.
## 2. Script & môi trường
X06_projection: REIMPLEMENTED_SCRIPT_READY, dry-run PASS; ước tính training peak VRAM 8,6–10,6 GiB (HEURISTIC, chưa đo). X06_qformer: REIMPLEMENTED_SCRIPT_READY, ước 10–13 GiB. X06_opt_lora: REIMPLEMENTED_SCRIPT_READY, ước 10,6–13,6 GiB. X08/X09: BLOCKED do thiếu cache features/encoder adapter. X03/X04/X05/X07: thiếu checkpoint hoặc quyền.
QA: 8/8 tests PASS; --train không có approval gây PermissionError và không tạo output folder. Driver NVIDIA 440.33 giới hạn CUDA 10.2, GPU K80 không có bf16/FlashAttention/bitsandbytes hiện đại. Env backport cho R2-LLM đã viết config nhưng CHƯA cài; giữ HOLD.
# IV. QUYẾT ĐỊNH P4 — PHÊ DUYỆT CÓ ĐIỀU KIỆN
Quyết định: GO P4.1; GO có điều kiện P4.2 và P4.3, chỉ sau khi gate trước PASS. Giữ HOLD X06_qformer, X06_opt_lora, legacy env, X08/X09, weights R2-LLM/META-CXR. Không tải thêm weights, không cấp phép cho những experiments khác.
## P4.1 — VRAM/optimizer feasibility gate — GO
Chạy X06_projection tối đa 20 optimizer steps, 1 card K80, tổng thời gian tối đa 15 phút. Dữ liệu MIMIC official train; freeze vision encoder, Q-Former, OPT, train đúng projection module đã khai báo. Bắt buộc đo GPU peak VRAM, runtime/step, loss finite, forward/backward/optimizer updates, gradient flow và checkpoint save/resume. Preflight phải xác nhận GPU rảnh, Conda env, manifest, CUDA và bộ nhớ disk còn đủ.
PASS để mở P4.2 chỉ khi không OOM/CUDA error, 20 optimizer steps thực sự hoàn tất, peak VRAM ≤ 10,0 GiB trên GPU 11,4 GiB và không có khác biệt cấu hình giữa gate với pilot (ngoài kích thước dữ liệu). Nếu sai bất kỳ điều kiện nào → HOLD, ghi nguyên nhân; không tự chuyển 2 GPU hay đổi batch/sequence length mà không chạy lại gate.
## P4.2 — X06_projection pilot training — GO nếu P4.1 PASS
Train đúng 2.000 unique MIMIC studies có JPG/reference hợp lệ thuộc official train, 1 epoch, 1 card K80, hard timeout 4 giờ. Tái sử dụng checkpoint BLIP-2 local. Save adapter/config/checkpoint và resume; no downloads/new env; chỉ tạo file ở project /storage. Dùng nohup/detached log, lưu PID, git commit, seed, data manifest, số update steps, loss/grad, GPU peaks, CPU/RAM và checkpoint SHA. Không dùng official MIMIC test hoặc Indiana làm train hoặc chọn hyperparameters.
Nếu vượt VRAM/time, loss non-finite, mismatch data mapping, hoặc không thể recover checkpoint: dừng, đánh dấu PARTIAL/BLOCKED; không tự kéo dài ngân sách hoặc bắt đầu training một architecture khác.
## P4.3 — Paired report evaluation — GO nếu P4.2 hoàn tất
Đánh giá X06_projection đã train với BLIP-2 zero-shot trên cùng 500 studies MIMIC official validation và 500 Indiana studies frontal được chọn xác định từ cohort X00. Chỉ dùng ca có ảnh/reference thực sự tồn tại; một study một reference theo policy, fixed prompt/decoding, identical manifests; khởi đầu 10 cases để xác nhận output không rỗng. Giới hạn tổng GPU inference 2 giờ trên 1 K80, trừ khi có phê duyệt mới.
Metrics: BLEU-1/2/3/4, ROUGE-L, METEOR, clinical precision/recall/F1, per-class support/coverage, empty outputs, hallucination/omission review; paired bootstrap theo patient/study. MIMIC validation chỉ là development/internal (contamination risk với các mô hình khác). Indiana là external-to-current-finetuning nhưng chưa thể kết luận không có pretraining overlap nếu chưa audit.
So sánh với X00 chỉ trên 500 Indiana studies trùng ID và cùng policy. Trên MIMIC validation chỉ được so hai nhánh BLIP-2 cùng cohort trừ khi CXR-LLaVA baseline có output đúng trên cùng studies; không lấy F1/BLEU X00 Indiana để so trực tiếp với MIMIC.
## P4 HOLD
HOLD: X06_qformer; X06_opt_lora; cài legacy R2-LLM/META-CXR env; download R2-LLM/Vicuna/META-CXR weights; X08/X09 CLEAR/CXR-LLaVA encoder feature extraction; mở rộng X02 chỉ để có 100 cases. Chỉ xem xét sau P4.1–P4.3 và bằng chứng khoa học/tài nguyên.
# V. ARTIFACTS & BÁO CÁO CẦN DUY TRÌ
P1/P2/P3: reports/P1_progress.md, reports/P2_progress.md, reports/P3_progress.md; reports/03_paper_evidence_matrix.md; reports/04_experiment_tracker; reports/05_results_and_decisions.md; manifests/papers.csv; manifests/mimic_official_coverage.json; runs/P2_e4_paired/; runs/P1_blip2_download/.
P4 yêu cầu cập nhật sau mỗi gate: reports/P4_progress.md, reports/04_experiment_tracker, reports/05_results_and_decisions.md, approvals/X06_projection.yaml, manifests/ train-val cohort, runs/P4_X06_projection/{preflight,metrics,logs,status,checkpoints}. Tên cụ thể phải theo convention repository hiện hữu, không tạo root mới.
Dừng sau P4, gửi GO/HOLD mới với bảng so sánh cùng cohort. Báo kết quả đã đo, không biến feasibility hoặc số ước lượng VRAM thành phát hiện khoa học.
# VI. PAPER LINKS
BLIP-2: https://arxiv.org/abs/2301.12597
ClinicalBLIP: https://formative.jmir.org/2024/1/e32690/
R2-LLM: https://ojs.aaai.org/index.php/AAAI/article/view/29826 | Code: https://github.com/synlp/R2-LLM
META-CXR: https://doi.org/10.1109/ACCESS.2025.3606961 | Code: https://github.com/DasithEdirisinghe/META-CXR
RRG-LLM: https://doi.org/10.3233/SHTI250976
Swin-Qwen3: https://doi.org/10.3389/fradi.2026.1875905
CXR-LLaVA: https://github.com/ECOFRI/CXR_LLAVA | CLEAR: https://doi.org/10.1038/s41551-026-01741-4

# P4 — Projection Adaptation: Results and Clinical Error Analysis
Chốt P4 ngày 25/10/2026 · thực thi PASS; đánh giá đầy đủ về metrics sinh báo cáo, chưa xác nhận lợi ích lâm sàng.
P4.1 đo trên một Tesla K80: 20 optimizer updates trong 71,499 giây, peak reserved 7,953 GiB (allocated 7,568 GiB), gradient/update và adapter reload PASS. P4.2 huấn luyện projection khởi tạo mới từ pretrained, 2.000 unique MIMIC train studies/updates, một epoch trong 4.944,690 giây; checkpoint SHA-256 7f833eea69fb6d2743b19bbc808be949bb1a5e014bf43a81f08090f9a5f0d8af, reload PASS.
P4.3: 500/500 MIMIC official validation và 500/500 Indiana unique studies, 2.000/2.000 M0/M1 generations hợp lệ. ID study/image/reference khớp manifest, không thiếu hoặc trùng; M0/M1 dùng cùng ảnh, QA prompt, preprocessing, greedy decoding, EOS=2, max_new_tokens=128. GPU lũy kế 15.243,812/18.000 giây. Indiana M2 dùng output X00 lưu sẵn trên đúng 500 study/image/reference; protocol mô hình khác.
## Bảng so sánh chính (corpus BLEU-4; historical CheXpert/NegBio micro-F1):
MIMIC 500: M0 0,0000 / 0,2167; M1 0,0346 / 0,4351; Δ BLEU-4 M1−M0 +0,034638, CI95% [0,027476; 0,045860].
Indiana 500: M0 0,0031 / 0,1839; M1 0,0515 / 0,1045; M2/X00 0,0448 / 0,5448; Δ BLEU-4 M1−M0 +0,048423, CI95% [0,043004; 0,054389].
Khoảng tin cậy dùng bootstrap ghép cặp 1.000 lần, seed 42, cụm theo patient ở MIMIC (121 patient), theo study ở Indiana vì thiếu patient ID. Hai CI BLEU-4 đều loại 0; các metric phụ và phân tích lỗi là thăm dò, không điều chỉnh multiple comparisons.
Clinical F1 sensitivity: historical policy chỉ giữ cặp nhãn GT/pred definite; coverage rất thấp và khác nhau: MIMIC M0/M1 107/353 trên 3.500 class-cases; Indiana 79/463. Khi chỉ giữ GT-definite và xem prediction missing/uncertain là âm tính, micro-F1 MIMIC M0/M1 = 0,0340/0,1372; Indiana = 0,0556/0,0488; M2 Indiana = 0,3883. GT không bị thay nhãn. Per-class TP/FP/FN/TN, precision/recall/F1, support, coverage và missing/uncertain rates nằm trong P4.3_final_scientific_analysis.json tại repo; F1 không xác định giữ N/A. Kết luận cải thiện lâm sàng không nhất quán.
Ứng viên lỗi chỉ dựa trên văn bản/labeler: trên MIMIC, proxy unsupported-positive M0→M1 là 94→47 ca, omission 334→322; Indiana 71→69 và 150→149. M1 xuất hiện cue phủ định sai mới ở 34 MIMIC và 8 Indiana; cùng bệnh lý có cue khẳng định lẫn phủ định trong 22/500 báo cáo M1 ở mỗi cohort (M0: 0/500). Đây không phải lỗi lâm sàng được xác nhận; regex có thể hiểu sai phạm vi, thời gian và vị trí. Hai tập chọn tái lập có 267 candidate records gồm disagreement và control, không xuất patient ID hay nguyên văn báo cáo.
Kết luận: projection-only adaptation cải thiện lexical overlap trên cả hai cohort cố định. Chưa chứng minh giảm hallucination/omission thật, clinical F1 hoặc an toàn lâm sàng. M0 chạm trần 128 token trên cả 1.000 output; model có lặp text; nhãn tự động thưa/không đồng nhất; cohort lấy theo source order; MIMIC validation là development set; pretraining overlap chưa loại trừ; không có đọc ảnh hay radiologist adjudication.
P5: GO chỉ cho đề xuất/thiết kế P5.0 trên CPU; HOLD mọi GPU P5 cho tới phê duyệt riêng. Đề xuất so X06_qformer (Q-Former + query tokens + projection trainable, ViT/OPT frozen; 107.130.880 parameters) với P4 projection trên cùng train/eval manifests và protocol. P5.1 đề xuất gate 20 updates, 15 phút, ≤10,5 GiB trên một K80; P5.2 tối đa 4 giờ nếu gate dự báo đủ 2.000 updates; P5.3 tối đa 1.000 generations mới và ngân sách đề xuất 3 giờ. Chưa chạy P5. Cần review thủ công các candidate lỗi trước khi diễn giải lâm sàng.
Nguồn: run P4_projection_pilot_20261023; metrics.json SHA-256 ed6be843294d14deafa1fd9c9366bf302a53c08f12f6804c530e680393fb9be5; final scientific analysis SHA-256 ab30c506b6ccf7d10d8cec3d3a36c4ccd15167e7924b1a51381c33a4ff24799e; config SHA-256 d010d2cf32cb6cc9b96b834e5eb0a5df1d8bd13985ec31cf5faa447260471eb9; manifest index SHA-256 74965a79b31394a65af10dcb7987db0881e5083a8bb0da888bccbe120bd03175.

Hiệu chỉnh đo lường nhãn uncertain: metrics.json gốc đếm thiếu vì so chuỗi “-1” với CSV “-1.0”. Bảng/rates trong phân tích cuối đã tính lại từ labeler output bất biến; confusion và F1 lịch sử tái lập đúng. Giữ nguyên metrics.json gốc và SHA để truy vết.

