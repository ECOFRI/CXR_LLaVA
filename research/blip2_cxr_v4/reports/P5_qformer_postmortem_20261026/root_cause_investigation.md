# P5 Q-Former: điều tra nguyên nhân suy sụp đầu ra (26-10-2026)

**Phạm vi:** phân tích artifact và mã nguồn trên CPU; không huấn luyện, sinh báo cáo mới, tải mô hình lên GPU hay sửa P4/P5. Đầu ra P5 đã hoàn thành về vận hành nhưng **thất bại khoa học** so với P4 projection. Đường dẫn trong tài liệu này tương đối với gốc repository.

## 1. Bằng chứng và tính toàn vẹn

- Kiểm tra tái lập bằng `research/blip2_cxr_v4/reports/P5_qformer_postmortem_20261026/postmortem_diagnostics.py`; kết quả chi tiết ở `diagnostics.json`. Script chỉ đọc artifact lịch sử, tải tokenizer/config và các tensor adapter trên CPU, không chạy forward model. Kiểm tra độc lập lần hai tính lại corpus BLEU-1…4/ROUGE-L từ cặp P4 và P5; tất cả giá trị khớp `metrics.json` chính xác đến sai số <1e-12.
- Manifest được ghim trong `research/blip2_cxr_v4/runs/P4_projection_pilot_20261023/manifest_index.approved.json`: MIMIC train 2.000 study/image duy nhất (`ec8bc5fd…d4e2`), MIMIC validate 500 (`b0d8f209…21a3`), Indiana 500 (`5878e73e…9944`); SHA-256 hiện tại bằng index đã lưu. Không có subject MIMIC trùng giữa **hai tập được chọn**. Không thể từ phép kiểm này kết luận toàn bộ dữ liệu tiền huấn luyện không nhiễm validation.
- `P4.3_pairs.jsonl` có 1.000 bản ghi M0/M1; `P5.3_predictions.jsonl` có 1.000 bản ghi M3. Cả 1.000 stable key, study ID và image ID khớp theo thứ tự; mỗi cohort 500 study/image duy nhất, không thiếu hoặc lặp. Đường dẫn và SHA-256 đầy đủ xem `diagnostics.json > source_artifacts, manifests`.
- P5 train khởi tạo từ BLIP-2 gốc, không dùng adapter P5.1: `scripts/p5/p5_train.py:97-122`; lần tiếp tục P5.2 sau gián đoạn được ghi trong `runs/P5_qformer_20261025/P5.2_recovery_20261025.json`. Checkpoint cuối `P5.2_last.pt` SHA-256 `d207ffb290040b25ed16fd440ccff413fc03334a38d8f0a9d033e03c7517809b`, stage P5.2, epoch 1, cursor/optimizer updates 2.000, config/manifest hash đúng và 257 optimizer-state entries. `P5.3_status.json` cùng SHA; `scripts/p5/p5_infer.py:89-105` nạp BLIP-2 gốc, chép adapter và chuyển eval. Không có bằng chứng artifact bị thay nhầm; lần nạp và forward độc lập không được chạy vì yêu cầu CPU-only.
- Thời gian và VRAM thực: P5.1 20 bước/193,156 giây/9,939 GiB; P5.2 2.000 bước/9.849,692 giây/9,943 GiB; P5.3 1.000 mẫu/9.546,181 giây/7,695 GiB reserved. Nguồn: `runs/P5_qformer_20261025/P5.{1,2,3}_status.json`. PASS ở đây chỉ xác nhận vận hành.

## 2. Đối chiếu giao thức P4/P5

| Component | P4 setting | P5 setting | Evidence source | Match status | Potential impact |
|---|---|---|---|---|---|
| Cohort/reference | Train MIMIC 2.000; eval MIMIC 500 + Indiana 500 | Cùng index/manifests và 1.000 stable key/reference qua P4 evaluator | `diagnostics.json > manifests`; `scripts/p5/p5_evaluate.py:147-159` | **MATCH** | Lệch pairing khó giải thích sụt điểm |
| BLIP-2 gốc | `Salesforce/blip2-opt-2.7b`, local revision 59a1ef6 | Cùng `model.checkpoint_dir` | Hai `configs/experiments/P4_projection_contract.approved.json`, `P5_qformer_contract.approved_20261025.json`; `scripts/p5/p5_train.py:97-105` | **MATCH** theo config/code | Không kiểm tra lại full weight hash trong lần điều tra này |
| Tokenizer/BOS/EOS/PAD | OPT GPT2; 2/2/1 | Cùng; tokenizer local đọc được 2/2/1 | Hai config; `diagnostics.json > tokenizer`; `p5_train.py:98-101` | **MATCH** | Lỗi token đặc biệt không được nguồn hiện có ủng hộ |
| Prompt/target/loss | QA prompt X02; FINDINGS + IMPRESSION; BOS+prompt masked, target+EOS supervised | Dùng chung `target_for`/`tokenize_training` | `scripts/train/p4_projection_gate.py:49-78`; `p5_train.py:146-155` | **MATCH** | Không thấy mask sai tĩnh; logits thực theo token chưa lưu |
| Ảnh/preprocessing | `load_image` RGB + cùng processor checkpoint | Cùng hàm/processor | `src/vlm_medical/data/preprocess.py:5-8`; `scripts/experiments/p4_paired_infer.py:270-288`; `scripts/p5/p5_infer.py:113-121` | **MATCH** theo code | Tensor ảnh lịch sử không được lưu để so sánh byte |
| Độ dài | target 128 gồm EOS, text 256, 32 visual queries | Cùng | Hai config; `diagnostics.json > model_metadata, training_targets` | **MATCH** | 388/2.000 target bị cắt, cũng xảy ra theo giao thức P4 |
| Optimizer | AdamW LR 1e-4, microbatch/accumulation 1, seed 42, constant schedule | Cùng key config | `diagnostics.json > protocol_equal`; `p5_train.py:114-119` | **MATCH về hyperparameter** | Cùng LR tác động lên 107M thay vì 2M params; tương đương mã số không có nghĩa là tương đương độ ổn định |
| Tham số trainable | Projection 1.968.640; ViT/QFormer/OPT frozen | QFormer 105.137.664 + query 24.576 + projection 1.968.640; ViT/OPT frozen | Checkpoint CPU; `scripts/p5/p5_common.py:50-75` | **INTENDED DIFFERENCE** | Giả thuyết kiến trúc chính |
| Dtype bridge | QFormer frozen theo BLIP-2 FP16; projection trainable | FP32 QFormer/query/projection wrapper; đầu ra projection về FP16 | `scripts/p5/p5_common.py:56-64`; `src/vlm_medical/models/bridges.py` | **INTENDED DIFFERENCE** | Cần đo activation/gradient để loại trừ mất tín hiệu |
| Decoding/metrics | Greedy, EOS 2 override, 128 tokens; corpus BLEU | Dùng chung helper/metric | `scripts/experiments/p4_paired_infer.py:41-73`; `scripts/p5/p5_infer.py:113-135`; `src/vlm_medical/eval/text_metrics.py:26-50` | **MATCH** theo source/config | Không có token IDs/logits để kiểm tra điểm rẽ đầu tiên |

## 3. Kiểm tra loss, gradient và checkpoint

`tokenize_training` tạo `[BOS, prompt, target, EOS]`; labels của BOS/prompt là `-100`, còn target/EOS đúng ID (`scripts/train/p4_projection_gate.py:62-78`). Giao diện Transformers BLIP-2 đã cài, `.../transformers/models/blip_2/modeling_blip_2.py:1701-1746`, cắt logits theo độ dài labels rồi shift logits/labels một vị trí trước `CrossEntropyLoss` mặc định `ignore_index=-100`. Kiểm tra CPU toàn bộ 2.000 target xác nhận mask, alignment, EOS và hash nối reference. Target median 86 token, tối đa 376 trước cắt; 388 bị cắt xuống giới hạn, supervised tokens 4–128. Tỷ lệ lặp 4-gram trung bình của target chỉ 0,0139; tiền tố “A.\n…” không phải mẫu mục tiêu phổ biến. Không có bằng chứng tĩnh rằng loss bị mask hết hoặc prompt bị tính loss. Không có per-token logits đã lưu nên tác động động của từng token vẫn chưa được quan sát.

Checkpoint P5 chứa đúng **107.130.880** tham số adapter: QFormer 105.137.664/254 tensors; query 24.576/1; projection 1.968.640/2. Các nhóm khác giá trị gốc, không có phần tử nonfinite; thay đổi L2 tương đối lần lượt 0,0492, 0,1285, 0,1391. Frozen ViT/OPT được `requires_grad_(False)` và eval; trainer kiểm tra gradient frozen và version tại `scripts/p5/p5_train.py:141-207`. Chứng cứ này xác nhận nhóm trainable đã đổi, **không** xác nhận biểu diễn ảnh còn phân biệt các ảnh.

Log 2.000 cập nhật liên tục tại `runs/P5_qformer_20261025/P5.2_updates.jsonl`: loss trung bình 100 bước đầu 2,8810, 100 bước cuối 2,4239. Median norm gradient query tokens giảm từ **0,09639** (đầu) xuống **2,14e-7** (bước 701–800) và **6,33e-8** (100 bước cuối); 1.198/1.200 bước sau bước 800 dưới 1e-5. Trong 100 bước cuối, QFormer median 0,681 và projection 2,041 vẫn khác 0. Đây là **dấu hiệu gradient đi tới query suy yếu**, không chứng minh nó là nguyên nhân duy nhất hay rằng toàn bộ QFormer không học. Không có checkpoint trung gian/feature hoặc validation generation theo bước để xác định thời điểm đầu ra bắt đầu sụp.

## 4. Kết quả định lượng tái kiểm

BLEU-1…4 là corpus BLEU theo token `[a-z0-9]+`, clipped n-gram precision và brevity penalty, không smoothing; ROUGE-L là trung bình theo case (`src/vlm_medical/eval/text_metrics.py:1-50`). METEOR lấy từ artifact đã lưu, không tính lại trong postmortem. Bootstrap là paired percentile 95%, 1.000 lần seed 42, MIMIC cluster theo subject (121 cluster), Indiana theo study (500), `scripts/p5/p5_evaluate.py:189-231`.

| Cohort/model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR | Unique / 500 | Hit limit / 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MIMIC P4 M1 | 0,17561 | 0,09346 | 0,05537 | 0,03464 | 0,16633 | 0,13988 | 311 | 113 |
| MIMIC P5 M3 | 0,02810 | 0,01451 | 0,00749 | 0,00197 | 0,02942 | 0,01890 | 2 | 500 |
| Indiana P4 M1 | 0,22881 | 0,12910 | 0,08203 | 0,05152 | 0,20952 | 0,21235 | 337 | 94 |
| Indiana P5 M3 | 0,03281 | 0,01421 | 0,00564 | 0 | 0,03012 | 0,02478 | 2 | 500 |

Paired Δ BLEU-4 **P5 − P4**: MIMIC **−0,0326666025429177**, CI95 **[−0,04494628927985667; −0,026138784306282362]**; Indiana **−0,05151685011825502**, CI95 **[−0,0574076372468186; −0,04620050574398297]**. Nguồn `runs/P5_qformer_20261025/P5.3_cpu_analysis/metrics.json > datasets.*.m3_minus_m1_bleu4`. Cả hai CI loại 0 theo phương pháp trên. BLEU-4 Indiana = 0 do không có 4-gram khớp theo corpus implementation; riêng giá trị này không chứng minh vô giá trị lâm sàng.

P5 có đúng hai chuỗi khác nhau mỗi cohort: MIMIC 286/214, Indiana 465/35, bắt đầu bằng “A. B. C.” hoặc lặp “A.”. Cả 1.000 báo cáo đều 128 generated tokens, `hit_max_length=true`, không rỗng. Mean repeated 4-gram fraction P5 MIMIC 0,6216 và Indiana 0,4004 (P4 lần lượt 0,2161/0,2197). Đây là suy sụp đầu ra quan sát trực tiếp. Chưa lưu raw token IDs/logits, không xác định được token đầu tiên lệch khỏi cấu trúc hợp lệ hoặc xác suất EOS; text cho thấy vấn đề xuất hiện ngay đầu báo cáo.

Clinical labeler: theo historical joint-definite, P5 definite-label coverage = 0 ở cả cohort, micro/macro-F1 **N/A**, không đổi thành 0. Theo GT-definite/prediction-missing→negative, P5 micro/macro-F1 đều 0; P4 MIMIC 0,1372/0,0798; Indiana 0,0488/0,0225. Candidate omission theo labeler MIMIC 606→658, Indiana 202→209; P5 không có unsupported-positive labeler candidate vì gần như không đưa ra phát hiện. Những con số này không phải thẩm định ảnh hoặc bác sĩ (`metrics.json`, `P5.3_cpu_analysis/error_pairing.json`).

## 5. Ma trận giả thuyết xếp hạng

Chi tiết máy đọc được: `diagnostic_matrix.csv`.

| ID | Hypothesis | Evidence supporting it | Evidence against it | Verification status | Confidence | Next discriminating test |
|---|---|---|---|---|---|---|
| H1 | Biểu diễn điều kiện ảnh/visual prefix mất thông tin hoặc bị QFormer đẩy vào vùng suy biến trong huấn luyện | Hai mẫu cho 1.000 ảnh; gradient query gần 0 từ ~bước 700; target đa dạng; projection/QFormer vẫn đổi | Không có saved visual features hay outputs theo bước | **LIKELY** về cơ chế, chưa xác định bước gây ra | Trung bình-cao | CPU probe nhẹ trên tensor adapter nếu đủ; khi được duyệt, so sánh prefix embeddings/entropy trên ảnh khác nhau tại checkpoint gốc/P5 và các mốc train |
| H2 | LR/schedule của P4 không ổn định cho 107M params trainable của P5 | Cùng LR 1e-4 cho 54,4× params, gradient query suy yếu, đầu ra lặp | Loss finite/giảm; không có sweep hay intermediate validation | **POSSIBLE** | Trung bình | Gate đối chứng LR thấp hơn **chỉ sau phê duyệt**, giữ mọi biến khác; theo dõi gradient/query diversity |
| H3 | Greedy decoder/EOS bộc lộ vòng lặp nhưng không phải mismatch giao thức | 1.000/1.000 chạm 128; 2 mẫu lặp; không phát EOS | Helper, EOS, prompt và config cùng P4; P4 không sụp | **CONFIRMED** về triệu chứng, **UNSUPPORTED** là nguyên nhân độc lập | Cao về triệu chứng | CPU kiểm thử stopping với token giả; GPU trace logits chỉ nếu được duyệt; không đổi decoding để tuyên bố cứu model |
| H4 | Sai mask/shift khiến loss không học report | Có thể gây giảm loss giả nếu có bug | Code shift đúng, CPU audit 2.000/2.000 mask/target/EOS đúng; loss có gradient | **UNSUPPORTED** theo chứng cứ hiện có | Thấp | Unit test causal loss với logits tổng hợp; per-token trace nếu có lần chạy được duyệt |
| H5 | Nạp sai checkpoint/pairing/reference/preprocessing | Có thể gây metric rơi mạnh | Hash/ID/mapping/decoder và eval code khớp; adapter khác gốc; đúng checkpoint SHA | **UNSUPPORTED** theo artifact hiện có | Thấp | CPU kiểm tra adapter copy vòng khép kín; nếu còn nghi ngờ, kiểm tra forward định danh trong gate được duyệt |
| H6 | Target bị cắt hoặc dữ liệu kém gây lặp | 388/2.000 target bị cắt; pretraining contamination chưa rõ | P4 dùng cùng target; target lặp 4-gram thấp; pairing SHA hợp lệ | **POSSIBLE** là yếu tố phụ, không đủ giải thích riêng | Thấp | Phân tầng lỗi theo target length/truncation, so sánh với P4; không dùng locked test |
| H7 | Khác biệt dtype FP32 bridge → FP16 làm mất tín hiệu ảnh | Biên dtype là điểm nguy cơ thực | Không có NaN, mọi nhóm adapter đổi, peak bình thường | **POSSIBLE / BLOCKED** ở mức tensor | Thấp-trung bình | Prefix statistics, finite/range/variance, cosine giữa ảnh ở các module trong gate được duyệt |

## 6. Giới hạn còn lại

Không có feature, attention map, logit/token ID, per-token loss hoặc validation generations theo checkpoint trung gian. Không thể phân biệt dứt khoát suy biến do tối ưu QFormer, cầu FP32→FP16, hay tương tác decoder với prefix chỉ bằng artifact hiện có. Phải giữ H1/H2/H7 là giả thuyết; không suy luận từ BLEU hoặc labeler thành kết luận lâm sàng. Indiana là external cohort; nguy cơ tiền huấn luyện MIMIC không được loại trừ. Không dùng locked test để chọn fix.
