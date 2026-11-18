# P5 Q-Former — chẩn đoán giới hạn sinh 640 token

**Ngày:** 2026-10-26  
**Phân loại:** kế hoạch nghiên cứu và CPU validation; không phải inference, benchmark hay xác nhận lâm sàng.  
**Trạng thái:** CPU validation PASS; GPU smoke/evaluation **HOLD, chờ phê duyệt riêng**.

## 1. Câu hỏi và giới hạn suy luận

Cần phân biệt (a) giới hạn 128 token cắt mất phần report hợp lệ, (b) mô hình đã sụp trước mốc 128, (c) cả hai. Artifact P5 đã cho thấy 1.000/1.000 output chạm 128 token và chỉ 2 mẫu report mỗi cohort; đoạn mở đầu đã lặp “A.”/“A. B. C.”. Dưới greedy decoding và cùng checkpoint/input, tăng ngưỡng dừng chỉ có thể cho thấy **phần tiếp theo**, không tự sửa đoạn đầu đã sinh. Vì raw token IDs/logits không lưu, tính bất biến của 128 token đầu chỉ là kỳ vọng theo mã, cần kiểm trong lần chạy được duyệt.

Train target P5 vẫn tối đa 128 token kể cả EOS (`scripts/train/p4_projection_gate.py:62-78`); 640-token generation là **thử nghiệm chẩn đoán lệch chiều dài train target**, không thay đổi training và không thay thế phép đánh giá P5 128 token. Không so sánh trực tiếp BLEU của giao thức 640 với BLEU của giao thức 128 như cùng một thí nghiệm.

## 2. Stage 1 — CPU validation đã thực hiện

Chạy `cpu_validate_640.py` với `CUDA_VISIBLE_DEVICES=''`; kết quả `cpu_validation_results.json`: **9/9 PASS**. Các kiểm tra:

| Kiểm tra | Bằng chứng | Kết quả |
|---|---|---|
| Target/EOS mask | `tokenize_training` có BOS+prompt masked và target+EOS supervised | PASS |
| Causal shift và `ignore_index=-100` | Logits tổng hợp: đổi ignored logits không đổi loss; đổi logits dự báo target làm loss đổi theo hướng dự kiến | PASS |
| Checkpoint loader identity | P5.2 checkpoint SHA khớp P5.2/P5.3 status, config/manifest hash và cursor 2.000 | PASS |
| Adapter interface | 257 tensor adapter có key/shape tương ứng safetensors BLIP-2 gốc; 107.130.880 params, đúng 3 nhóm | PASS |
| Special tokens | Checkpoint `generation_config.json`: EOS **50118**, `max_length=20`, `max_new_tokens=null`; tokenizer/contract: BOS=2, EOS=2, PAD=1 | PASS |
| Runtime helper 640 | Mock gọi **helper thực** `generate_single_sequence`: truyền `max_new_tokens=640`, không truyền `max_length`, `eos_token_id=None`, stopping criteria EOS2; config EOS của language model được khôi phục sau lời gọi | PASS |
| Dừng và đếm token | Synthetic sequence dừng khi token cuối bằng 2; `continuation_tokens` loại synthetic BOS, đếm 3 hoặc 640 token mới đúng | PASS |
| Smoke cohort cố định | 20 bản ghi đầu của từng cohort trong approved ordered manifest, 40 stable key duy nhất | PASS |

Config/manifest gốc được đọc nguyên trạng: P4 SHA-256 `d010d2cf32cb6cc9b96b834e5eb0a5df1d8bd13985ec31cf5faa447260471eb9`; P5 `841753a63743c62fe15be5cc2f9f2308a40d593668a891050da18f8e213475ce`; index `74965a79b31394a65af10dcb7987db0881e5083a8bb0da888bccbe120bd03175`. SHA-256 danh sách 40 stable keys theo thứ tự: `b9b847d4dbba794c514fe545fcce202b59657720f3a5b5721e88696e721378e0`.

**Mã thực:** `scripts/experiments/p4_paired_infer.py:41-73` xác định greedy helper/EOS và cách bỏ synthetic BOS; `scripts/p5/p5_infer.py:113-135` gọi helper và đếm `continuation.shape[1]`. Installed BLIP-2 `modeling_blip_2.py:1774-1842` chuyển `**generate_kwargs` xuống OPT `language_model.generate`; installed Transformers `generation/utils.py:1613-1623` ưu tiên `max_new_tokens` và đặt internal `max_length = input_ids_length + max_new_tokens`. Với synthetic BOS length 1, ngưỡng internal dự kiến 641. `max_length=20` trong checkpoint vì vậy không đè lên 640 trong API này.

**Giới hạn của PASS:** test dùng mock thay vì full model; chưa đo VRAM, thời gian, output hay logit của 640. Không gọi GPU. `p5_infer.py` lịch sử ràng buộc `contract_and_index` phải có generation **bằng P4**, status P5.2 và output `P5.3_predictions.jsonl`; vì vậy không thể chỉ sửa config P5 rồi dùng entrypoint cũ. Một runner chẩn đoán **riêng** phải import helper chung, chỉ override `max_new_tokens`, xác minh kwargs tại runtime, ghi output mới vào thư mục versioned, và không sửa pipeline sản xuất. Đây là đặc tả cho approval sau, chưa triển khai runner GPU.

## 3. Stage 2 — Đề xuất smoke test có đối chứng; chưa được phép chạy

**Giả thuyết:** giới hạn 128 che khuất EOS và phần sau có cấu trúc; giả thuyết cạnh tranh là loop lặp vẫn tiếp diễn đến 640. Pre-register 20 study MIMIC + 20 Indiana theo quy tắc Stage 1, cùng study/image cho P4 M1 và P5 M3, 80 reports tối đa. Dùng checkpoint hiện có, cùng processor, QA prompt, tokenizer, BOS/EOS/PAD, greedy, batch1, `use_cache`, beam1, repetition penalty1, no-repeat-ngram0. Chỉ `max_new_tokens: 128→640`; không thay temperature/top-p/prompt/weights. M0 không cần cho câu hỏi chính. Mọi output mới có run ID, model ID, study ID, image ID, manifest/config/checkpoint hash và số token mới.

**Kiểm tra bắt buộc trong runner được đề xuất:** ghi actual kwargs đến `model.generate`/OPT generate hoặc instrument call an toàn; `max_new_tokens=640`, không có `max_length` tường minh cạnh tranh, EOS override2 và custom stopping còn hiệu lực; ghi token IDs/tổng length/last token cho tối thiểu validity gate để đối chiếu số token, không lưu ảnh hay thông tin nhận dạng. Recompute SHA checkpoint trước GPU; lock GPU; timeout/watchdog và resource monitor; không ghi đè P4/P5. Validity gate 10 study đầu (5 mỗi cohort × 2 mô hình =20 outputs) trước phần còn lại; trạng thái PARTIAL nếu dừng giữa chừng.

**Metrics:** số token mới, EOS trước 640, hit640, empty/invalid, unique reports sau normalization, token và 4-gram repetition, latencies, throughput, peak allocated/reserved VRAM; ví dụ lỗi được khử nhận dạng. So sánh **trong cùng 40 study**, đặc biệt tiền tố 128 token giữa run mới và output cũ nếu có thể giải mã tương thích. Lưu raw token IDs của lần mới để xác định điểm đầu tiên bị lặp; artifact 128 cũ chỉ có text nên không thể so token bitwise với lịch sử.

**Execution PASS:** đủ 80 outputs hợp lệ, pairing/hash đúng, actual kwargs đã kiểm, time/VRAM trong giới hạn, có logs/status/metrics tái lập. **Scientific observation** riêng: nếu P5 vẫn gần như hai mẫu và lặp qua 640, bác bỏ ý tưởng rằng tăng limit một mình phục hồi report. Nếu EOS/đa dạng cải thiện, cần paired full cohort trước kết luận. Nếu chỉ dài hơn, không kết luận chất lượng tốt hơn.

**Ngân sách xin duyệt, chưa cấp phép:** 1 K80, ≤80 generations, ≤90 phút cumulative, peak reserved ≤10,0 GiB. Cơ sở thô: P4.3 lịch sử 15.243,8 giây/2.000 generations (~7,62 giây/report bao gồm overhead), P5.3 9.546,2 giây/1.000 (~9,55 giây/report). Nhân tuyến tính 5 lần cho 40+40 cho ~3.434 giây (~57 phút), rồi thêm dư địa cho tải mô hình/dao động. Đây **không phải** dự báo đo được cho 640; KV cache, EOS sớm và overhead có thể khiến độ dài/thời gian phi tuyến. Runner phải dừng trước 90 phút/10,0 GiB; không dùng ngân sách P4/P5 cũ.

## 4. Stage 3 — Quy tắc quyết định sau smoke

| Quan sát | Kết luận được phép | Bước kế tiếp |
|---|---|---|
| P5 EOS sớm, đa dạng hơn và giảm lặp | Limit 128 có thể che một phần nội dung; chưa chứng minh chất lượng | Xin duyệt paired full cohort với ngân sách đo từ smoke |
| P5 lặp và thường chạm 640 | Limit 128 không phải cách sửa collapse | HOLD full evaluation; ưu tiên visual-prefix variance, token-logit entropy/condition theo ảnh |
| Dài hơn/đa dạng hơn nhưng metrics hoặc lỗi text không tốt hơn | Thêm token không tương đương cải thiện báo cáo | HOLD, phân tích lỗi và đối chứng |
| Pairing/hash/runtime config không khớp hoặc OOM/time limit | Không có kết quả hợp lệ | BLOCKED/PARTIAL; không tự đổi khoa học hoặc tăng ngân sách |

## 5. Stage 4 — Đề xuất full paired evaluation; chưa được phép chạy

Chỉ nếu smoke hợp lệ và có phê duyệt ngân sách **mới**. Dùng đúng 500 MIMIC validation và 500 Indiana, cả P4 M1/P5 M3 tại 640: **2.000 reports mới**. Output/result vào run versioned riêng; mọi artifact 128 giữ nguyên. Báo cáo corpus BLEU-1…4, ROUGE-L, METEOR nếu cùng scorer, EOS/truncation, unique/repetition/length, paired Δ P5−P4 và CI95 bootstrap 1.000 lần seed42 (MIMIC cluster theo subject, Indiana theo study). Review blind text-error candidates; không gọi hallucination được xác nhận lâm sàng khi chưa có bác sĩ/ảnh adjudication.

Không mặc định 3 giờ P5.3 đủ: 2.000×640 có thể tốn gấp nhiều lần. Dùng measured smoke latency và VRAM để lập budget; nếu cần vượt tài nguyên khả dụng, HOLD/BLOCKED thay vì giảm cohort lặng lẽ. Không thay thế P5 primary 128-token metric bằng kết quả này; đây là sensitivity analysis theo giới hạn generation. Chỉ BLEU tăng không chứng minh clinical utility.

## 6. Stage 5 — Quyết định

**Hiện tại HOLD GPU:** CPU đã PASS, nhưng chưa có 640 report và chưa phê duyệt smoke. Một GO sau này chỉ là GO cho đề xuất/approval theo stage, không cấp quyền training. Nếu smoke thất bại vì lặp tới 640, tập trung điều tra suy biến điều kiện ảnh/gradient query theo postmortem; giữ P4 projection làm baseline. Nếu smoke hợp lệ và tín hiệu cải thiện có nội dung, xin duyệt full paired evaluation rồi mới xem xét phục hồi khoa học.

## Tái lập CPU

Từ gốc repository, trong env K80 có sẵn:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/student4/anaconda3/envs/cxr-llava-k80/bin/python research/blip2_cxr_v4/reports/P5_recovery_640_20261026/cpu_validate_640.py
```

Script chỉ ghi `cpu_validation_results.json` trong cùng thư mục. `generation_smoke_test_metrics.json` và `paired_generation_metrics.json` **chưa được tạo** vì các giai đoạn GPU chưa được duyệt/thực hiện.
