# P5 Q-Former: kế hoạch phục hồi có điều kiện (26-10-2026)

**Trạng thái hiện tại: HOLD mọi thực thi GPU.** Các giai đoạn sau là đề xuất cần phê duyệt riêng. Cơ sở: `root_cause_investigation.md` và artifact CPU `diagnostics.json`. Giữ P4 projection M1 làm đối chứng cố định; không sửa hoặc ghi đè P4/P5.

## Stage 0 — Cố định chứng cứ

1. Giữ nguyên `runs/P4_projection_pilot_20261023/` và `runs/P5_qformer_20261025/`; lưu SHA của config, 3 manifest, P4 M1 adapter, P5 P5.2 adapter, hai tập prediction vào sổ experiment mới. Script postmortem đã ghi hash hiện tại ở `diagnostics.json`.
2. Ký hiệu failure signature trước khi thử fix: P5 MIMIC/Indiana chỉ 2 unique reports mỗi cohort; 1.000/1.000 chạm 128 tokens; BLEU-4 0,001971/0; query-grad median xuống 2,14e-7 ở bước 701–800. Kiểm lại hash và pairing trước mỗi lần phân tích sau.
3. Nếu phát hiện source artifact đổi hash hoặc pairing lệch, dừng: `BLOCKED` về tính toàn vẹn, không dùng kết quả mới để so sánh.

## Stage 1 — Chẩn đoán CPU và đặc tả sửa tối thiểu

Ưu tiên phân biệt **mất tín hiệu thị giác** với **lỗi loss/decoder/loader** trước khi đề xuất một lịch huấn luyện khác. Không sửa production pipeline trong stage này; mọi prototype/test đặt ngoài các file P4/P5 lịch sử.

| Thứ tự | Kiểm tra CPU nhỏ | Kết quả PASS | Nếu FAIL, đặc tả sửa tối thiểu |
|---|---|---|---|
| 1 | Recheck manifest/hash, stable key, reference và checkpoint key/shape/cursor; round-trip adapter copy vào surrogate | Hash/ID và mọi tensor bằng nhau sau copy; cursor 2.000 | Sửa duy nhất loader/checkpoint mapping; thêm identity assertion trước GPU |
| 2 | Tokenizer local và prompt: kiểm BOS/EOS/PAD, 2.000 target SHA, `[BOS]+prompt+target+EOS`, 128/256 limits | 2.000/2.000 alignment đúng; prompt/pad bị mask, target/EOS supervised | Sửa hàm target/mask, giữ report và prompt theo contract; đóng băng lại config/hash nếu khoa học bị ảnh hưởng |
| 3 | Causal-loss unit test với logits tổng hợp theo installed Transformers shift/`ignore_index=-100` | Chỉ target/EOS ảnh hưởng loss; thay logits ở vị trí ignored không đổi loss | Sửa alignment/attention mask và yêu cầu scientific review nếu mục tiêu tối ưu thực tế đổi |
| 4 | Surrogate gradient flow + frozen mode, shape/dtype bridge FP32→FP16 | Query/QFormer/projection có gradient finite và cập nhật; ViT/OPT không gradient; tensor shape 32×768→32×2560 | Sửa graph/dtype/copy với regression test; không đổi số query/precision tùy tiện |
| 5 | EOS stopping với token sequence giả; `max_new_tokens=128`; decoded continuation bỏ synthetic BOS | EOS2 kết thúc ngay; thiếu EOS đúng là chạm 128; không lẫn prompt vào output | Sửa helper chỉ khi có lỗi tái lập; giữ decoding contract P4 |
| 6 | Phân tầng các 2.000 target theo truncation/độ dài và 1.000 output theo pattern; kiểm tra reference dài/ngắn | Không có artifact pairing sai hoặc target “A.” lặp | Nếu lỗi data-link, sửa manifest qua approval; nếu không, ưu tiên visual-prefix probe |

Đã có CPU audit theo mục 1, 2, một phần 4 và phép kiểm source cho 3/5. Phần còn lại là tiêu chí rõ ràng để xác nhận trước GPU, không giả định chúng đã PASS. Nguồn số liệu hiện tại: `postmortem_diagnostics.py`, `diagnostics.json`; `scripts/train/p4_projection_gate.py:49-78`; `scripts/p5/p5_common.py:50-134`; `scripts/experiments/p4_paired_infer.py:41-73`.

**Bước chẩn đoán phân biệt quan trọng nhất sau CPU:** nếu Stage 1 không tìm lỗi phần mềm, xin phê duyệt một gate GPU nhỏ để trích xuất **chỉ thống kê ẩn danh**: variance/cosine của vision features, QFormer query states và projected prefix giữa ≥10 ảnh khác nhau; norm/finite theo layer; top-k token/probability của 10 câu sinh; tỷ lệ EOS theo bước; lưu không ảnh, không dữ liệu định danh. So sánh mô hình BLIP-2 gốc, P4 M1 và P5 checkpoint trên cùng ảnh. Kết quả sẽ xác định liệu prefix mất phân biệt trước decoder hay decoder phản ứng bất thường với một prefix vẫn đa dạng. Đây là **đề xuất**, chưa được phép chạy.

## Stage 2 — Đề xuất thí nghiệm nhỏ có đối chứng

Chỉ đề xuất gate huấn luyện nếu Stage 1 xác định lỗi có thể sửa hoặc phép probe chứng minh mất tín hiệu ảnh. **Một thay đổi giả thuyết tại một thời điểm**:

- Nếu có lỗi mask/loader/dtype tái lập: patch tối thiểu + unit test; chạy lại gate 20 cập nhật từ BLIP-2 gốc, cùng 20 study/seed/prompt/target như P5.1, không tiếp tục checkpoint P5 lỗi.
- Nếu code/parity đúng nhưng visual-prefix suy biến: giữ architecture/cohort/decoding cố định, đề xuất một LR/schedule thay thế được định nghĩa trước trong approval. LR mới là **thay đổi khoa học**, không tự áp dụng. Đối chứng P5 cũ và P4 M1 chỉ dùng artifact hiện có. Theo dõi gradient query ở từng bước và đa dạng prefix trên ảnh cố định.
- Nếu chỉ decoder có lỗi chứng minh bằng test: sửa helper, tái đánh giá checkpoint P5 trên subset được phê duyệt trước khi huấn luyện lại; không xem decoding khác giao thức là phục hồi model.

**Giới hạn đề xuất tham chiếu lịch sử, chưa được duyệt:** đúng 1 Tesla K80; gate 20 optimizer updates, ≤900 giây cumulative, peak reserved ≤10,0 GiB; nếu qua gate, 2.000 updates/≤14.400 giây; đánh giá ≤1.000 reports/≤10.800 giây. Không chuyển thời gian thừa giữa stage, không thêm GPU, không dùng lại ngân sách P4. P5 cũ 20 bước/193 giây, 2.000 bước/9.850 giây và peak gần sát 10 GiB; đây là đo **cấu hình cũ**, không phải dự báo bảo đảm cho fix. Gate mới phải đo VRAM/throughput thực; muốn 2.000 updates trong 4 giờ thì thời gian trung bình cả overhead ≤7,2 giây/update.

**Execution PASS cho gate 20 bước:** đúng 20 updates, loss/gradient hữu hạn, ViT/OPT frozen, query/QFormer/projection cập nhật, checkpoint save/reload bitwise/shape hợp lệ, VRAM/thời gian đúng hạn, không lỗi CUDA/OOM. **Scientific signal** tách riêng: gradient query không suy yếu hệ thống về gần 0 trong 20 bước; visual-prefix variance giữa ảnh và token diversity không thấp hơn mức baseline đã pre-register; ≥10 output validity cases không lặp vô hạn/chạm 128 hàng loạt. Ngưỡng định lượng cho feature-diversity phải cố định từ diagnostic baseline trước khi chạy, tránh điều chỉnh sau khi xem kết quả. 20 bước không đủ chứng minh hiệu quả báo cáo; chỉ cho phép xem xét giai đoạn kế tiếp.

**Stop ngay:** hash/pairing sai; thay đổi không được phê duyệt; >10,0 GiB; hết thời gian stage; NaN/Inf; frozen params đổi; target/mask sai; mất gradient query kéo dài theo ngưỡng pre-registered; output collapse lặp ở validity gate. Nếu gate cho thấy tốc độ không thể đáp ứng 4 giờ, HOLD training đầy đủ.

## Stage 3 — Đề xuất tái đánh giá có điều kiện

Chỉ sau checkpoint huấn luyện đầy đủ, integrity và output-validity gate PASS. Dùng đúng MIMIC validation 500/Indiana 500 cố định; P4 M1 predictions/reference đã lưu, không tái sinh. Pre-register primary **paired corpus BLEU-4 P5 mới − P4 M1** cho từng cohort; BLEU-1/2/3, ROUGE-L, METEOR, length, repetition, truncation, unique reports, empty rate là secondary. Bootstrap paired 1.000 lần seed42, cluster MIMIC theo patient/subject và Indiana theo study. CI95 cần loại 0 và delta dương trên cả hai cohort để xem xét lexical improvement; nếu không, không tuyên bố vượt P4.

Thêm clinical proxy theo historical joint-definite và GT-definite/prediction-missing→negative, báo cáo per-class TP/FP/FN/TN/precision/recall/F1, coverage, uncertain/missing; F1 không xác định để N/A. Có bộ review mù gồm disagreements và controls cân bằng hai dataset; báo cáo *candidate* omission, unsupported finding, negation, contradiction, không gọi là lỗi đã xác nhận. Để tuyên bố **clinical utility**, cần bác sĩ đọc ảnh độc lập/chuẩn lâm sàng và đánh giá độ an toàn, vượt phạm vi hiện tại. BLEU tăng một mình chỉ hỗ trợ kết luận lexical. Kiểm soát multiple comparisons và nguy cơ MIMIC pretraining contamination.

## Stage 4 — Quyết định

- **GO chỉ cho bước review/đề xuất tiếp:** CPU unit tests và integrity PASS; cơ chế lỗi được phân biệt bằng probe; fix là thay đổi tối thiểu có giả thuyết, controls, ngưỡng và ngân sách được đóng băng. GO trong postmortem **không cấp phép GPU**.
- **HOLD:** bằng chứng hiện tại (gradient query gần 0, đầu ra lặp) nhưng chưa rõ cơ chế; CPU test không tìm defect chắc chắn hoặc thiếu feature/logit trace. Đây là quyết định hiện hành.
- **BLOCKED:** artifact mất/không khớp hash, checkpoint không đọc được, không thể bảo đảm protocol, hoặc sửa bắt buộc vượt 1 K80/VRAM/thời gian được phê duyệt.
- **Giữ P4 và bỏ cấu hình QFormer hiện tại** nếu gate có đối chứng sau fix vẫn tạo 2 mẫu/lặp/chạm limit, không hồi phục gradient/visual diversity, hoặc không có lợi ích lexical paired ổn định trên cả cohort trong ngân sách hợp lệ. Không lặp lại 2.000 bước chỉ để tìm may mắn.
