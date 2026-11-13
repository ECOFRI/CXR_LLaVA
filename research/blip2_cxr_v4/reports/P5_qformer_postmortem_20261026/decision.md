# Quyết định P5 Q-Former postmortem (26-10-2026)

**HOLD** mọi huấn luyện/suy luận GPU tiếp theo của cấu hình P5 hiện tại. P5.1/P5.2/P5.3 **PASS vận hành** (20/2.000 updates, 1.000 generations) nhưng **FAIL mục tiêu khoa học**: P5 − P4 paired corpus BLEU-4 MIMIC −0,0326666 (CI95 [−0,0449463; −0,0261388]), Indiana −0,0515169 (CI95 [−0,0574076; −0,0462005]). P5 chỉ tạo hai mẫu báo cáo mỗi cohort; mọi output chạm 128 tokens.

**Đã xác minh:** cohort/pairing/config chính khớp; checkpoint P5 cuối đúng SHA và 107.130.880 trainable parameters; target/mask CPU đúng; gradient query suy yếu rõ ở log; BLEU/ROUGE tái tính khớp artifact. **Chưa xác minh:** cơ chế chính làm mất tín hiệu ảnh, điểm bắt đầu suy biến, per-token logits/EOS, feature variance, giá trị lâm sàng. Chi tiết và đường dẫn bằng chứng ở `root_cause_investigation.md` và `diagnostics.json`.

**Điều kiện xem xét GO cho một đề xuất GPU mới, không phải cấp phép chạy:** CPU tests loss/loader/stopping và integrity đều PASS; phép kiểm phân biệt prefix/decoder được thiết kế với mẫu cố định và ngưỡng pre-register; nêu rõ một thay đổi khoa học hoặc một defect được tái lập, có đối chứng P4/P5 cũ; phê duyệt ngân sách và cấu hình riêng. Nếu artifact/hash sai hoặc sửa cần vượt giới hạn máy, chuyển **BLOCKED**. Nếu gate được duyệt sau này vẫn sụp hoặc không vượt P4 theo so sánh paired, giữ P4 projection và bỏ cấu hình QFormer này.

**Không chạy follow-up trong postmortem.** `recovery_plan.md` là phương án kiểm chứng theo giai đoạn, không phải lệnh thực thi.
