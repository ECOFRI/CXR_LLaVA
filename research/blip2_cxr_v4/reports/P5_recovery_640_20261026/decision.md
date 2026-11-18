# Quyết định chẩn đoán generation 640 (26-10-2026)

**CPU validation: PASS (9/9). GPU smoke/evaluation: HOLD, chưa được phê duyệt. Training: không thực hiện.**

Đã xác minh helper chung có thể nhận `max_new_tokens=640` trong mock CPU, giữ EOS override2 thay cho EOS50118 của checkpoint, không truyền `max_length` tường minh và đếm continuation không gồm synthetic BOS. Checkpoint P5.2 và interface adapter khớp hash/shape; causal mask/shift test đạt. Chưa có lần chạy BLIP-2 640 thật, nên không biết EOS/truncation, chất lượng, VRAM hoặc thời gian tại 640.

Giới hạn 128 không giải thích riêng việc P5 đã lặp ngay từ đoạn đầu và chỉ có hai mẫu report, nhưng có thể che khuất phần tiếp theo. Bước đề xuất là smoke đối chứng 40 study × P4 M1/P5 M3 = 80 reports, chỉ đổi `max_new_tokens`, ngân sách xin duyệt 1 K80/≤90 phút/≤10,0 GiB reserved. Phải dùng runner riêng để không sửa hoặc ghi đè P4/P5; phê duyệt GPU cần được cấp riêng. Không mở rộng full cohort trước khi có số liệu chi phí và validity từ smoke.

Xem `recovery_plan_640.md` và `cpu_validation_results.json`. Không có `generation_smoke_test_metrics.json` hay `paired_generation_metrics.json` vì chưa thực hiện.
