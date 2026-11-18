# Nhật ký kết quả (giữ mục cũ, mới nhất ở trên)

- 2026-11-18 — WO_F1 COMPLETE: A2/A3/A4 (Qwen3-1.7B LoRA, 30k Findings) PASS, không collapse; eval 1.000 dev; GPU 35.634s (9,90h). [Report](../../reports/2026-11-13_F1_clear_tokens_qwen_30k.md).
  Các arm A vượt M1 rõ (A2−M1 MIMIC METEOR Δ [0,038; 0,067]). A4 (CLEAR tokens) nhỉnh hơn A2 về text trên cả hai cohort; A3 (BCE) không đổi text; micro-F1 lâm sàng giữa A2/A3/A4 chưa phân biệt được. Shuffle chỉ đổi 10/100 report nên A4 dùng token yếu.

- 2026-11-13 — WO_F0 COMPLETE technical preflight: đủ 2.500 feature P4, disk256/load64; LoRA/bridge 20+20 updates, tùy chọn256 thêm3; GPU tổng 1.191,4s (19,86 phút), không upload MIMIC/commit. [Report](../../research/rrg_f0/F0_preflight_report.md).
  Batch5: LoRA 0,274s/mẫu, bridge 0,179s/mẫu; CPU tests5/5 + catalog2/2 PASS; loss giảm trên minibatch lặp, BCE giả, không đo chất lượng; generation cap128 chạm1/8 và8/8.

- 2026-11-13 — WO_ORG: tạo cửa vào STATUS/MAP/memory, chuyển README model card bằng git mv, cập nhật plan P0–P5; chỉ tài liệu, không chạy model/commit. [WO](../../plan/work_orders/WO_ORG_2026-11-13.md), [STATUS](../STATUS.md).
  Kiểm tra: AGENTS 47/90, STATUS 50/120, README 20/40, CLAUDE đúng `@AGENTS.md`; 125 link tương đối tồn tại; test_project 2/2 PASS; status đã đối chiếu. Đây là kiểm tra tài liệu, không phải thí nghiệm/benchmark.
