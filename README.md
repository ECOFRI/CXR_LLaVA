# CXR-LLaVA Improvement

Nghiên cứu sinh báo cáo X-quang ngực bằng CXR-LLaVA, khảo sát CLEAR và pilot BLIP-2.
Output model là văn bản thực nghiệm, không phải kết luận lâm sàng hoặc lời khuyên y khoa.

- Agent bắt đầu tại [AGENTS.md](AGENTS.md).
- [Trạng thái và kết quả hiện tại](docs/STATUS.md) · [Quyết định bền vững](docs/memory/DECISIONS.md).
- [Bản đồ code, dữ liệu và môi trường](docs/MAP.md) · [Kế hoạch hiện hành](plan/PROJECT_PLAN.md).
- [Tiêu chí dự án](CRITERIA.md) · [Template báo cáo](experiment_analysis_template.md).
- [Model card upstream và hướng dẫn CXR-LLaVA](docs/MODEL_CARD.md).
- [Modules nghiên cứu](research/README.md) · [Kế hoạch BLIP-2 V4](research/blip2_cxr_v4/plan/VLM_MEDICAL_CLAUDE_CODEX_MASTER_PLAN_V4_2026-10-22.md).
- [Snapshot báo cáo trình bày](docs/report_snapshot/README.md) · [Nhật ký memory](docs/memory/LOG.md).

Kiểm trạng thái run bằng Python standard library từ root (không load model):
```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m research.project status
```
Baseline X00 có 3.667 output; P4 projection tăng lexical overlap trên cohort pilot;
P5 Q-Former bị collapse và HOLD. Chưa chứng minh CLEAR fusion cải thiện báo cáo.
Đối chiếu số liệu và giới hạn tại STATUS trước khi sử dụng; không suy quyền chạy từ kế hoạch.
