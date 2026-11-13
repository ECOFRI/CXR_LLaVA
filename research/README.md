# Research workspace

Đầu mối kế hoạch: [PROJECT_PLAN](../plan/PROJECT_PLAN.md). Kết quả, input/output và nội dung trình Prof: [nhật ký E/G](../reports/2026-10-17_E_G_execution_status_log.md). Registry/state cũ là snapshot lịch sử; không dùng một dòng cũ để suy experiment chưa chạy.

## Một lệnh quản lý và tổng hợp

```bash
python3 -m research.project status
python3 -m research.project status --format json
python3 -m research.project status --output reports/run_catalog_new.md
```

Chạy từ root repo, chỉ cần Python standard library. `status` tự phát hiện manifests ở từng run và phase, giữ nguyên raw status/decision, nêu rõ manifest thiếu hoặc lỗi. Không import torch, không tự chạy inference và không sửa output cũ. `--root` hỗ trợ checkout khác; `--output` chỉ tạo file mới.

## Các module giữ lại để tái sử dụng

| Nhóm | Modules / chức năng |
|---|---|
| Quản lý / I/O | `project.py`: catalog run/phase; `common.py`: paths, hashing, CSV/JSON, run lifecycle. |
| Audit / dữ liệu | `e_audit`, `split_audit`, `e1_data_gate`, `inventory_once`: kiểm provenance/nhãn/split; inventory chỉ khi cần. |
| Encoder / feature | `encoder_harness`, `clear_encoder_harness`, `clear_cpu_gates`, `e1_features`: native features và smoke gates. |
| Probe / similarity / fusion baseline | `e1_probe`, `v3_cached`, `v3_followup`, `v3_e5_execute`, `v3_e5_fit_audit`: dùng cache, patient-grouped fits và metric. |
| CLEAR direct | `e2_direct`; E2 analysis/finalize giữ vì còn dependency trong analysis chain. |
| Prompt / generation | `e4_b1_cohort`, `e4_b1_prepare`, `e4_b1_clear`, `e4_b1_gpu`, `e4_b1_background.sh`: cohort, evidence, templates, worker và resume guard của B1. |
| Evaluation | `e4_b1_label`, `e4_b1_analyze`, `e4_b1_validate`, `e4_b1_synthetic`, `e4_b1a_posthoc`, `e4_a3a`: label states, gates, post-hoc. |
| Reports / plots | `e1_analysis_report`, `e2_analysis_report`, `plot_audit`, `plot_encoder` và helpers còn được gọi. |

Các module B1 và source của completed runs có cấu hình/path đã khóa. Tái sử dụng hàm thuần như `label_state`, `gate`, `change_category`, `derangement`, `chat_for`; giữ nguyên snapshot để audit. Không gọi entrypoint có hardcoded output cũ như một experiment mới. New runs cần config/cohort/output mới và phạm vi thực thi đã được giao.

Không gộp encoder, inference, metrics và report text vào cùng một file vì các môi trường CXR K80 và CLEAR CPU khác nhau. `project.py` là đầu mối quản lý chung, các module nhỏ là implementation tái sử dụng.

## File đã loại bỏ

10 script chạy một lần và 14 plan cũ đã xóa theo yêu cầu; archive trung gian cũng đã xóa. Danh sách quyết định và giới hạn tái lập trong [PROJECT_PLAN](../plan/PROJECT_PLAN.md). `research.project` hiện chỉ đọc catalog, không có lệnh history/restore.

## Kiểm tra nhẹ

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_project.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_e4_b1*.py' -v
```

`research/runs`, inventory, images, feature caches, environments và upstream giữ nguyên. Dữ liệu nghiên cứu và report sinh ra không phải kết luận lâm sàng đã được xác nhận.
