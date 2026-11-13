# CXR-LLaVA × CLEAR — kế hoạch dự án hiện hành

**Cập nhật 13/11/2026.** Đây là đầu mối kế hoạch tổng; [STATUS](../docs/STATUS.md) là bản trạng thái ngắn, [MAP](../docs/MAP.md) dẫn tới evidence theo task. Chi tiết kết quả, input/output và giới hạn để báo cáo Prof ở [nhật ký E/G](../reports/2026-10-17_E_G_execution_status_log.md). Protocol và quyết định đã khóa nằm trong từng run; bản kế hoạch này không sửa các ngưỡng đó.

## Mục tiêu và hiện trạng

Đánh giá CLEAR có giúp CXR-LLaVA sinh báo cáo X-quang ngực tốt hơn hay không. Ảnh frontal → report tiếng Anh; dùng cho nghiên cứu. Phân biệt ba đường: frozen-feature probes, evidence qua prompt, và trained adapter/fusion.

| Nhánh | Đã làm | Quyết định hiện tại |
|---|---|---|
| Baseline / E0 / E-AUDIT | Indiana 3.667 reports; audit CheXpert baseline | Có pipeline và output để so sánh; chưa xác minh clinical correctness. |
| E0A/E0B/E1/E2 | Hai encoder chạy được; NIH matched probes; CheXpert direct contrastive | CLEAR có tín hiệu finding trong các phép thử exploratory; full concept bank còn thiếu assets. |
| E1.3/E5 | CKA/ridge; 780 fits, frozen-feature concat | Concat không tăng AP so với CLEAR đơn; không suy ra mọi fusion đều thất bại. |
| G1/G2/E4-R4 | Readiness + cùng ảnh CLEAR evidence; đủ 80/80 reports | PASS kỹ thuật; C2a/C3 trùng text 17/20, chưa chứng minh downstream benefit. |
| E4-A1/A2/A3A | Audit, gói blind review, automated proxy analysis | A2 chưa có annotation; A3A insufficient coverage. |
| E4-B1/B1a | 264/264 outputs; format gate và post-hoc direction analysis | `E4_B1_PROMPT_ROUTE_REJECTED` trong setup đã thử. |
| E3 / trained fusion / E6 | Chưa training | `HOLD_COMPLEX_FUSION`; chưa có training preflight để kết luận bất khả thi trên K80. |

## Nhánh V4 BLIP-2 — trạng thái đối chiếu 13/11/2026

Kế hoạch chi tiết: [master plan V4](../research/blip2_cxr_v4/plan/VLM_MEDICAL_CLAUDE_CODEX_MASTER_PLAN_V4_2026-10-22.md), [WO phase](../research/blip2_cxr_v4/work_orders/).
Các tên P4/P5 pilot sau đây là tiến độ thực thi, không đồng nhất mục P5 “repository structure” trong master plan.
Đây là pilot BLIP-2 riêng, chưa ghép CLEAR hoặc LLaMA-2 CXR-LLaVA; không thay các quyết định E3/trained fusion/E6 phía trên.

| Phase | Đã làm / trạng thái | Bằng chứng |
|---|---|---|
| P0 | Inventory server/data/env; README V4 còn snapshot cũ P0 | [V4 README](../research/blip2_cxr_v4/README.md) |
| P1 | X00 re-score 3.667 outputs, data/split audit và checkpoint candidate; readiness từng dataset còn PARTIAL | [P1 progress](../research/blip2_cxr_v4/reports/P1_progress.md) |
| P2 | E4 264 outputs re-score, coverage MIMIC, METEOR; X02 v1 FAIL rỗng, v2 smoke 10/10 nonempty, F1 N/A | [P2 progress](../research/blip2_cxr_v4/reports/P2_progress.md) |
| P3 | Paper/asset audit và script/dry-run REIMPLEMENTED; không coi là reproduced; R2 weights và META access/license BLOCKED, legacy env PLAN_ONLY | [P3 progress](../research/blip2_cxr_v4/reports/P3_progress.md) |
| P4 projection | PASS 2.000 train/1 epoch; 500+500 evaluation, 2.000 outputs. BLEU-4 MIMIC 0→0,034638; Indiana 0,003094→0,051517; metric-complete pilot, chưa clinical benefit | [P4 closeout](../research/blip2_cxr_v4/reports/P4_progress.md) |
| P5 Q-Former | Execution COMPLETE: 2.000 updates, 1.000 M3 outputs; chỉ 2 text/cohort, BLEU-4 0,001971/0, historical F1 N/A. HOLD cấu hình | [P5 closeout](../research/blip2_cxr_v4/reports/P5_progress.md) |

Ưu tiên tiếp theo theo MH: (1) chuyên gia review mù lỗi P4/P5; (2) chẩn đoán collapse bằng artifact/đối chứng; (3) thống nhất cohort/policy/coverage và overlap trước metric mới hoặc fusion/LoRA.
Đề xuất [640-token diagnostic](../research/blip2_cxr_v4/reports/P5_recovery_640_20261026/decision.md) mới CPU 9/9, GPU HOLD; không tự chạy từ plan này.
Xem [STATUS — Chưa khớp](../docs/STATUS.md#chưa-khớp) cho số X00, snapshot lịch sử và disk; không sửa report/raw artifact để làm khớp tài liệu trình bày.

## Các việc còn mở để lựa chọn cùng Prof

1. Đánh giá image-grounded trên gói A2 hiện có: cần annotators đủ chuyên môn, rubric/adjudication và coverage đủ. Hiện 0/400 annotation slots được đánh giá độc lập.
2. Thử giả thuyết prompt tiếp trên cohort/protocol mới nếu có lý do khoa học. B1 không chọn được format; không đổi ngưỡng cũ sau khi xem kết quả.
3. Khảo sát CLEAR như bộ gắn cờ bất đồng giữa ảnh và report: cần định nghĩa cờ, tham chiếu đánh giá và false-positive/negative analysis; 42/52 chỉ là khớp report-label proxy.
4. Chỉ xét training adapter/fusion khi có câu hỏi, paired data, đối chứng và resource preflight cụ thể. Chưa có bằng chứng để tự động mở training từ các kết quả hiện tại.

Các mục trên là phương án; nhật ký E/G giữ ước lượng và kết quả chi tiết. Tác vụ mới được thực hiện theo phạm vi người dùng giao; không suy inference/training từ việc đọc plan hay xem catalog.

## Dữ liệu và tài nguyên phải giữ đúng khi thiết kế run

- NIH: 4.999 ảnh; E1 có TEST 872 ảnh đã chấm, E5 dùng TRAIN OOF/DEV. Không dùng test đã xem để tune rồi gọi independent test.
- CheXpert: 518 ảnh / năm findings, locked test; không tune prompt/ngưỡng trên đó.
- Indiana: baseline đã exposed; patient independence và multi-view reference vẫn là giới hạn. B1 loại 20 ảnh E4-R4 nhưng không trở thành external validation độc lập.
- MIMIC-CXR: hạn chế quyền ở snapshot E/G cũ; V4 đã audit và dùng official train/validation với dữ liệu trên server. Validation là dev, chưa loại trừ pretraining contamination; CSV testset repo không phải official test. Full CLEAR concepts chưa đủ assets.
- K80 CXR: FP32, torch 1.12.1+cu102 / Transformers 4.36.2. CLEAR CPU: torch 2.3.1+cpu; dùng environment riêng. Các snapshot tài nguyên không bảo đảm quota hoặc GPU reservation hiện tại.
- Giữ raw states Positive/Negative/Uncertain/Not mentioned/Extraction failure theo protocol; metric phải kèm denominator. Clinical claims cần đánh giá phù hợp trên ảnh.

## Cách làm việc sau khi gom dự án

```bash
# CPU/std-library: đọc trạng thái gốc của mọi run và phase, không chạy thí nghiệm.
python3 -m research.project status
python3 -m research.project status --format json
# Ghi bản catalog mới, không ghi đè file có sẵn.
python3 -m research.project status --output reports/run_catalog_new.md
```

`status` đọc manifest thực tế, kể cả failed/blocked attempts lịch sử và các phase lồng trong bundle. Nó không thay quyết định khoa học hoặc tạo trạng thái cho thí nghiệm chưa có run. Dùng bảng nhật ký E/G để hiểu mốc nào thay thế mốc nào.

Các thuật toán đang dùng giữ trong `research/`; xem [hướng dẫn module](../research/README.md). Dùng lại functions hiện có và config/output directory mới khi thiết kế nghiên cứu mới. Không tạo thêm `*_final_v2_fixed.py` cho cùng chức năng. Tách thuật toán theo nhiệm vụ và dùng một đầu mối quản lý chung.

## Historical plans

Các tài liệu dưới đây đã xóa khỏi cây làm việc; tên được giữ để giải thích các tham chiếu lịch sử trong [nhật ký E/G](../reports/2026-10-17_E_G_execution_status_log.md), không phải protocol hiện hành:

| Tài liệu cũ | Ý chính nhật ký còn dựa vào |
|---|---|
| `CODEX_CXR_CLEAR_MASTER_PLAN_2026-10-13.md`; `CODEX_CXR_CLEAR_MASTER_PLAN_2026-10-13_V2_E1_E2.md` (Master plan V2) | Đặt tên các nhánh E0–E6. E3 so B0 model phát hành, B1 encoder CXR gốc + adapter được train, B2 CLEAR + adapter đối chứng tương ứng; E4 trained fusion là nhánh học tham số, khác E4 prompt pilot. E6A so linear/MLP projector, Q-Former chỉ khi có cơ sở; E6B khảo sát nén visual tokens 1024→256→64 cùng chất lượng, latency và bộ nhớ. Đây là đề xuất chưa chạy cho E3/trained fusion/E6. |
| `CODEX_CXR_CLEAR_V3_POST_E1_E2_EXECUTION_PLAN.md` (V3 plan) | G0 kiểm cache, folds và ngân sách trước E1.3/E5; lần dự báo E5 đầu vượt cap, sau refinement và preflight mới thì E5 đã hoàn tất. |
| `CXR_CLEAR_V4_G1_TO_E4_EXECUTION_PLAN_2026-10-16.md`; `CXR_CLEAR_MASTER_STATUS_V4_1_2026-10-16.md` (V4/V4.1) | G1/G2 chuẩn bị cohort, same-image evidence, evaluator và compute cho E4. Trạng thái `NOT STARTED` ngày 16/10 đã bị các run E4-R4/A/B ngày 17/10 thay thế; không dùng làm trạng thái hiện tại. |
| `CXR_LLaVA_EXP02_Indiana_External_Eval_Roadmap.md`; `CXR_LLaVA_EXP02_Recovery_Reevaluation_Runbook.md`; `CXR_LLaVA_Stratified_Control_Protocol.md`; `K80_Indiana_Full_Cohort_Evaluation_Plan.md`; `K80_Paper_Subset_Evaluation.md`; `K80_Unquantized_Setup.md`; `Manual_Linux_Experiment_Setup.md` | Các roadmap/protocol và hướng dẫn chạy baseline, EXP02, Indiana/K80 trước khi gom dự án; kết quả thực tế phải đọc từ report và raw run, không suy trạng thái từ kế hoạch cũ. |

## Dọn file ngày 18/10

Theo yêu cầu, 14 plan/runbook cũ và 10 script dùng một lần đã xóa khỏi cây làm việc. Archive trung gian cũng đã xóa; không còn lệnh khôi phục archive. Các report nháp/trung gian trùng lặp đã xóa, giữ báo cáo kết quả chính và [nhật ký E/G](../reports/2026-10-17_E_G_execution_status_log.md). Raw run manifests, CSV và kết quả thí nghiệm vẫn nằm trong `research/runs` để đối chiếu. Một số manifest lịch sử ghi SHA/đường dẫn tới file nguồn đã xóa nên tái lập nguyên trạng run cũ từ cây làm việc hiện tại có thể không thực hiện được; không sửa các manifest để che giới hạn này.

`research.project` hiện chỉ làm một việc dùng lại được: đọc trạng thái các run thành Markdown/JSON, không chạy model. Module xử lý artifact/metric còn được import hoặc dùng trong phân tích giữ nguyên. Dung lượng lớn thuộc `research/envs` (~2,2GiB), `research/conda_pkgs` (~1,5GiB) và `research/cache` (~1,2GiB) theo lần đo trước; các thư mục này không thuộc đợt xóa report/script.
