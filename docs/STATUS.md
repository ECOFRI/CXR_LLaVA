# Trạng thái CXR-LLaVA / CLEAR / BLIP-2 / Qwen F0
Cập nhật: 2026-11-18. Bằng chứng mới nhất: F1 30k COMPLETE 18/11/2026; P4/P5 quality artifacts vẫn đến 26/10/2026.
Mục tiêu: baseline sinh báo cáo, phân tích lỗi và kiểm tra CLEAR/connector có giúp CXR-LLaVA không.
Chưa có bằng chứng hệ fusion CLEAR × CXR-LLaVA cải thiện chất lượng lâm sàng; hai nhánh fusion vẫn là đề xuất.
Nguồn trình bày: [MH](report_snapshot/00_MH.md); số đo dưới đây đối chiếu artifact, không chạy lại model.

## Các nhánh
| Nhánh | Trạng thái / kết quả chính | Phân loại và bằng chứng |
|---|---|---|
| Baseline X00 | 3.667/3.667 report Indiana; BLEU1–4 0,28794/0,14676/0,08183/0,044947; ROUGE-L 0,20789; METEOR 0,236041 | Metric-complete local re-score sau bổ sung METEOR; [metrics](../research/blip2_cxr_v4/runs/X00_indiana_rescore_20261022T062016Z/metrics.json), [METEOR](../research/blip2_cxr_v4/runs/X00_indiana_meteor_P2_20261022/metrics.json) |
| E0/E0A/E0B/E-AUDIT | Pipeline baseline và encoder hoạt động; CLEAR concepts chưa đủ assets | Smoke/audit, không phải full clinical eval; [nhật ký E/G](../reports/2026-10-17_E_G_execution_status_log.md) |
| E1/E2 | NIH matched probes và CheXpert direct contrastive có tín hiệu finding; không suy downstream benefit | Exploratory; [E1](../reports/2026-10-14_E1_NIH_exploratory_analysis.md), [E2](../reports/2026-10-13_E0B_E2_direct_contrastive_analysis.md) |
| E1.3/E5 | Frozen concat DEV macro AP 0,2515 < CLEAR 0,2566; 780 fits không phải LLM training | Exploratory; HOLD_COMPLEX_FUSION; [E/G](../reports/2026-10-17_E_G_execution_status_log.md) |
| E4-R4/A/B1 | R4 80/80; B1 264/264 (24 ảnh × 11 arms), không format qua đủ 4 gate; B1a post-hoc không đổi reject | Exploratory; E4_B1_PROMPT_ROUTE_REJECTED; [decision](../research/runs/20261017_E4_B1_evidence_format_gate_v1/E4_B1_research_decision.md) |
| E3/trained fusion/E6 | Chưa triển khai training nhánh CLEAR/CXR tương ứng; HOLD | P4/P5 BLIP-2 riêng không thay trạng thái này; [plan](../plan/PROJECT_PLAN.md) |
| V4 P0 | Kiểm kê server/data/env | Inventory, không inference; [workspace](../research/blip2_cxr_v4/README.md) là snapshot P0 cũ |
| V4 P1 | Re-score X00, audit split MIMIC/VinDr, vị trí NIH/CheXpert, checkpoint candidate | Audit; nhiều readiness còn PARTIAL; [progress](../research/blip2_cxr_v4/reports/P1_progress.md) |
| V4 P2/X02 | E4 re-score, official split coverage, METEOR; X02 v1 10/10 rỗng, v2 10/10 nonempty, BLEU-4=0, F1 N/A | X02 smoke n=10; [v2 metrics](../research/blip2_cxr_v4/runs/X02_smoke_v2_20261022T0810Z/evaluation_v2/metrics.json), [progress](../research/blip2_cxr_v4/reports/P2_progress.md) |
| V4 P3 | Paper/code/asset audit, scripts/dry-runs; không tái lập mọi model | Metadata/REIMPLEMENTED; R2 weights chưa public, META assets/license chưa xác minh; [progress](../research/blip2_cxr_v4/reports/P3_progress.md) |
| V4 P4 | Projection-only BLIP-2: 2.000 train/1 epoch; 500 MIMIC val + 500 Indiana, 2.000 M0/M1 outputs; PASS | Metric-complete pilot theo bộ metric đã thiết lập; [metrics](../research/blip2_cxr_v4/runs/P4_projection_pilot_20261023/P4.3_evaluation_full/metrics.json), [closeout](../research/blip2_cxr_v4/reports/P4_progress.md) |
| V4 P5 | Q-Former/query/projection: 2.000 train, 1.000 M3 outputs; execution COMPLETE, collapse; HOLD cấu hình | Metric-complete matched pilot với F1 N/A, kết quả âm; [decision](../research/blip2_cxr_v4/runs/P5_qformer_20261025/P5_final_decision.json), [metrics](../research/blip2_cxr_v4/runs/P5_qformer_20261025/P5.3_cpu_analysis/metrics.json) |
| F0 CLEAR finding tokens + Qwen3 | 2.500 feature P4; LoRA/bridge batch5, 0,274/0,179 s/mẫu; optional256 batch1 0,567 s/step; GPU tổng19,86 phút | Smoke/technical preflight; 20+20+3 updates, không quality eval; [report](../research/rrg_f0/F0_preflight_report.md) |
| F1 CLEAR tokens + Qwen 30k | COMPLETE 18/11: A2/A3/A4 mỗi arm 29.992 mẫu/959 updates; eval 500 MIMIC val + 500 Indiana; GPU 9,90h/14h | Metric-complete trên dev exposed, không phải test. MIMIC BLEU-4 A2/A3/A4 0,0559/0,0533/0,0636 (M1 0,0346); hist micro-F1 0,724/0,755/0,741; A4−A2 BLEU-4 Δ CI [0,0015; 0,0124]. [report](../reports/2026-11-13_F1_clear_tokens_qwen_30k.md) |

## Kết quả P4/P5 và giới hạn
- P4 BLEU-4 M0→M1: MIMIC 0→0,034638 (Δ CI95% [0,027476; 0,045860]); Indiana 0,003094→0,051517 (Δ [0,043004; 0,054389]); paired bootstrap 1.000 lần.
- P4 historical micro-F1 M0→M1: MIMIC 0,2167→0,4351, coverage 107→353/3.500 class-cases; Indiana 0,1839→0,1045, 79→463/3.500. Không so như cùng denominator; M0 cả 1.000 output chạm 128 token.
- P5 M3 BLEU-4: MIMIC 0,001971; Indiana 0; mỗi cohort chỉ 2 chuỗi khác nhau, cả 1.000 output chạm 128 token. Historical F1 N/A vì 0 nhãn paired definite; không đổi thành 0.
- X00 historical macro-F1 0,4457, 7 classes, 6.243 jointly definite/25.669 class-cases; policy blank_as_negative cho macro-F1 0,1931 trong re-score V4. Không trộn với F1 E4 khác cohort/ontology.
- MIMIC val là dev, Indiana baseline exposed; pretraining overlap chưa loại trừ. BLEU đo trùng từ; labeler là proxy, chưa có chuyên gia đối chiếu ảnh. Không có benchmark tổng quát hoặc clinical validation.

- F0 chỉ đo tài nguyên ở text padding320, target≤256, generation≤128; BCE nhãn giả và minibatch lặp. Giữ preprocess448, disk256/load64. Hai arm sinh8 report/arm, cap128 chạm1/8 (LoRA) và8/8 (bridge); không suy chất lượng hay tốc độ report đủ dài. Full training cần WO riêng.

## Việc tiếp theo (theo 3 mục cuối MH; chưa phải quyền thực thi)
1. Review mù các candidate bỏ sót, khẳng định sai, mâu thuẫn, lặp/cắt của P4/P5 với chuyên gia trên ảnh/report.
2. Chẩn đoán P5 bằng artifact và đối chứng thông tin ảnh, stopping, gradient; đề xuất 640 token trên 40 ca × 2 model có CPU 9/9, chưa chạy GPU: [decision](../research/blip2_cxr_v4/reports/P5_recovery_640_20261026/decision.md).
3. Củng cố cùng cohort/policy/coverage, audit overlap; cân nhắc RadGraph/CheXbert khi env phù hợp, rồi mới đề xuất fusion/Q-Former/LoRA và controls/ngân sách.

## Blocker
- E4-A2 chưa có đánh giá độc lập (0/400 annotation slots); không có clinical adjudication cho P4/P5.
- P5 collapse chưa rõ nguyên nhân; thiếu trace theo bước; 640-token GPU diagnostic còn HOLD.
- Stack K80/driver cũ vẫn là giới hạn của các run lịch sử; ict11 F0 đã xác minh RTX4000 với env CUDA12.4 riêng; full CLEAR concept assets và một số paper weights/license chưa đủ.
- `/storage` báo 100% theo `df -h /storage` ngày 13/11; không tạo output lớn hoặc giả định quota cá nhân.

## Chưa khớp
- MH ghi X00 BLEU-4 0,0450; JSON re-score là 0,0449473824 (4 chữ số thập phân = 0,0449). MH sensitivity 0,2766 không khớp các policy macro-F1 V4: 0,4570/0,3959/0,1931; chưa xác định cùng policy/cohort. Giữ nguyên nguồn, không tự sửa số.
- Snapshot P4 còn đoạn “chưa full cohort”/RUNNING 474 cặp và plan tab ghi P5 chưa chạy/HOLD; artifact cuối cho P4 PASS 1.000 cặp, P5 COMPLETE rồi HOLD khoa học. Đây là các mốc lịch sử, không dùng làm trạng thái mới nhất.
- Google Doc tổng hợp P1/P2/P3 COMPLETE; progress repo vẫn giữ readiness PARTIAL và P3.5–P3.7 HOLD. COMPLETE khâu tài liệu/audit không đồng nghĩa mọi experiment đã chạy.
- P4 raw metrics đếm uncertain bằng `-1` trong khi CSV dùng `-1.0`; coverage uncertain sửa ở [final analysis](../research/blip2_cxr_v4/runs/P4_projection_pilot_20261023/P4.3_final_scientific_analysis.json), raw không đổi; F1/confusion không bị lỗi này.

## Câu hỏi còn mở
Nguồn/policy nào tạo X00 sensitivity 0,2766? Ai đánh giá mù và trên cohort nào?
Nguyên nhân collapse và protocol/ngân sách đối chứng tiếp theo cần quyết định riêng; F0 chỉ đo tài nguyên kiến trúc mới, chưa giải thích P5 hoặc cấp quyền full training.
