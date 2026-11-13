# Nhật ký trạng thái E/G — CXR-LLaVA × CLEAR

**Chốt theo artifact local:** 17/10/2026 (giờ Việt Nam). **Loại tài liệu:** kiểm kê trạng thái và đối chiếu kế hoạch; không phải một experiment mới, không chạy lại model, evaluator hoặc kiểm toán toàn bộ SHA-256. Trạng thái được suy ra từ các manifest, decision và báo cáo đã lưu; `COMPLETE` chỉ nói công việc của gate/run đó hoàn tất, không đồng nghĩa đạt mục tiêu khoa học.

**Bổ sung nội dung bàn giao cho Prof ngày 18/10/2026:** phần “Hồ sơ đầu vào–xử lý–đầu ra” bên dưới trình bày trực tiếp dữ liệu, cách làm, số liệu và giới hạn của từng E/G. Hai bảng E/G đầu file là mục lục trạng thái; các link là nguồn kiểm chứng, không thay thế nội dung. Không có experiment mới trong lần bổ sung này.

## Quy tắc đọc

- **Đã chạy:** có output/decision của công việc tương ứng. **Đã chuẩn bị:** protocol, code hoặc gói review có sẵn nhưng endpoint chính chưa thực hiện. **Chưa chạy:** chỉ có trong plan/đề xuất, không thấy run tương ứng. **Bị chặn/HOLD:** đã kiểm tra hoặc thử nhưng điều kiện đầu ra chưa đạt.
- Các tên `E4` trong plan có **hai nghĩa khác nhau**: pilot đưa bằng chứng vào *prompt* đã chạy; `E4 trained fusion` là nhánh kiến trúc học tham số, **chưa chạy**. `E4-B2` được nhắc trong decision B1 chỉ là đề xuất, không phải run.
- Plan V2/V3/V4 và master status V4.1 là mốc lịch sử. Với trạng thái vận hành, ưu tiên artifact có thời điểm mới hơn. Không diễn giải chữ `NOT STARTED` trong bản status ngày 16/10 thành hiện trạng ngày 17/10.

## Bảng E — thí nghiệm và các nhánh tiếp nối

| ID | Mục tiêu trong plan / phạm vi thực tế | Trạng thái đến 17/10 | Kết quả hoặc giới hạn quyết định | Log gốc |
|---|---|---|---|---|
| Phase 0 / E0 | Baseline sinh báo cáo và inventory | **Đã chạy** | Indiana frontal baseline 3.667/3.667; không xác lập tái lập chính xác paper hoặc độ đúng lâm sàng | [`2026-10-05_k80_indiana_full_indiana_full_k80_20261004_031940_1252388.md`](2026-10-05_k80_indiana_full_indiana_full_k80_20261004_031940_1252388.md), [`CXR_CLEAR_MASTER_STATUS_V4_1`](../plan/PROJECT_PLAN.md#historical-plans) |
| E-AUDIT | Kiểm toán offline baseline CheXpert | **Đã chạy** | Artifact/metric audit hoàn tất; không phải benchmark mới | [`2026-10-13_E_AUDIT.md`](2026-10-13_E_AUDIT.md), [`run_manifest.json`](../research/runs/20261013T191910_E_AUDIT/run_manifest.json) |
| E0A | Smoke/harness hai encoder | **Đã chạy sau một blocker** | CXR K80 1/10/50 PASS; CLEAR preflight trên torch cũ bị chặn do thiếu SDPA, sau đó CLEAR CPU 1 và 10/50 hoàn tất. Không suy ra so sánh chất lượng bệnh | [`../research/runs/20261013T192139_E0A_CXR/run_manifest.json`](../research/runs/20261013T192139_E0A_CXR/run_manifest.json), [`2026-10-13_E0A_CLEAR_CPU_paired_10_50.md`](2026-10-13_E0A_CLEAR_CPU_paired_10_50.md) |
| E0B | CLEAR text/concept feasibility | **Một phần** | Tokenize/encode text và direct contrastive chạy được; full concept bank 368.294 concepts thiếu assets, chưa chạy | [`2026-10-13_E0B_E2_direct_contrastive_analysis.md`](2026-10-13_E0B_E2_direct_contrastive_analysis.md) |
| E1 | Frozen features và matched NIH probes | **Đã chạy** | 4.999 ảnh NIH; CLEAR patch-mean AUROC 0,767117 so với CXR 0,631182 trong exploratory probe. Có rủi ro pretraining overlap; không chứng minh cải thiện sinh báo cáo | [`2026-10-14_E1_NIH_exploratory_analysis.md`](2026-10-14_E1_NIH_exploratory_analysis.md), [`run_manifest.json`](../research/runs/20261014_E1_NIH_probes/run_manifest.json) |
| E1.3 / E13 | CKA/ridge similarity và preflight tài nguyên | **Đã chạy** | Có khác biệt biểu diễn, chưa chứng minh tính bổ trợ dự đoán; E5 profile ban đầu vượt cap rồi được tinh chỉnh kế hoạch | [`E13/results.json`](../research/runs/20261015T005545_V3_E13_RESOURCE_REFINEMENT/E13/results.json), [`E13/run_manifest.json`](../research/runs/20261015T005545_V3_E13_RESOURCE_REFINEMENT/E13/run_manifest.json) |
| E2 | CLEAR direct contrastive trên CheXpert | **Đã chạy** | Lần đầu lỗi mapping `Edema`, lần sau hoàn tất 518 ảnh / 5 findings; test đang khóa, không dùng tune | [`2026-10-13_E0B_E2_direct_contrastive_analysis.md`](2026-10-13_E0B_E2_direct_contrastive_analysis.md), [`run_manifest.json`](../research/runs/20261013T214811_E2_direct_contrastive/run_manifest.json) |
| E5 | Frozen-feature single encoder vs concat | **Đã chạy** | 780 fits, 3 patient-grouped folds, 13 labels; CLEAR AP 0,17879 TRAIN OOF / 0,25657 DEV; concat 0,17719 / 0,25152. Không có uplift AP vững chắc; `HOLD_COMPLEX_FUSION` | [`E5/results.json`](../research/runs/20261015T010818_V3_E5_BASELINE/E5/results.json), [`E5/run_manifest.json`](../research/runs/20261015T010818_V3_E5_BASELINE/E5/run_manifest.json) |
| E4 pilot ban đầu | 20 ảnh × C0/C1/C2a/C3 | **Chạy một phần rồi dừng** | Bản v2 có 1 report thành công, 1 lượt bị watchdog dừng, 78 chưa thử; bản v1 dừng ở preflight. Không lấy trạng thái này làm kết quả cuối | [`E4_decision.md`](../research/runs/20261016_E4_technical_pilot_v2/E4_decision.md), [`preflight_interruption.md`](../research/runs/20261016_E4_technical_pilot_v1/preflight_interruption.md) |
| E4-D1 | Chẩn đoán lại một C1 trong process mới | **Đã chạy** | 1/1 report; PASS kỹ thuật, chưa tự động nhập vào grid lúc đó | [`E4_D1_decision.md`](../research/runs/20261016_E4_D1_single_C1_diagnostic_v1/E4_D1_decision.md) |
| E4-R1 / R2 / R3 | Review memory, sửa guard, lập kế hoạch resume | **Đã chạy phần audit/chuẩn bị** | R1 không xác định chính xác pha gây tăng VRAM; R2 kiểm tra guard; R3 xác nhận 2 report có thể tái sử dụng và 78 lượt còn thiếu. Ba bước này không phải 78 lượt generation | [`R1_decision.md`](../research/runs/20261016_E4_R1_memory_review_v1/R1_decision.md), [`E4_R2_decision.md`](../research/runs/20261016_E4_R2_guard_amendment_v1/E4_R2_decision.md), [`E4_R3_decision.md`](../research/runs/20261016_E4_R3_controlled_resume_plan_v1/E4_R3_decision.md) |
| E4-R4 | Resume bốn nhánh đúng grid | **Đã chạy, PASS kỹ thuật** | 78 report mới + 2 tái sử dụng = 80/80; evaluator PASS, không ghi nhận OOM. Chưa chứng minh CLEAR cải thiện báo cáo | [`E4_R4_decision.md`](../research/runs/20261017_E4_R4_controlled_resume_v1/E4_R4_decision.md), [`2026-10-17_E4_R4_controlled_resume.md`](2026-10-17_E4_R4_controlled_resume.md) |
| E4-A1 | Audit evidence/prompt/output đã lưu | **Đã chạy** | C2a/C3 khác evidence nhưng output giống 17/20; report ngắn/lặp, cơ chế chưa cô lập | [`E4_A1_research_decision.md`](../research/runs/20261017_E4_A1_evidence_utilization_audit_v1/E4_A1_research_decision.md) |
| E4-A2 | Chuẩn bị đánh giá mù có chuyên gia | **Đã chuẩn bị; đánh giá lâm sàng chưa chạy** | Gói 80 reports/400 ô annotation đã tạo; 0/400 ô có annotation độc lập; endpoint lâm sàng `NOT_ESTIMABLE` | [`E4_A2_research_decision.md`](../research/runs/20261017_E4_A2_blinded_paired_report_evaluation_v1/E4_A2_research_decision.md), [`E4_A2_evaluation_results.md`](../research/runs/20261017_E4_A2_blinded_paired_report_evaluation_v1/E4_A2_evaluation_results.md) |
| E4-A3A | Đánh giá tự động proxy, dùng output cũ | **Đã chạy; kết luận inconclusive** | Hash/grid 80/80 PASS; primary paired improved 1, reduced 0, tied 0, không đánh giá được 19/20. Không thay annotation lâm sàng | [`E4_A3A_research_decision.md`](../research/runs/20261017_E4_A3A_automated_proxy_eval_v1/E4_A3A_research_decision.md) |
| E4-B1 | Development evidence-format gate | **Đã chạy; route bị bác bỏ trong setup này** | 24 ảnh × 11 nhánh = 264/264 reports. F1/F2/F3 lần lượt đạt 0/4, 1/4, 2/4 tiêu chí đã khóa; `E4_B1_PROMPT_ROUTE_REJECTED` | [`E4_B1_research_decision.md`](../research/runs/20261017_E4_B1_evidence_format_gate_v1/E4_B1_research_decision.md), [`E4_B1_gate_metrics.csv`](../research/runs/20261017_E4_B1_evidence_format_gate_v1/E4_B1_gate_metrics.csv) |
| E4-B1a | Phân tích hậu nghiệm ô bất đồng, CPU only | **Đã chạy** | Tái dùng nhãn đã lưu; hướng output theo cả matched/shuffled là F1 0/62, F2 2/62, F3 0/62 ô bất đồng. `AUTOMATED PROXY, POST-HOC`; không đổi quyết định B1 | [`E4_B1a_findings.md`](../research/runs/20261017_E4_B1a_posthoc_discordant_analysis_v1/E4_B1a_findings.md), [`E4_B1a_run_manifest.json`](../research/runs/20261017_E4_B1a_posthoc_discordant_analysis_v1/E4_B1a_run_manifest.json) |
| E4-B2 | Cohort lớn/đánh giá tiếp được nhắc trong B1 | **Chưa chạy — đề xuất** | Không thấy run E4-B2; B1 decision không cấp phép tự động. Không chọn được format qua gate B1 | [`E4_B1_research_decision.md`](../research/runs/20261017_E4_B1_evidence_format_gate_v1/E4_B1_research_decision.md) |
| E3 | Thay vision encoder, huấn luyện adapter | **Chưa chạy / HOLD** | E1 chỉ là probe; không có adapter training/control theo plan | [`Master plan V2`](../plan/PROJECT_PLAN.md#historical-plans), [`Master status V4.1`](../plan/PROJECT_PLAN.md#historical-plans) |
| E4 trained fusion | Learned visual/concept fusion | **Chưa chạy / HOLD** | Không đồng nhất với E4 prompt pilot; chưa có bằng chứng downstream để mở nhánh học tham số | [`Master plan V2`](../plan/PROJECT_PLAN.md#historical-plans) |
| E6A / E6B | Projector comparison / visual-token compression | **Chưa chạy / HOLD** | Phụ thuộc adapter training hoặc throughput/VRAM bottleneck đã chứng minh; không thấy run tương ứng | [`Master plan V2`](../plan/PROJECT_PLAN.md#historical-plans) |

## Bảng G — gate và độ sẵn sàng

| ID | Nội dung | Trạng thái đến 17/10 | Quyết định / điểm còn mở | Log gốc |
|---|---|---|---|---|
| G0 (V3) | Kiểm tra resume, cache, fold và ngân sách trước E1.3/E5 | **Đã chạy, có lần BLOCKED ban đầu** | Ước tính E5 ban đầu vượt 7.200 s; sau resource refinement và preflight mới, E5 hoàn tất. Không coi blocker lịch sử còn hiệu lực | [`V3 plan`](../plan/PROJECT_PLAN.md#historical-plans), [`research/experiment_registry.csv`](../research/experiment_registry.csv), [`E5 preflight`](../research/runs/20261015T010818_V3_E5_BASELINE/PREFLIGHT/run_manifest.json) |
| G1 / T0–T5 | Audit E5, dữ liệu Indiana, evidence, interface, compute | **Đã chạy; `HOLD_MORE_EVIDENCE` lúc chốt G1** | Hash E5 PASS; patient grouping/same-image Indiana evidence/allocation chưa đủ lúc đó. Các blocker vận hành sau đó được xử lý từng G2; giới hạn khoa học vẫn còn | [`G1_decision.md`](../research/runs/20261016_G1_E4_readiness_v1/G1_decision.md) |
| G2-A | Triage Indiana view–report/cohort | **Đã chạy** | `GO_TECHNICAL_PILOT_PROPOSAL` cho 20 nhóm; scientific evaluation `HOLD_MORE_EVIDENCE` vì pre-exposure, grouping và view mismatch | [`G2A_decision.md`](../research/runs/20261016_G2A_Indiana_triage_v1/G2A_decision.md) |
| G2-B1 | Preflight checkpoint/schema CLEAR trên 20 ảnh | **Đã chạy** | `GO_G2B2_PROPOSAL`, chưa phải inference tại bước B1 | [`G2B1_decision.md`](../research/runs/20261016_G2B1_CLEAR_preflight_v1/G2B1_decision.md) |
| G2-B2 | CLEAR CPU scoring cùng 20 ảnh | **Đã chạy** | `PASS_G2B2`, đủ 100/100 image–finding scores; chỉ direct contrastive uncalibrated, không phải full concepts | [`G2B2_decision.md`](../research/runs/20261016_G2B2_CLEAR_20_CPU_v1/G2B2_decision.md) |
| G2-C readiness | Cohort/prompt/evaluation/compute protocol | **Đã chạy, ban đầu HOLD** | Same-image evidence và prompt static PASS; evaluator runtime và compute allocation chưa xác nhận trong quyết định ban đầu | [`G2C_decision.md`](../research/runs/20261016_G2C_readiness_v1/G2C_decision.md) |
| G2-C evaluator | Labeler/runtime/adapter integration | **Đã chạy** | `PASS_G2C_EVALUATOR_READY`; clinical correctness vẫn chưa chứng minh | [`G2C evaluator decision`](../research/runs/20261016_G2C_evaluator_runtime_v1/G2C_decision.md) |
| G2-C compute | K80, output allowance, host snapshot | **Đã chạy** | `PASS_G2C_COMPUTE_READY` cho bounded, nonexclusive host snapshot; không xác nhận quota dài hạn hoặc GPU reservation | [`G2C_compute_decision.md`](../research/runs/20261016_G2C_compute_resolution_v1/G2C_compute_decision.md) |
| Scientific/clinical evaluation gate | Đánh giá image-grounded bởi chuyên gia, độc lập bệnh nhân/external | **Chưa đạt** | E4-A2 mới chuẩn bị, 0/400 annotation; automated proxy và reference-report agreement không đủ thay thế | [`E4_A2_evaluation_results.md`](../research/runs/20261017_E4_A2_blinded_paired_report_evaluation_v1/E4_A2_evaluation_results.md), [`E4_A3A_research_decision.md`](../research/runs/20261017_E4_A3A_automated_proxy_eval_v1/E4_A3A_research_decision.md) |

## Dòng thời gian ngắn và kết luận đối chiếu plan

| Ngày | Dấu mốc kiểm chứng được |
|---|---|
| 13–14/10 | E-AUDIT, E0A/E0B, E1, E2: smoke, frozen feature/probe và direct contrastive hoàn tất trong phạm vi đã ghi. |
| 15/10 | G0 resource refinement, E1.3 và E5 frozen-feature benchmark; giữ `HOLD_COMPLEX_FUSION`. |
| 16/10 | G1, G2-A/B/C; E4 ban đầu bị watchdog dừng sau 1 output, D1 1 output; R1/R2/R3 chuẩn bị resume. |
| 17/10 | E4-R4 hoàn thành 80/80; A1, A2 preparation, A3A, B1 264/264 và B1a post-hoc. |

**Theo plan, đã hoàn tất đường kiểm tra khả thi kỹ thuật và các pilot giới hạn.** E4-R4 chứng minh có thể sinh đủ grid; B1 chứng minh có thể chạy kiểm tra format với prompt/`input_ids` được lưu. **Chưa hoàn tất mục tiêu khoa học chính:** không có đánh giá image-grounded độc lập đủ support để kết luận CLEAR giúp báo cáo chính xác hơn; không có format B1 nào qua bốn tiêu chí; E3, trained fusion và E6 chưa chạy. Giữ `HOLD_MORE_EVIDENCE` và `HOLD_COMPLEX_FUSION`. Đây là bản kiểm kê chỉ đọc artifact; không cập nhật registry hay thay đổi quyết định gốc của bất kỳ run nào.

## Cập nhật diễn giải và phương án được nêu ngày 18/10/2026

Phần này đối chiếu đề xuất tổng hợp ngày 18/10 với **artifact đã lưu đến 17/10**. Chỉ tính lại số đếm trên `E4_B1a_clear_reference_proxy.csv`; không inference, không đổi gate hoặc quyết định E4-B1.

| Câu hỏi | Điều artifact thực sự cho thấy | Phân loại và giới hạn |
|---|---|---|
| CLEAR encoder mang tín hiệu finding mạnh hơn trong phép thử E1? | CLEAR patch-mean AUROC **0,767117** so với CXR patch-mean **0,631182** trên NIH exploratory probe. | **AUTOMATED PROXY, exploratory.** Đây là chênh lệch hiệu năng của probe trong thiết lập E1; không phải phép đo lượng thông tin tuyệt đối, và có rủi ro pretraining overlap. |
| Concat hai encoder có ích trong E5? | AP TRAIN OOF: CLEAR **0,17879**, concat **0,17719**; DEV: CLEAR **0,25657**, concat **0,25152**. | **AUTOMATED PROXY.** Không thấy uplift AP của phép concat frozen-feature đã thử; không loại trừ mọi phương pháp fusion. |
| LLM dùng evidence qua prompt theo hướng mong muốn? | B1 không format nào qua gate oracle. B1a: trong **62 ô M–S bất đồng hướng/format**, output theo cả hai hướng là F1 **0/62**, F2 **2/62**, F3 **0/62**. N1 khác C0 về văn bản **19/19 ảnh** cùng tập. | **AUTOMATED PROXY, POST-HOC** cho hướng finding; **VERIFIED** cho thay đổi văn bản. Có phản ứng với prompt, nhưng chưa có bằng chứng sử dụng evidence đúng hướng một cách đáng tin trong setup này. Không kết luận kênh prompt bất khả dụng nói chung. |
| Bảng số gây sụp output? | E4-R4 C2a/C3 sinh cùng văn bản ở **17/20 ảnh** khi sampling; B1 F1-M có output collapse **17/24 ảnh** khi greedy. F2-M/F3-M collapse **0/24** theo định nghĩa B1. | **VERIFIED** về hành vi quan sát được. Quan hệ nhân quả “bảng số gây ra” là **giả thuyết**, vì các prompt/evidence/decoding/cohort giữa R4 và B1 không đồng nhất và F1-O dùng bảng nhãn thay vì cosine. |
| CLEAR evidence đúng với ảnh? | Trong **52 ô** có nhãn Positive/Negative trích từ *report tham chiếu* và CLEAR score, matched khớp **42/52**, shuffled **18/52**; mức nền luôn Negative **29/52**, luôn Positive **23/52**. Tách matched: oracle Positive **18/23**, oracle Negative **24/29**; shuffled tương ứng **3/23** và **15/29**. | **AUTOMATED PROXY** cho khớp report; **UNVERIFIED** cho đúng với ảnh. Các ô được chọn từ cohort dev làm giàu ca dương tính, có report đa view, và chưa được bác sĩ adjudicate. Mức nền 29/52 đã được tính; nó không thay thế đánh giá trên ảnh. |

**Diễn giải được hỗ trợ:** CLEAR có tín hiệu finding trong E1 và direction của nó khớp report tham chiếu thường xuyên hơn hai đối chứng trên 52 ô B1a. Prompt route đã thử không đạt tiêu chí phát sinh báo cáo theo evidence. Chưa chứng minh rằng “thông tin có ở encoder nhưng không truyền được vào LLM” là một cơ chế đã cô lập: khả năng có tín hiệu ở encoder, chất lượng evidence, cách diễn đạt, decoder, độ nhạy với văn bản trung tính và labeler đều có thể góp phần.

| Hướng được nêu để thầy cân nhắc | Tình trạng hiện tại và tính khả thi cần ghi đúng |
|---|---|
| A. F3 trên khoảng 100 ảnh × 5 nhánh để xác nhận kết quả âm | **Chưa chạy; không được tự động mở bởi B1.** 500 lượt × **21,08 giây/lượt** B1 ≈ **2,93 giờ model-call** trên cấu hình bốn K80 đã dùng; **3,5 giờ chỉ là ước lượng** cộng setup/evaluation, chưa gồm bất định tải hệ thống, quota và cohort. F3 đã **trượt 2/4 tiêu chí** B1; ngưỡng mới và cohort mới phải được chốt trước nếu muốn một nghiên cứu khác, không sửa ngược quyết định B1. |
| B. CLEAR làm bộ kiểm tra 3.667 report Indiana có sẵn | **Chưa có pipeline/validation tương ứng.** Tốc độ CLEAR CPU quan sát: 24 ảnh/28,54 giây ở B1 hoặc 4.999 ảnh/5.337 giây ở E1; ngoại suy **~1,09–1,21 giờ chỉ cho bước CLEAR image scoring** 3.667 ảnh, nếu workload tương tự. Chưa tính khớp ảnh–report, trích nhãn 3.667 report, kiểm soát multi-view, ngưỡng, đánh giá false positives/negatives và RSS/I/O. Không gọi đây là “khoảng 1 giờ end-to-end” hoặc bộ kiểm tra đã được xác thực. |
| C. Trained fusion / adapter / Q-Former | **Chưa chạy; `HOLD_COMPLEX_FUSION`.** K80 FP32 và môi trường hiện tại làm nhánh này tốn tài nguyên/khó triển khai, nhưng **chưa có preflight training đủ để khẳng định bất khả thi** hoặc rằng bắt buộc phải có GPU FP16 và driver mới. Quyết định HOLD hiện dựa trước hết vào thiếu bằng chứng downstream và đánh giá độc lập. |
| Bước CPU tính mức nền 42/52 | **Đã làm trong cập nhật này:** luôn Negative **29/52**, luôn Positive **23/52**, shuffled **18/52**; chia theo oracle Positive/Negative như trên. Đây là tính toán hậu nghiệm chỉ đọc, không phải thí nghiệm mới hay kiểm định thống kê. |

**Giữ nguyên:** `E4_B1_PROMPT_ROUTE_REJECTED`, `HOLD_MORE_EVIDENCE`, `HOLD_COMPLEX_FUSION`. Không khởi chạy A/B/C từ bảng đề xuất này.

## Hồ sơ đầu vào–xử lý–đầu ra để tổng hợp báo cáo cho Prof

### 1. Câu hỏi nghiên cứu, mô hình và dữ liệu

**Câu hỏi chính:** bằng chứng thị giác từ CLEAR có giúp CXR-LLaVA sinh báo cáo X-quang ngực ít bỏ sót và ít nêu phát hiện không được ảnh hỗ trợ hơn các đối chứng không? Đầu vào hiện tại là một ảnh X-quang ngực frontal; đầu ra là báo cáo tiếng Anh. Người dùng hiện tại là nhóm nghiên cứu. Chưa có đánh giá lâm sàng hoặc tiếng Việt.

**Hai đường tích hợp cần phân biệt:** (1) frozen features → classifier/probe, đã thử trong E1/E5; (2) CLEAR scores → văn bản evidence → prompt của CXR-LLaVA, đã thử trong E4-R4/B1. Việc đưa CLEAR visual tokens qua adapter học tham số vào LLM chưa thực hiện.

| Thành phần | Đầu vào | Đầu ra đã sử dụng | Vai trò / cấu hình |
|---|---|---|---|
| CXR-LLaVA `ECOFRI/CXR-LLAVA-v2` | Ảnh grayscale 512×512 và prompt | Vector CLS/patch-mean 1024-D khi chỉ khảo sát encoder; report tiếng Anh khi chạy toàn VLM | Base LLaMA2-7B-Chat, vision ViT-L/16 theo mô tả dự án. E4 giữ weights cố định; FP32, không quantization. Revision CXR đã lưu: `b2224786bb90d54b1e1291171866706cfbb44e2b`. |
| CLEAR `peterhan91/CLEAR` | Ảnh RGB 448×448 và cặp text positive/negative của finding | Feature native 768-D; cosine positive/negative và paired softmax cho năm findings | DINOv2 ViT-B/14; CPU FP32. Checkpoint SHA-256 `da1617a7d47970667a7a158eabfa4c4f3798d5d03b7f98caec150d8ee6ab6a03`. Đây là direct contrastive, chưa có full concept pipeline. |
| CheXpert labeler đã lưu | Text report tham chiếu hoặc text report sinh ra | Trạng thái Positive, Negative, Uncertain, Not mentioned; Extraction failure được ghi riêng khi có | Chuyển text thành nhãn proxy để so sánh. Không tự xác nhận phát hiện có thật trên ảnh. Trong E4/B1/B1a không gộp blank/uncertain thành Negative. |

| Dữ liệu / cohort | Quy mô và cách dùng | Nhãn / target | Giới hạn cần ghi trên báo cáo |
|---|---|---|---|
| Indiana/Open-I local | 7.470 ảnh, 3.955 XML; baseline chọn 3.667 ảnh frontal, một ảnh/report | Text FINDINGS + IMPRESSION; nhãn trích từ text | Cohort baseline đã được xem/đánh giá; patient independence chưa xác minh. Report có thể nói nhiều view. |
| NIH local, E1 | 4.999 ảnh: TRAIN 3.783/1.222 patients, VAL 344/57, TEST 872/56 | 13 nhãn bệnh sau loại Hernia theo support rule | Không có overlap ID giữa splits theo audit; có nguy cơ overlap nguồn pretraining. Đây là exploratory subset, không independent external validation. |
| NIH TRAIN/DEV, E1.3/E5 | Dùng lại cache E1; E1.3 trên 3.783 TRAIN; E5 dùng TRAIN OOF và 344 VAL như DEV | Các nhãn NIH, không có report-generation target | Không đánh giá lại 872 test để chọn fusion. VAL ở đây là DEV, không phải test độc lập. |
| CheXpert E2 | 518 ảnh frontal / 500 patient-key clusters | Năm findings: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion | Locked test; không tune prompt/ngưỡng. Inventory 668 file không phải mẫu số E2. |
| Indiana E4-R4 | 20 reportUID groups frontal-only theo source metadata | 100 ô reference = 20×5; chỉ 4 reference positives, đều Cardiomegaly | Technical pilot; không độc lập với exposure baseline. “Frontal-only” metadata không bảo đảm patient independence. |
| Indiana E4-B1 | 24 ảnh dev khác 20 ảnh E4-R4; không dùng CheXpert locked | Sáu finding labels từ reference report; positive support 4/5/4/4/6/4 theo thứ tự Atelectasis/Cardiomegaly/Consolidation/Edema/Effusion/Pneumothorax | Làm giàu ca dương tính, không đại diện prevalence tự nhiên. CLEAR chỉ có score cho năm findings, thiếu Pneumothorax. Không gọi cohort này là external test mới. |
| PadChest / MIMIC-CXR | PadChest inventory 24 ảnh, chưa là cohort chính; MIMIC không có quyền truy cập theo status | Không có kết quả E/G mới từ hai nguồn này | Không đưa quy mô hai nguồn này vào mẫu số kết quả hiện tại. |

License/quyền sử dụng và ẩn danh chưa được kiểm toán toàn diện. Các bảng sau chỉ trình bày số liệu tổng hợp; không đính kèm ảnh hoặc report bệnh nhân.

### 2. E0, E-AUDIT, E0A, E0B — xây nền và kiểm tra pipeline

| Mốc | Đầu vào thực tế | Xử lý đã làm | Đầu ra / số đo | Ý nghĩa và đầu vào chuyển tiếp |
|---|---|---|---|---|
| Phase 0 / E0 | 3.955 XML + 7.470 ảnh Indiana; nhãn view NLM; CXR checkpoint phát hành | Chọn một frontal/report, sinh report FP32 trên bốn K80; label reference và prediction | 3.667/3.667 reports; macro F1 **0,4457** trên 7 findings có F1 xác định theo joint-definite policy; mean latency **22,70 s/report**. Coverage khác nhau lớn theo finding | Baseline kỹ thuật và kho text/label để audit, triage và chọn dev. F1 này không so trực tiếp với E1, E2 hoặc E4 vì task, finding set và mẫu số khác. |
| E-AUDIT | 518 ảnh CheXpert ×14 câu trả lời baseline đã lưu, GT image/study và cache parser | Tính lại metric offline; đối chiếu parser-cache, GT; phân loại output không parse được | **6.950/7.252** câu trả lời parse được, **302/7.252** thiếu; mismatch parser-cache **0**, GT mismatch **0**. Five-finding F1 parsed-only **0,248720**, missing-as-negative sensitivity **0,247987** | Không thấy lỗi nối GT/cache để giải thích hiệu năng thấp. Đây là audit câu trả lời Yes/No lịch sử, không phải sinh report mới. Missing-as-negative chỉ là sensitivity policy của audit này, không truyền sang B1/B1a. |
| E0A CXR | Frozen 50 ảnh CheXpert dùng cho smoke; vision/projector weights | Encoder-only gates 1→10→50; kiểm tensor finite, shape, repeatability, native/projected outputs | CXR CLS và patch-mean 1024-D; native patches `[1,1024,1024]`, projected patches `[1,1024,4096]`; gates PASS. Peak allocated khoảng **1,340 GiB**, mean forward **0,533 s/ảnh** | Xác nhận harness trích feature. E1 về sau chỉ giữ CLS/patch-mean gọn; không suy cache E1 có đầy đủ spatial sequence. |
| E0A CLEAR | Cùng nguồn ảnh smoke; checkpoint CLEAR | Preflight torch cũ thất bại vì thiếu SDPA; sau đó môi trường CPU torch 2.3.1 chạy 1 ảnh và paired 10/50 | CLEAR feature 768-D; completed CPU smoke, repeat/numeric checks theo báo cáo paired | Đóng blocker tương thích bằng môi trường riêng; không sửa environment K80. Timing CPU và GPU khác hardware nên không xếp hạng tốc độ encoder. |
| E0B | 10 text prompts = positive/negative cho năm findings; ảnh smoke/cache | Official tokenize/encode_text; paired cosine/softmax; repeat gates 1 và 10 | Text tensor **10×768**, tokenizer **10×77**, normalized finite; feature/score repeat delta **0** ở gates đã lưu | Cấp đường direct contrastive cho E2, G2-B2, B1. Full 368.294 concepts và các assets liên quan vẫn thiếu; không gọi text finding prompts là concept bank đầy đủ. |

Baseline E0 ở trên là kết quả reconstructed-cohort Indiana báo cáo ngày 05/10, không phải một lần chạy mới ngày 13–17/10. Các lần smoke/harness chỉ là kiểm tra kỹ thuật, không cộng vào cohort đánh giá chất lượng.

Nguồn bổ sung: [baseline Indiana](2026-10-05_k80_indiana_full_indiana_full_k80_20261004_031940_1252388.md), [E-AUDIT](2026-10-13_E_AUDIT.md), [E0A](../research/runs/20261013T192139_E0A_CXR/run_manifest.json), [CLEAR CPU paired](2026-10-13_E0A_CLEAR_CPU_paired_10_50.md), [E0B](2026-10-13_E0B_E2_direct_contrastive_analysis.md).

### 3. E1 → E1.3 → E5 và E2 — tín hiệu ở encoder có chuyển thành lợi ích không?

**E1 — đầu vào:** 4.999 NIH images, source labels và bốn feature families: CXR CLS, CXR patch-mean, CLEAR CLS, CLEAR patch-mean. **Cách làm:** frozen encoders; train-only StandardScaler; logistic regression L2 với C ∈ {0,001; 0,01; 0,1; 1}; chọn C theo VAL AP và threshold theo VAL F1; đóng băng rồi mở TEST một lần. **Đầu ra:** cache features, fitted probes, lựa chọn C/ngưỡng, predictions và metric cùng 13 findings trên TEST 872 ảnh. Không training VLM.

| E1, cùng TEST và finding set | Macro AUROC | Macro AP | Macro F1 |
|---|---:|---:|---:|
| CXR CLS | 0,630225 | 0,148528 | 0,189721 |
| CXR patch-mean | 0,631182 | 0,154900 | 0,173991 |
| CLEAR CLS | 0,761800 | 0,250440 | 0,290319 |
| CLEAR patch-mean | 0,767117 | 0,264146 | 0,305094 |

**E1.3 — đầu vào:** tái dùng bốn cache trên 3.783 TRAIN images / 1.222 patients. **Cách làm:** linear CKA và ridge cross-prediction theo ba patient-grouped OOF folds, không test evaluation. **Đầu ra:** CKA và R² hai chiều. CKA đo tương đồng hình học; R² đo khả năng dự đoán feature của encoder kia, không phải hiệu năng chẩn đoán.

| E1.3 representation | CKA, toàn TRAIN | CKA, một ảnh/patient | Ridge R² CXR→CLEAR, OOF | Ridge R² CLEAR→CXR, OOF |
|---|---:|---:|---:|---:|
| CLS | 0,401838 | 0,326174 | 0,332632 | 0,471733 |
| Patch-mean | 0,233407 | 0,180874 | 0,375362 | 0,592522 |

**E5 — đầu vào:** TRAIN và DEV patch-mean cache; A=CXR 1024-D, B=CLEAR 768-D, A+B=concat 1792-D. **Cách làm:** ba outer folds grouped theo patient, inner split để chọn C/ngưỡng, scale từng encoder block trong training fold; 780 fits; pooled TRAIN OOF là endpoint chính, DEV chấm một lần sau freeze. **Đầu ra:** OOF/DEV predictions và so sánh ba nhánh. Đây là huấn luyện logistic heads trên feature cố định, không backprop qua encoder/LLM.

| E5 partition / arm | Macro AP — endpoint chính | Macro AUROC | Macro F1 |
|---|---:|---:|---:|
| TRAIN OOF — CXR | 0,079991 | 0,606881 | 0,125048 |
| TRAIN OOF — CLEAR | 0,178787 | 0,709392 | 0,251887 |
| TRAIN OOF — concat | 0,177186 | 0,728990 | 0,250796 |
| DEV — CXR | 0,147303 | 0,639778 | 0,128361 |
| DEV — CLEAR | 0,256568 | 0,748612 | 0,243131 |
| DEV — concat | 0,251516 | 0,738461 | 0,250574 |

**Cách diễn giải:** E1 thấy CLEAR tốt hơn trong matched probe. E1.3 thấy biểu diễn khác nhau nhưng không chứng minh bổ trợ hữu ích. E5 concat không hơn CLEAR trên AP, dù một vài metric phụ cao hơn; không được bỏ AP để chọn kết quả thuận lợi. E1 và E5 không dùng cùng partition/protocol nên không so chéo các F1/AP như một ablation trực tiếp. Kết quả E5 dẫn đến HOLD complex fusion; không chứng minh mọi kiến trúc fusion đều vô ích.

**E2 — đầu vào:** 518 frontal CheXpert images / 500 patient-key clusters, labels của năm findings, CLEAR checkpoint và 10 fixed text prompts. **Cách làm:** normalized paired cosine → unscaled softmax; ranking là chính, threshold 0,5 chỉ exploratory; 50 image features tái sử dụng, 468 image forwards mới. **Đầu ra:** 518×768 global features và 518×5 scores, ranking metrics và đối chiếu với CXR Yes/No cache. Macro AUROC **0,892814**, AP **0,620352**, F1@0,5 **0,519722**.

| E2 finding | AUROC CLEAR | AP CLEAR | F1 CLEAR @0,5 | F1 CXR parsed-only |
|---|---:|---:|---:|---:|
| Atelectasis | 0,857648 | 0,658237 | 0,711485 | 0,527536 |
| Cardiomegaly | 0,894552 | 0,773596 | 0,730878 | 0,240000 |
| Consolidation | 0,882589 | 0,330906 | 0,197952 | 0,139535 |
| Edema | 0,899801 | 0,587351 | 0,453488 | 0,070796 |
| Pleural Effusion | 0,929482 | 0,751669 | 0,504808 | 0,265734 |

CXR parsed-only và CLEAR full-cohort khác mask. So sánh cùng CXR parsed mask: CLEAR macro F1 **0,521036** so với CXR **0,248720**. Đây vẫn là so sánh hai pipeline với cách sinh quyết định khác nhau; không kết luận encoder causation hoặc clinical superiority. Paired softmax chưa calibrated; không gọi score đó là xác suất bệnh. Lần E2 đầu tiên lỗi mapping `Edema`, lần sau hoàn tất; không có tuning trên locked test.

Nguồn: [E1 report](2026-10-14_E1_NIH_exploratory_analysis.md), [E13 results](../research/runs/20261015T005545_V3_E13_RESOURCE_REFINEMENT/E13/results.json), [E5 results](../research/runs/20261015T010818_V3_E5_BASELINE/E5/results.json), [E5 protocol](../research/runs/20261015T010818_V3_E5_BASELINE/E5/e5_protocol.yaml), [E2 report](2026-10-13_E0B_E2_direct_contrastive_analysis.md).

### 4. G0/G1/G2 — các gate nhận gì và bàn giao gì?

| Gate | Đầu vào cần kiểm tra | Xử lý / kiểm tra | Đầu ra cụ thể | Chuyển tiếp và phần còn thiếu |
|---|---|---|---|---|
| G0 | Cache E1, labels, patient splits, config E1.3/E5, caps CPU | Check alignment/leakage, thử timing nhỏ và dự báo workload | Preflight, fold/resource estimates; ước tính E5 đầu tiên **11.483 s >7.200 s**, nên BLOCKED trước full fit | Resource refinement và preflight mới mở E1.3/E5. Blocker lịch sử không có nghĩa E5 chưa chạy. |
| G1-T0 | E5 bundle đã lưu | Đối chiếu manifest, output hashes và trạng thái fits | 21 output hashes + report hash khớp, 780 fits được xác nhận | Xác lập E5 hoàn tất, giữ HOLD complex fusion. |
| G1-T1 / G1.1 | Indiana inventory, XML linkage, baseline cohort | Kiểm report/image identity, paths, grouping/provenance | 7.470 paths, 3.667 baseline pairs; technical linkage PASS | Patient/study độc lập chưa xác minh; reportUID chỉ là đơn vị nhóm kỹ thuật. Cần G2-A triage. |
| G1-T2 / G1.2 | E2 scores 518×5 và Indiana cohort | Đối chiếu nguồn ảnh của evidence | Không có overlap Indiana–CheXpert trong các score đã lưu | Không dùng E2 scores làm evidence cho ảnh Indiana; cần CLEAR scoring cùng ảnh tại G2-B. |
| G1-T3 / G1.3 | Baseline texts, generation interface, nguồn code | Kiểm khả năng đưa text evidence vào chat và giữ C0/control | Interface `generate_cxr_repsonse(chat,image,...)` hỗ trợ text theo static inspection | Chưa phải runtime test hoặc bằng chứng attention/usefulness. |
| G1-T4 / G1.4 | Log K80, memory/time baseline, nguồn lực thấy được | Ước tính 20×4 pilot và audit allocation | Resource proposal; allocation/quota còn HOLD lúc G1 | G2-C giải quyết readiness có giới hạn. |
| G1-T5 | Các đầu ra T0–T4 | Tổng hợp quyết định | `HOLD_MORE_EVIDENCE`; G1 hoàn tất | Gate đã làm xong nhưng E4 chưa sẵn sàng ở thời điểm đó. |
| G2-A | 3.955 report groups + view labels, 3.667 baseline pairs | Triage view; kiểm mẫu 40 multiview +10 frontal-only XML; chọn 20 nhóm theo metadata/hash | Toàn inventory: **301 frontal-only /3.384 frontal+lateral verified /270 other**. Trong baseline: **290 /3.372 /5**. Manifest 3.955 rows, 20 pilot flags | Chọn 20 frontal-only groups cho technical pilot; không chọn theo outcome. Còn pre-exposure và patient grouping chưa verified. |
| G2-B1 | 20 candidate identities, checkpoint/preprocess/text schema | Static integrity và CPU inference protocol | Frozen scoring schema, hash identities, bounded execution proposal | `GO_G2B2_PROPOSAL`; B1 chưa sinh score. |
| G2-B2 | Đúng 20 frontal images của G2-A + CLEAR +10 text prompts | CPU FP32, batch1, bốn threads; 21 image forwards gồm một repeat; hai text forwards | **100/100 rows =20×5**, positive/negative cosine + paired softmax; repeat delta0, PASS. Guard wall **57,51 s**, RSS khoảng **1,39 GiB** | Same-image evidence có thật để render E4; chưa nói nó đúng lâm sàng. |
| G2-C readiness | 20 images +100 score rows + baseline chat | Render bốn nhánh, derangement, kiểm context, chuẩn bị evaluator | **80 frozen chats**; C1/C2a/C3 **498 tokens**; expanded input +512 output =**2.033 ≤4.096**. Frozen control/prompt protocol | Runtime evaluator và allocation còn HOLD ở bước đầu. |
| G2-C evaluator | Pinned labeler/NLP assets + bridge/adapter cho 4 arms×5 findings | Chạy actual sample và 100 synthetic texts; fault/schema checks | 12/12 tests PASS, **500/500 controlled label-state checks**; adapter đủ 80 reports/100 references/400 predictions | Đóng runtime blocker, không chứng minh semantic/clinical correctness. |
| G2-C compute | Host access, GPU visibility, nguồn lực/allowance được cho phép | Kiểm bốn K80 đủ VRAM, workloads và bounded output write | `PASS_G2C_COMPUTE_READY`, allowance kiểm tới64MiB, snapshot không độc quyền | Chỉ sẵn sàng cho run giới hạn; không bảo đảm future quota/reservation. |

Trong G2-A, “baseline có ít nhất một lateral” là **3.376**, còn “frontal+lateral verified, không unknown” là **3.372**. Hai số này dùng định nghĩa khác nhau, không phải lỗi đếm. G2-A đã làm rõ mâu thuẫn bề ngoài đó. Các source decisions trong bảng G đầu file lưu lịch sử HOLD/PASS theo từng bước; không phải tất cả HOLD lịch sử còn là blocker vận hành hiện tại.

Nguồn bổ sung: [G2-A triage](../research/runs/20261016_G2A_Indiana_triage_v1/G2A_cohort_triage.md), [G2-B2 analysis](../research/runs/20261016_G2B2_CLEAR_20_CPU_v1/G2B2_analysis_report.md). Các G1/G2 decision còn lại có link ở bảng trạng thái đầu file.

### 5. E4 pilot → D1/R1–R4 → A1/A2/A3A — từ đầu ra kỹ thuật tới giới hạn đánh giá

**Đầu vào chung E4:** 20 frontal images G2-A, 100 CLEAR score rows G2-B2, reference reports và 80 chats đã khóa. Mỗi ảnh được dùng nguyên vẹn ở bốn nhánh. Weights, preprocess và decoding giữ cố định; sampling temperature0,2, top_p0,8, top_k50, max_new_tokens512, FP32 trên bốn K80.

| Nhánh E4 | Evidence vào prompt | Mục đích đối chứng |
|---|---|---|
| C0 | Prompt gốc, không evidence bổ sung | Baseline trong cùng run/config. |
| C1 | Context trung tính/padding theo protocol E4 cũ, token length khớp C2a/C3 | Tác động do thêm context/độ dài; khác N1 natural-language của B1 sau này. |
| C2a | Bảng số CLEAR đúng ảnh | Nhánh có evidence dự kiến hữu ích. |
| C3 | Bảng số lấy nguyên từ ảnh khác theo derangement | Kiểm tra output có phụ thuộc evidence đặc hiệu theo ảnh hay chỉ bị prompt làm thay đổi. |

| Mốc vận hành | Nhận đầu vào | Đầu ra / sự kiện | Kết luận đúng phạm vi |
|---|---|---|---|
| E4 v1/v2 | Protocol/chats và model | v1 dừng preflight; v2 thành công một C0, C1 bị watchdog dừng, 78 cấu hình chưa thử | PARTIAL; watchdog interruption không tự đồng nghĩa CUDA OOM. |
| R1 | Logs/guard của v2 | Phân tích ngưỡng device-wide; thiếu telemetry theo token/phase | Không cô lập nguyên nhân memory spike; cần đo tốt hơn. |
| R2 | R1 + guard amendment được duyệt | Guard `min(10.624 MiB,total−768 MiB)`, free floor768MiB; 30 logic checks | PASS cấu hình guard, chưa phải generation PASS. |
| D1 | C1 cũ trong process mới + guard sửa | Một attempt, một C1 hoàn tất, không OOM quan sát | Xác nhận cấu hình chạy được một case; report chưa tự nhập grid khi D1 kết thúc. |
| R3 | Original C0 +D1 C1 + frozen grid | Xác nhận hai output tái dùng và 78 cấu hình còn thiếu; 303 planning/integrity checks | Hoàn tất resume proposal. |
| R4 | Hai outputs hợp lệ +78 cấu hình thiếu | **78/78 new reports +2 reused =80/80**, không missing key; 81 cumulative attempts gồm một aborted trước đó; evaluator PASS | `PASS_E4_TECHNICAL_PILOT`; không có retry hoặc claim cải thiện lâm sàng. |

**Đầu ra đánh giá R4:** 80 report texts → 400 prediction finding states; 20 references →100 reference states. Chỉ có bốn reference positives nên coverage/đếm hỗ trợ quan trọng hơn nhìn macro F1 đơn lẻ.

| Arm | Macro F1 có điều kiện joint-definite | Joint-definite cells /100 | Findings có F1 xác định /5 | Unknown prediction cells /100 | Median report chars |
|---|---:|---:|---:|---:|---:|
| C0 | 0,1111 | 20/100 | 3/5 | 52/100 | 225 |
| C1 | 0,5000 | 11/100 | 1/5 | 77/100 | 143 |
| C2a | 0,3333 | 4/100 | 2/5 | 93/100 | 33 |
| C3 | 0,5000 | 2/100 | 2/5 | 95/100 | 33 |

Không xếp hạng F1 ở bảng này như cùng support. C2a/C3 text giống nhau **17/20** dù evidence khác trên **20/20**. C2a có **92 blank +1 uncertain**; không coi 93 ô này là âm tính hoặc đều là hallucination. Đây là lý do pipeline chạy được nhưng scientific endpoint còn inconclusive.

| Phân tích sau R4 | Đầu vào tái sử dụng | Đầu ra phân tích | Điều chưa làm |
|---|---|---|---|
| A1 | 80 texts, source code, prompt/score hashes, 400 prediction states | Evidence content đã vào đường prompt là credible theo hashes/code; C2a/C3 equality17/20; review độ dài/coverage/parser | Không có in-call token/embedding capture của R4; không chứng minh attention hoặc cơ chế không dùng evidence. |
| A2 | 20 images +80 frozen reports | Gói đánh giá mù, opaque IDs, rubric, 400 annotation slots và code phân tích | **0/400** annotation độc lập; không có clinical paired endpoint. Chỉ chuẩn bị, chưa phải đánh giá lâm sàng đã xong. |
| A3A | Cached label states và toàn bộ frozen grid/provenance | Hash checks PASS; paired endpoint: improved1, reduced0, tied0, non-evaluable19/20. `INCONCLUSIVE_INSUFFICIENT_COVERAGE` | Không generation lại, không bổ sung nhãn của bác sĩ. |

Nguồn bổ sung: [R4 four-arm comparison](../research/runs/20261017_E4_R4_controlled_resume_v1/E4_R4_four_arm_comparison.md). Các report/decision D1/R1–R4/A1–A3A được liên kết trong bảng E đầu file.

### 6. E4-B1 — kiểm tra định dạng evidence có đối chứng oracle

**Đầu vào:** 24 Indiana dev images disjoint với20 E4-R4; six reference-label states/image; CLEAR năm finding scores/image; model giữ nguyên. Chọn cohort seed2026101701, shuffled donor seed2026101702. Có27 ô oracle Positive trên24×6=144 ô; Pneumothorax có4 positives nhưng không có CLEAR score.

**Xử lý:** CLEAR CPU tạo120 score rows; render và lưu264 prompts/`input_ids`; greedy `do_sample=False`, max512 tokens, batch1 FP32 trên4 K80. Smoke26 calls được lưu riêng; full grid264 calls gồm11 arms. Một interruption có chủ đích để tách background script khỏi phiên Codex, sau đó resume giữ30 report đã xong. Không cộng smoke vào kết quả gate.

| Ký hiệu | Nội dung đưa vào | Ý nghĩa |
|---|---|---|
| C0 | Base request viết report có sáu findings | Baseline. |
| N1 | Đoạn văn trung tính tự nhiên, không chứa tên findings, không padding bằng `x` | Đo tác động của việc thêm lời chỉ dẫn; không khẳng định token-matched chính xác từng case. |
| F1 | Bảng số cosine/paired score với M/S; bảng trạng thái nhãn với O | Giữ format numeric kiểu R4 ở M/S. O khác schema numeric, nên hạn chế causal isolation. |
| F2 | Câu văn tự nhiên cho từng finding | Kiểm tra model hiểu evidence qua prose. |
| F3 | Câu văn evidence kèm yêu cầu nói rõ từng finding | Kiểm tra thêm hướng dẫn về reporting. |
| M / S / O | M=matched đúng ảnh; S=donor khác ảnh; O=nhãn từ report tham chiếu | O là oracle text diagnostic, không phải nhãn đúng tuyệt đối của ảnh. |

**Đầu ra:**264/264 successful report records;120 CLEAR score rows;264 label rows =1.584 finding states, không extraction failure. Mỗi record lưu prompt đã render, input IDs, output và config/provenance. C0 greedy repeat byte-identical ở smoke. Tổng CXR wall windows **6.563,21 giây** gồm smoke/load, dưới cap14.400 giây; CLEAR24-image CPU wall **28,54 giây**.

**Định nghĩa:** collapse nếu output <60 ký tự hoặc đúng một trong hai generic-normal strings đã khóa. Mention count tính các trạng thái Positive/Negative/Uncertain, không tính Not mentioned/Extraction failure. Oracle match chỉ dùng69 ô có reference Positive/Negative. Sensitivity M–S dùng19 ảnh có ít nhất một CLEAR direction khác nhau; numerator là ảnh đổi ít nhất một trong sáu label states, chưa xét đúng hướng.

| Arm, mỗi arm24 reports | Collapse /24 | Mention cells /144 | Mean findings/report /6 | Median chars | Median output tokens |
|---|---:|---:|---:|---:|---:|
| C0 | 0/24 | 71/144 | 2,958 | 218 | 60 |
| N1 | 6/24 | 38/144 | 1,583 | 140,5 | 41 |
| F1-M | 17/24 | 14/144 | 0,583 | 34 | 12 |
| F1-S | 16/24 | 19/144 | 0,792 | 34 | 12 |
| F1-O | 10/24 | 38/144 | 1,583 | 129 | 34,5 |
| F2-M | 0/24 | 58/144 | 2,417 | 171,5 | 49 |
| F2-S | 0/24 | 57/144 | 2,375 | 187 | 50 |
| F2-O | 7/24 | 29/144 | 1,208 | 97,5 | 27 |
| F3-M | 0/24 | 64/144 | 2,667 | 235 | 61 |
| F3-S | 0/24 | 73/144 | 3,042 | 221,5 | 61 |
| F3-O | 0/24 | 69/144 | 2,875 | 258,5 | 68 |

| Tiêu chí khóa trước inference | F1 | F2 | F3 |
|---|---|---|---|
| M và O đều collapse≤20% | M17/24, O10/24: **FAIL** | M0/24, O7/24: **FAIL** | M0/24, O0/24: **PASS** |
| Mean mention M≥4/6 | 0,583: **FAIL** | 2,417: **FAIL** | 2,667: **FAIL** |
| O hơn S≥20 điểm phần trăm, ít nhất20 eligible cells | O15/69 vs S8/69; +10,1 điểm: **FAIL** | O10/69 vs S31/69; −30,4 điểm: **FAIL** | O33/69 vs S36/69; −4,3 điểm: **FAIL** |
| M/S sensitivity≥30%, ít nhất10 eligible images | 3/19: **FAIL** | 11/19: **PASS** | 13/19: **PASS** |
| Toàn bộ4 tiêu chí | **FAIL, 0/4** | **FAIL, 1/4** | **FAIL, 2/4** |

F1-O **có hơn S về số đếm trên69 ô**, nhưng mức hơn không đạt20 điểm. Vì vậy phát biểu “oracle không vượt shuffled ở mọi so sánh” là sai; đúng là **không format nào đạt oracle gate**, và trong subset bất đồng34 ô ở B1a thì O không hơn S. Các kết quả giữ nguyên quyết định `E4_B1_PROMPT_ROUTE_REJECTED` **trong setup đã thử**.

Nguồn: [B1 protocol](../research/runs/20261017_E4_B1_evidence_format_gate_v1/E4_B1_protocol.md), [B1 report](2026-10-17_E4_B1_evidence_format_gate.md), [B1a arm summary](../research/runs/20261017_E4_B1a_posthoc_discordant_analysis_v1/E4_B1a_arm_summary.csv). Median length ở B1a lấy từ bảng output đã lưu; R4/B1 là hai cohort và decoding khác nhau, không xem thay đổi median như một causal ablation duy nhất.

### 7. E4-B1a — đầu vào cũ, câu hỏi hậu nghiệm mới

**Đầu vào:** protocol/cohort/scores/prompts/reports và cached CheXpert states của B1, với SHA-256 đã đối chiếu theo manifest khi B1a chạy. **Xử lý:** CPU-only; chọn ô bất đồng O–S và M–S, phân tầng oracle state/finding, đối chiếu C0/N1, kiểm block oracle, giữ đủ năm trạng thái label. **Đầu ra:** cell-level CSVs, direction categories, arm lengths/mentions, oracle inventory, manifest và findings. Không generation/labeler inference lại; không đổi ngưỡng B1. Toàn bộ diễn giải nhãn là **AUTOMATED PROXY, POST-HOC**; không p-value/CI.

Các mẫu số khác nhau: **144** ô tổng/arm (24×6), **69** oracle definite trên6 findings, **52** oracle definite trên5 CLEAR findings, **34** O–S direction-discordant cells/format, **62** M–S direction-discordant cells/format, **27** oracle-positive cells trên6 findings, **19** ảnh có ít nhất một M/S direction khác. Không hoán đổi các mẫu số này.

| Trên đúng34 ô O–S bất đồng | F1 | F2 | F3 |
|---|---:|---:|---:|
| C0 output khớp oracle | 17/34 | 17/34 | 17/34 |
| S output khớp oracle | 6/34 | 17/34 | 15/34 |
| O output khớp oracle | 6/34 | 7/34 | 12/34 |
| S output theo shuffled direction | 2/34 | 3/34 | 5/34 |
| O và S giữ nguyên extracted state | 23/34 | 15/34 | 20/34 |
| O output Not mentioned | 24/34 | 26/34 | 19/34 |
| S output Not mentioned | 25/34 | 14/34 | 13/34 |

| Trên cùng19 ảnh có M/S direction khác | N1–C0 (mức nền) | F1 M–S | F2 M–S | F3 M–S |
|---|---:|---:|---:|---:|
| Text thay đổi | 19/19 | 4/19 | 17/19 | 15/19 |
| Có ít nhất một state đổi trong6 findings | 19/19 | 3/19 | 11/19 | 13/19 |
| State thay đổi trên5 CLEAR findings,19×5 ô | 36/95 | 6/95 | 24/95 | 32/95 |
| Theo cả hai direction tương ứng trên62 ô M/S bất đồng | Không áp dụng: N1 không có direction | 0/62 | 2/62 | 0/62 |

Không lấy hiệu giữa sensitivity và N1 như một phép hiệu chỉnh noise đã calibrated. Chỉ thấy context trung tính cũng có thể đổi text/state mạnh; thay đổi output chưa chứng minh evidence được dùng đúng hướng. “Theo cả hai hướng” là tiêu chí chặt; không đánh đồng mọi trường hợp còn lại với “không dùng evidence”, vì còn partial/uncertain/not-mentioned transitions.

| Nhánh | Output Positive /27 oracle-positive cells |
|---|---:|
| C0 / N1 | 13/27 · 8/27 |
| F1-M / F1-S / F1-O | 6/27 · 6/27 · 8/27 |
| F2-M / F2-S / F2-O | 13/27 · 9/27 · 8/27 |
| F3-M / F3-S / F3-O | 5/27 · 8/27 · 5/27 |

F2-O collapse ở7/24 ảnh. Oracle blocks của7 ảnh này vẫn có **13 Positive +8 Negative +3 Uncertain +18 Not mentioned =42 câu**, nhưng output chỉ mention **2/42** finding slots. 17 ảnh không collapse có **14+34+4+50=102 câu**, output mention **27/102**. Đây là association trong output đã lưu, chưa chứng minh nguyên nhân collapse.

Đối chiếu trực tiếp CLEAR với reference-report labels: matched **42/52**, shuffled **18/52**, luôn Negative **29/52**, luôn Positive **23/52**. Trên23 positives: matched18/23, shuffled3/23; trên29 negatives: matched24/29, shuffled15/29. Đây là tín hiệu proxy đáng báo cáo cho Prof nhưng chưa đủ gọi CLEAR là bộ kiểm tra report đúng với ảnh.

**Hạn chế của thiết kế B1 cần trình bày cùng kết quả âm:** (i) ngưỡng mention4/6 cao hơn C0≈2,96/6 và mọi arm đều dưới4/6; (ii) oracle gate69 ô trộn các ô O/S cùng direction và17 Pneumothorax cells không có CLEAR score; (iii) sensitivity ban đầu đo đổi state, chưa đo hướng; (iv) F1-O dùng schema categorical; (v) oracle là report-derived, có missing/uncertain, có thể nói lateral view. B1a giúp hiểu các giới hạn, không làm B1 “PASS” hậu nghiệm.

Nguồn: [B1a findings](../research/runs/20261017_E4_B1a_posthoc_discordant_analysis_v1/E4_B1a_findings.md), [CLEAR/reference cell table](../research/runs/20261017_E4_B1a_posthoc_discordant_analysis_v1/E4_B1a_clear_reference_proxy.csv).

### 8. Các nhánh chưa chạy — đầu vào và đầu ra mong đợi theo plan

| Nhánh | Đầu vào còn phải xác lập | Đầu ra cần có nếu triển khai | Hiện trạng thực tế |
|---|---|---|---|
| Hoàn tất E4-A2 | Annotators đủ chuyên môn, ảnh/report đã blind và rubric | Nhãn độc lập cho400 slots, adjudication/coverage và paired clinical comparison | Có package, **chưa có annotation**. Không thay bằng proxy labeler. |
| E4-B2 / phương án A | Cohort/protocol mới, rationale cho F3 dù B1 chưa qua, năm arms/ngưỡng/budget chốt trước | Report grid mới và evaluation độc lập với lựa chọn hậu nghiệm B1 | Chỉ đề xuất, chưa chạy.100×5=500 calls;2,93h model-call ngoại suy, chưa phải runtime đo được. |
| Phương án B, CLEAR checker | Ghép đúng ảnh–3.667 report, labels output, định nghĩa disagreement/flag, validation tham chiếu phù hợp | Danh sách cờ bất đồng và đánh giá đúng/sai của cờ trên tập review | Có texts baseline và CLEAR pipeline, **chưa có checker validated**;1,09–1,21h là image-scoring estimate, không toàn pipeline. |
| E3 | Paired train reports, frozen encoder/LLM, adapter design, matched original-encoder retraining control và resource preflight | So B0 released /B1 original encoder+trained adapter /B2 CLEAR+matched adapter về report generation | Chưa training; E1 signal chưa thay thế downstream control. |
| E4 trained fusion | Visual/concept representation alignment, paired training target, model/ablation/resource protocol | Learned fusion và image-only/finding-only/concept-only/matched-capacity controls | Chưa chạy; không nhập chung với prompt E4. |
| E6A | Adapter-training path và compute đủ cho controlled comparison | Linear vs2-layer MLP projector; Q-Former chỉ nếu có cơ sở | Chưa chạy, chưa feasibility preflight training. |
| E6B | Bottleneck latency/VRAM được đo và protocol quality trade-off | 1024→256→64 visual-token comparison, quality/latency/memory và small-pathology analysis | Chưa chạy. Không coi việc có K80 là bằng chứng rằng compression đã được kiểm tra. |
| Full CLEAR concept pathway | Concept bank và các assets đầy đủ đúng checkpoint, nguồn lực và protocol | Full concept inference đã validated | Hiện BLOCKED_ASSETS; direct five-finding scores không đáp ứng nhánh này. |

### 9. Tài nguyên, chi phí thực đo và khả năng tái lập

Snapshot server đã lưu ngày17/10: **2×Xeon E5-2620 v3**,12 cores/24 threads,47,07GiB RAM, không swap;8 logical CUDA devices K80,11.441MiB/device. E4 dùng4 devices, không phải toàn bộ8; tổng VRAM không phải một vùng nhớ chung. Driver440.33.01; `/home` chỉ còn khoảng1,1GiB khả dụng trong snapshot, `/storage` là NFS dùng chung. Chưa có quota cá nhân hoặc reservation dài hạn được xác nhận. Đây là snapshot lịch sử, không kiểm tra lại server trong lần viết nhật ký này.

| Run | Tài nguyên / thời gian đã lưu | Không được suy ra |
|---|---|---|
| E1 CXR | Một K80 FP32; runner **3.005,93s**, peak RSS3,20GiB,0,835GPU-device-hours | Không phải thời gian full VLM. |
| E1 CLEAR | CPU; runner **5.336,71s**, peak RSS1,41GiB,0GPU-hours | Không dùng so CPU-vs-K80 để kết luận encoder nhanh/chậm. |
| E2 | CPU; wall **549,46s**,50 cached +468 new forwards | Không phải full concept cost. |
| E1.3/E5 | CPU cache-only; caps E1.3≤60phút, E5≤120phút, RSS≤8GiB theo protocol | Caps là giới hạn, không phải actual time; không có LLM training trong780 fits. |
| G2-B2 | CPU; guard wall **57,51s**,100 score rows | Không bao gồm E4 report generation. |
| E4-R4 | Bốn K80; model subprocess **1.900,646s**, tới finalization **2.374,532s**;2,111829 GPU-device-hours; peak whole-device10.228,5MiB trên device cao nhất | Không dùng39,58phút làm toàn bộ chi phí viết code/audit, không extrapolate sang training. |
| E4-B1 | Bốn K80; cumulative CXR wall **6.563,21s** gồm smoke/load; mean full report call **21,08s**; CLEAR24-image CPU **28,54s** | Không có guarantee500 reports sẽ hoàn tất trong3,5h nếu output length/load thay đổi. |
| A1/A2 preparation/A3A/B1a | Audit/package/phân tích output đã lưu; B1a CPU-only | Không tính thêm80/264 generated reports cho mỗi analysis stage. |

Môi trường tách biệt: CXR K80 torch1.12.1+cu102 /Transformers4.36.2; CLEAR CPU torch2.3.1+cpu /Python3.10.22 theo manifest. Checkpoints/inputs/protocol/output hashes, seeds và per-run logs có sẵn; lần biên soạn này chỉ đối chiếu nguồn, không chạy lại toàn bộ verification. Không có fine-tune, LoRA/QLoRA, adapter, Q-Former hoặc end-to-end fusion training đã được chứng minh trong các run này.

### 10. Cách sắp xếp báo cáo cho Prof từ các đầu ra trên

| Phần trình bày | Nội dung và bảng nên đưa | Thông điệp có thể kết luận |
|---|---|---|
| 1. Bài toán và hypothesis | Mục tiêu report generation; sơ đồ bằng lời ảnh→CLEAR evidence→CXR prompt→report; hai đường prompt/frozen-feature | Cần kiểm tra downstream benefit, không lấy probe accuracy làm kết luận cuối. |
| 2. Baseline và dữ liệu | E0 3.667 reports, E-AUDIT6950/7252 parsed, bảng cohort/split/exposure | Pipeline tái lập được trong phạm vi local; giới hạn reference và data independence đã nhận diện. |
| 3. Có tín hiệu hữu ích ở CLEAR không? | E1 bốn representation rows; E2 ranking và cùng-mask F1 | Có tín hiệu finding trong thiết lập exploratory. |
| 4. Tín hiệu có bổ trợ encoder CXR không? | E1.3 CKA/R²; E5 AP ba arms trên OOF/DEV | Khác biệt biểu diễn chưa chuyển thành AP uplift của concat; hold complex fusion. |
| 5. Đưa evidence vào report có hoạt động không? | G1/G2 tóm tắt readiness; R4 bốn arms, coverage và17/20 equality | Technical grid hoàn tất; useful image-specific report benefit chưa chứng minh. |
| 6. Kiểm tra negative result kỹ hơn | B1 thiết kế11 arms/greedy/oracle; bốn gate; B1a C0/S/O, neutral background, directional counts | Prompt nhạy với context nhưng không format nào đạt gate; nguyên nhân cơ chế chưa cô lập. |
| 7. Kết luận và quyết định cần Prof | Phân biệt encoder signal, report agreement proxy, image truth; A/B/C và dependency chuyên gia/tài nguyên | Chọn câu hỏi khoa học kế tiếp và phạm vi phù hợp; giữ nguyên các quyết định frozen trước khi có protocol mới. |

**Đoạn tổng hợp có thể đưa vào báo cáo:** “Chúng tôi đã tái lập pipeline sinh báo cáo trên 3.667 ảnh frontal Indiana và khảo sát CLEAR ở cấp biểu diễn, phân loại direct contrastive và tích hợp qua prompt. CLEAR đạt kết quả tốt hơn CXR encoder trong matched NIH probes, nhưng ghép frozen features không tăng AP so với CLEAR đơn. Pilot E4-R4 hoàn thành 80 reports nhưng coverage đánh giá thấp và matched/shuffled evidence sinh cùng text ở 17/20 ảnh. Gate E4-B1 với 24 ảnh, 11 nhánh và 264 reports hoàn tất; không format nào đạt đủ bốn tiêu chí định trước. Phân tích hậu nghiệm thấy output thay đổi cả khi thêm context trung tính và ít trường hợp theo cả hai direction evidence. CLEAR vẫn khớp report tham chiếu ở 42/52 ô definite, cao hơn shuffled 18/52 và baseline luôn Negative 29/52, song chưa có adjudication trên ảnh. Kết quả hiện hỗ trợ một kết luận âm về prompt integration trong setup đã thử, chưa chứng minh lợi ích lâm sàng, không loại trừ mọi phương pháp prompt hoặc trained fusion.”

**Các quyết định cần trình Prof, chưa phải việc đã được chạy:** ưu tiên clinical review của output hiện có hay nghiên cứu proxy tiếp; mục tiêu A là kiểm tra giả thuyết nào với protocol/cohort mới; B cần validation cờ bất đồng như thế nào; C có đủ lý do khoa học để làm training preflight và nguồn lực nào được cấp. Không coi thiếu phần cứng là nguyên nhân đã chứng minh của negative result hiện tại.

**Chú giải:** AP=average precision; AUROC=diện tích dưới ROC; OOF=prediction ngoài fold huấn luyện; joint-definite=cả reference và prediction là Positive/Negative; unknown ở R4 gồm blank/uncertain; oracle ở đây=nhãn trích từ report tham chiếu, không phải ground truth trên ảnh. `VERIFIED` áp dụng cho identity/config/count/text đã kiểm; `AUTOMATED PROXY` cho label/metric tự động; `UNVERIFIED` cho correctness và clinical utility chưa có đánh giá phù hợp.

## Ghi chú dọn cấu trúc ngày 18/10/2026

Các plan, script chạy một lần, archive trung gian và báo cáo nháp/trùng lặp đã xóa theo yêu cầu. Xem [kế hoạch hiện hành](../plan/PROJECT_PLAN.md) và [module đọc catalog](../research/README.md). Các link nguồn còn lại trỏ tới báo cáo chính hoặc raw run manifest; tên hiển thị của một số link là tên báo cáo lịch sử để dễ nhận diện nội dung. Số liệu E/G và raw artifacts không thay đổi. Một số manifest cũ tham chiếu file nguồn/báo cáo đã xóa, nên tái lập nguyên trạng run cũ từ cây làm việc hiện tại có giới hạn. Dùng `python3 -m research.project status` để đọc trạng thái run/phase.
