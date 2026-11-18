# E4-B1a — Phân tích hậu nghiệm các ô evidence bất đồng

## 1. Experiment Information

- **ID/ngày:** E4-B1a, 17/10/2026 (giờ Việt Nam).
- **Dự án/repo:** CXR-LLaVA × CLEAR, `https://github.com/hanhpm/CXR_LLaVA_Improvement`.
- **Nhánh:** `research/e4-b1-evidence-format-gate`; **commit nguồn E4-B1:** `9f7b054`.
- **Trạng thái:** hoàn tất phân tích phụ, **AUTOMATED PROXY, POST-HOC**; không phải thí nghiệm inference mới hoặc đánh giá lâm sàng.
- **Quyết định E4-B1:** giữ nguyên `E4_B1_PROMPT_ROUTE_REJECTED`.
- **Code và đầu ra:** `research/e4_b1a_posthoc.py`, `tests/test_e4_b1a.py`, `research/runs/20261017_E4_B1a_posthoc_discordant_analysis_v1/`.

## 2. Objective

Kiểm tra lại các ô mà evidence oracle và CLEAR-shuffled hoặc CLEAR-matched và CLEAR-shuffled **thực sự bất đồng**, thay vì chỉ dùng số liệu gộp của E4-B1. Tách hai câu hỏi: mô hình có đổi output theo **hướng** evidence trong prompt không, và hướng CLEAR có khớp nhãn trích từ reference report không. Phân tích này không sửa ngưỡng hay đảo quyết định E4-B1.

## 3. Survey / Technical Motivation

E4-B1 cho thấy F2/F3 làm output bớt ngắn nhưng không qua cổng oracle-following. Chỉ số sensitivity gốc đếm mọi thay đổi trạng thái, không cho biết hướng thay đổi; đối chứng N1 cũng làm output thay đổi nhiều. Vì vậy, E4-B1a chỉ xét các cặp có xung đột evidence và đặt chúng cạnh C0 không evidence cùng N1 trung tính. Đây là kiểm tra hậu nghiệm, không có p-value hoặc CI.

## 4. Environment

Chỉ dùng Python trong môi trường CLEAR CPU đã có tại `research/envs/cxr-clear-cpu/`, với `CUDA_VISIBLE_DEVICES=''`. Không nạp CXR-LLaVA/CLEAR, không gọi GPU, không chạy CheXpert labeler lại, không cài package và không ghi vào `/home`. Nhãn và token counts lấy nguyên từ artifact E4-B1. Script xác minh 10 SHA-256 từ manifest E4-B1 và hash của `E4_B1_evidence_blocks.jsonl` từ artifact index **trước khi** tạo run dir E4-B1a.

## 5. Dependency Setup

Không thay đổi môi trường. Dùng thư viện chuẩn Python và hàm chuẩn hóa năm trạng thái nhãn đã dùng trong `research/e4_b1_analyze.py`.

## 6. Code / Notebook Changes

Thêm script CPU-only xuất sáu bảng bắt buộc, một bảng phụ so CLEAR với reference report, manifest và `E4_B1a_findings.md`. Thêm 5 unit tests cho hash, tập eligible, phân loại hướng thay đổi, trạng thái Not mentioned/Uncertain/Extraction failure và định nghĩa collapse. Không sửa file nào trong run E4-B1.

## 7. Experiment Configuration

| Đơn vị/tập | Quy mô |
|---|---:|
| Ảnh Indiana dev E4-B1 | 24 |
| Ô oracle Positive/Negative trên 6 findings | 69 |
| Ô oracle Positive/Negative trên 5 findings có CLEAR score | 52 |
| Ô oracle Positive trên 6 findings | 27 |
| Ảnh có ≥1 hướng CLEAR M/S khác nhau | 19/24 |
| Ô O–S bất đồng (oracle definite, shuffled ngược oracle) | 34/format |
| Ô M–S khác hướng CLEAR | 62/format |

Pneumothorax có nhãn trong report nhưng không có hướng CLEAR; nó không vào tập ô bất đồng hướng. `Not mentioned`, `Uncertain` và `Extraction failure` giữ riêng, không đổi thành Negative. O–S “không đổi” được báo cả trạng thái trích xuất và chuỗi output giống hệt; hai nghĩa này khác nhau.

## 8. Code Executed

Chạy `python -m unittest tests.test_e4_b1a -v` và `python -m research.e4_b1a_posthoc` bằng Python của môi trường CLEAR CPU, với `CUDA_VISIBLE_DEVICES=''`. Script đối chiếu các file nguồn với `E4_B1_run_manifest.json`, rồi đọc 264 báo cáo và 264 hàng nhãn đã lưu. Không có bước inference.

## 9. Output

Run mới chứa `E4_B1a_discordant_cells.csv`, `E4_B1a_positive_cells.csv`, `E4_B1a_direction_of_change.csv`, `E4_B1a_arm_summary.csv`, `E4_B1a_oracle_block_inventory.csv`, bảng phụ `E4_B1a_clear_reference_proxy.csv`, `E4_B1a_findings.md` và `E4_B1a_run_manifest.json`. Các bảng cell-level giữ image ID, finding, trạng thái và cờ đếm để kiểm toán từng mẫu số; không có ảnh hoặc report bệnh nhân mới được sao chép.

## 10. Output Analysis

Trên 34 ô O–S bất đồng mỗi format, output O khớp oracle **6/34 F1, 7/34 F2, 12/34 F3**. S khớp oracle lần lượt **6/34, 17/34, 15/34**; C0 khớp **17/34** trên đúng các ô đó. O không vượt S ở format nào. Trạng thái O và S giống nhau lần lượt 23/34, 15/34, 20/34, nhưng văn bản không giống hệt ở ô nào (0/34); vậy biến đổi câu chữ không tương đương với làm theo hướng oracle.

Trong 62 ô M–S khác hướng mỗi format, cả M và S cùng theo đúng hướng riêng của evidence ở **0/62 F1, 2/62 F2, 0/62 F3**; ngược cả hai hướng là **0/62, 0/62, 2/62**. Trường hợp chỉ một phía đổi, hoặc rơi vào Not mentioned/Uncertain, được giữ ở nhóm `other_target_state_change`, không ép vào “theo evidence”. Bảng chi tiết và phân tách theo oracle Positive/Negative, từng finding nằm trong `E4_B1a_findings.md`.

## 11. Quantitative Results

| Đối chứng trên cùng 19 ảnh | Text khác | Có thay đổi ≥1/6 trạng thái | Thay đổi trạng thái trên 5 findings |
|---|---:|---:|---:|
| N1 so C0 | 19/19 | 19/19 | 36/95 |
| F1-M so F1-S | 4/19 | 3/19 | 6/95 |
| F2-M so F2-S | 17/19 | 11/19 | 24/95 |
| F3-M so F3-S | 15/19 | 13/19 | 32/95 |

Đây là mức nền của độ nhạy với **văn bản trung tính**, không phải phép trừ nhiễu có hiệu lực nhân quả. Arm summary còn ghi collapse trên 24 báo cáo, số trạng thái finding được nhắc trên 144 ô, và chiều dài ký tự/token; C0 nhắc 71/144 ô (2,96/6 mỗi ảnh), F2-M 58/144, F3-M 64/144. F2-O collapse 7/24; nhóm này có 13/42 câu oracle Positive, so với 14/102 ở 17 ảnh không collapse. Chưa xác định nguyên nhân collapse.

Trên 52 ô definite của năm finding có CLEAR score, hướng CLEAR-matched khớp **reference-report label 42/52**, shuffled **18/52**. Theo trạng thái tham chiếu: Positive 18/23 so 3/23; Negative 24/29 so 15/29. Đây là proxy so với report, **không phải độ đúng trên ảnh**.

## 12. Qualitative Results

Không đánh giá hình ảnh hoặc chép report để phán đoán lâm sàng. Bảng `E4_B1a_oracle_block_inventory.csv` kiểm kê sáu câu trong mỗi khối oracle và số finding mà output F1/F2/F3-O nhắc đến. Các câu `Uncertain` và `Not mentioned` vẫn là trạng thái riêng. Với F1-O, bảng oracle dạng phân loại khác cấu trúc bảng cosine của F1-M/S, nên không cô lập tác động riêng của giá trị số.

## 13. Comparison with Baseline

Đối với **toàn bộ 69 ô oracle definite**, C0 khớp 36/69; F1-S/O 8/69 và 15/69; F2-S/O 31/69 và 10/69; F3-S/O 36/69 và 33/69. Trên 34 ô bất đồng, C0 luôn 17/34. Những số đếm này không cho thấy oracle evidence được dùng **đúng hướng mong đợi**; đồng thời output thay đổi theo prompt là quan sát trực tiếp. Không diễn giải các cặp như một thử nghiệm hiệu quả CLEAR được phân bổ ngẫu nhiên.

## 14. Problems Encountered

Không có lỗi chạy hoặc thiếu file nguồn. Vấn đề phương pháp là nhãn report tự động không đồng nhất với truth trên ảnh; ô thiếu nhắc tới vẫn rất nhiều. Một số nhóm theo finding × oracle Negative có 0 ô eligible; bảng ghi `0/0 (N/A)` thay vì tính tỷ lệ. Phân loại M–S có nhóm thứ năm cho chuyển trạng thái không trọn hai hướng, tránh nhãn “không liên quan” sai nghĩa.

## 15. Conclusion

**VERIFIED:** evidence/prompt khác và output text có thay đổi, nhưng thay đổi văn bản không tự nó chứng minh mô hình theo nội dung evidence. **AUTOMATED PROXY, POST-HOC:** trên các ô bất đồng, sự theo hướng evidence đầy đủ rất ít, oracle O không hơn S; biến đổi trạng thái do N1 trung tính là mức nền lớn. **AUTOMATED PROXY, POST-HOC:** CLEAR-matched khớp nhãn reference report 42/52 ô definite năm finding. **UNVERIFIED:** mô hình có dùng evidence theo nghĩa ngữ nghĩa hay không, CLEAR đúng với ảnh hay không, và giá trị lâm sàng. Quyết định E4-B1 giữ nguyên `E4_B1_PROMPT_ROUTE_REJECTED`.

## 16. Next Experiment

Không đề xuất hoặc khởi chạy E4-B2 trong phân tích này. Giữ nguyên `HOLD_MORE_EVIDENCE` và `HOLD_COMPLEX_FUSION`.

## 17. Reproducibility Checklist

- [x] Nguồn, nhánh, commit E4-B1 và toàn bộ hash đầu vào được ghi trong manifest E4-B1a.
- [x] 24×11 output và CheXpert labels tái sử dụng nguyên trạng; mã chỉ chạy CPU.
- [x] Định nghĩa discordant, các tập eligible và mẫu số được ghi cạnh mỗi bảng.
- [x] Năm unit tests qua; không có p-value, CI hoặc đánh giá đúng sai ảnh.
- [x] Các file E4-B1 không bị sửa.

## 18. Short Lab Update

E4-B1a phân tích hậu nghiệm 34 ô oracle–shuffled bất đồng và 62 ô matched–shuffled khác hướng mỗi format. Oracle không tăng mức khớp report so với shuffled trên các ô xung đột; trường hợp cả hai nhánh cùng theo evidence riêng chỉ 0/62, 2/62, 0/62. N1 trung tính làm trạng thái output thay đổi trên 19/19 ảnh cùng tập, nên “độ nhạy” không đủ chứng minh evidence được dùng đúng hướng. CLEAR-matched khớp nhãn report 42/52 ô, nhưng chưa được xác nhận trên ảnh.
