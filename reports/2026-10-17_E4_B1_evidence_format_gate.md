# E4-B1 — Evidence-format gate trên cohort Indiana development

## 1. Experiment Information

- **Experiment ID / date:** E4-B1, 17/10/2026 (giờ Việt Nam).
- **Project / repo:** CXR-LLaVA × CLEAR; `https://github.com/hanhpm/CXR_LLaVA_Improvement`.
- **Branch:** `research/e4-b1-evidence-format-gate`; **HEAD lúc chạy:** `f6535c6d9d7eb23c2e9a2ac24f48771a1d276efc`.
- **Status:** hoàn tất technical run 264/264; quyết định theo protocol `E4_B1_PROMPT_ROUTE_REJECTED` trong thiết lập đã thử. Đây là pilot development, không phải clinical validation.
- **Scripts:** `research/e4_b1_{cohort,clear,prepare,gpu,label,analyze,synthetic}.py`; artifact gốc ở `research/runs/20261017_E4_B1_evidence_format_gate_v1/`.

## 2. Objective

Kiểm tra việc đổi định dạng evidence trong prompt có tránh được đầu ra ngắn/lặp của E4-R4 và có khiến CXR-LLaVA theo evidence tham chiếu (oracle từ report) hơn evidence xáo trộn hay không. Chỉ chọn định dạng cho nghiên cứu tiếp theo nếu bốn tiêu chí đã chốt cùng đạt; không ước lượng chất lượng báo cáo hay lợi ích lâm sàng của CLEAR.

## 3. Survey / Technical Motivation

E4-A3A xác nhận C2a/C3 khác evidence trên 20/20 ảnh nhưng output giống nhau 17/20 khi sampling; C1 dài tương đương mà không collapse tương tự. E4-B1 so sánh bảng số F1, câu văn F2 và câu văn kèm chỉ dẫn F3 bằng greedy, với C0, neutral N1, matched M, shuffled S và oracle O. Giả thuyết định dạng số gây collapse là hợp lý để kiểm tra, nhưng nguyên nhân cơ chế chưa được cô lập.

## 4. Environment

| Thành phần | Cấu hình đã chạy |
|---|---|
| CXR-LLaVA | Checkpoint FP32 local cùng revision E4-R4; PyTorch 1.12.1+cu102, Transformers 4.36.2; 4 CUDA devices Tesla K80, batch 1 |
| CLEAR | Checkpoint `best_model.pt` local; PyTorch 2.3.1+cpu, 24 ảnh CPU-only |
| CheXpert labeler | Bản local offline, revision `44ddeb363149aa657296237f18b5472a73c1756f`; NegBio revision `073199e2792824740e89844a59c13d3d40ce4d23` |
| Cache | `TMPDIR`, `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME` đều dưới `/storage/student4/hanhpm/.e4b1_cache/` |

UUID GPU, device map, phiên bản và peak bộ nhớ có trong `E4_B1_run_manifest.json`, `E4_B1_full_model_load.json` và từng report record. Không tải checkpoint/dataset, cài gói hay ghi vào `/home`.

## 5. Dependency Setup

Không cài hoặc nâng cấp thư viện. Model và labeler chạy bằng các môi trường đã có. Cảnh báo Transformers rằng `temperature=0` và `top_p=0.8` không có tác dụng khi `do_sample=False` là dự kiến; effective generation kwargs được lưu cho từng lượt.

## 6. Code / Notebook Changes

Thêm mã chọn cohort, tính CLEAR CPU, đóng băng prompt/`input_ids`, chạy CXR-LLaVA có resume, labeler, tính gate và kiểm thử. Script `research/e4_b1_background.sh` cho phép full run tiếp tục trong `tmux` độc lập với phiên Codex. Không sửa weights, hàm sinh gốc hoặc artifact E4 trước đó.

## 7. Experiment Configuration

| Tham số | Giá trị |
|---|---|
| Cohort | 24 Indiana frontal development images, loại 20 ảnh E4-R4 và CheXpert locked test |
| Positive theo finding | Atelectasis 4, Cardiomegaly 5, Consolidation 4, Edema 4, Pleural Effusion 6, Pneumothorax 4 |
| Arms | C0, N1, F1-M/S/O, F2-M/S/O, F3-M/S/O: 11 × 24 = 264 |
| Seeds | chọn cohort `2026101701`; derangement `2026101702`; greedy không cần seed |
| CLEAR | Direct-contrastive năm finding; threshold-free direction; không có score Pneumothorax |
| Oracle | Nhãn trích từ reference report, gồm Positive/Negative/Uncertain/Not mentioned; không phải ground truth ảnh |
| Decoding | Greedy (`do_sample=False`), 512 max new tokens, stop keyword gốc, batch 1, FP32 |
| Protocol | SHA-256 `a6eb9661672b65816630866ec2ad8a9676179562202c4e0b2966cebb0b599269` |

Prompt và ngưỡng được đóng băng trước inference trong `E4_B1_protocol.md`; từng bản ghi lưu rendered prompt và `input_ids`. F1-M/S dùng đúng bảng số E4-R4. F1-O dùng bảng trạng thái phân loại vì không được bịa cosine từ nhãn report; hạn chế này làm so sánh oracle F1 không cô lập riêng tác động định dạng số.

## 8. Code Executed

Sau đóng băng protocol: tính CLEAR CPU, render 264 prompt, smoke 26 lượt, full run 264 lượt với resume, chạy CheXpert labeler offline, rồi phân tích bằng `research.e4_b1_analyze`. `E4_B1_background.log`, `E4_B1_gpu_windows.jsonl` và `E4_B1_operator_interruptions.jsonl` ghi rõ một lần ngắt có chủ đích để tách tiến trình khỏi Codex; 30 report đã hoàn tất được giữ lại. Smoke outputs không được tái sử dụng cho gate.

## 9. Output

Full grid có 264 cặp ảnh–arm duy nhất, tất cả `success`; 264/264 lượt dừng theo stop keyword gốc. C0 lặp lại byte-identical hai lần trên mỗi trong hai ảnh smoke. CLEAR tạo 120 hàng score. Labeler tạo 264 hàng và 1.584 trạng thái finding, không có extraction failure. Các file raw được liệt kê trong `E4_B1_artifact_index.md`.

## 10. Output Analysis

F1-M collapse 17/24, và đúng câu `No acute cardiopulmonary process.` xuất hiện 17/24. F2-M và F3-M không có output collapse theo định nghĩa đã chốt (0/24), nhưng mean mention coverage chỉ 2,42/6 và 2,67/6, dưới ngưỡng 4/6. Điều này cho thấy đổi sang câu văn giúp tránh dạng output ngắn trong cohort này; không chứng minh báo cáo chính xác hơn.

Trên 19 ảnh mà direction CLEAR matched/shuffled khác ít nhất một finding, text M/S khác nhau ở F1 4/19, F2 17/19, F3 15/19. Đây là quan sát trực tiếp về **thay đổi văn bản**. Chỉ số gate nhạy evidence dùng **trạng thái nhãn trích từ output**, lần lượt 3/19, 11/19, 13/19. Mô hình có phản ứng với prompt trong F2/F3, nhưng phản ứng đó chưa chứng tỏ nghe đúng oracle hay dựa đúng ảnh.

## 11. Quantitative Results

| Format | O collapse | M collapse | M mention trung bình | Oracle O vs S trên cùng 69 cell xác định | M/S đổi trạng thái / 19 | Qua gate? |
|---|---:|---:|---:|---:|---:|---|
| F1 bảng số | 10/24 | 17/24 | 0,58/6 | 15/69 vs 8/69; +10,1 pp | 3/19 | Không |
| F2 câu văn | 7/24 | 0/24 | 2,42/6 | 10/69 vs 31/69; −30,4 pp | 11/19 | Không |
| F3 câu văn + chỉ dẫn | 0/24 | 0/24 | 2,67/6 | 33/69 vs 36/69; −4,3 pp | 13/19 | Không |

Ngưỡng oracle là ít nhất +20 điểm phần trăm với ≥20 cell; cả ba format đều không đạt. N1 collapse 6/24, C0 0/24. Không có CI hay kiểm định ý nghĩa theo protocol pilot.

## 12. Qualitative Results

`E4_B1_qualitative.md` chứa bảng M/S/O song song của **năm image ID đầu tiên theo thứ tự sort đã chốt trước inference**. Không diễn giải output đó thành đúng/sai lâm sàng vì chưa có bác sĩ đọc ảnh; oracle chỉ phản ánh text reference và có thể nói cả lateral view không đưa vào model.

## 13. Comparison with Baseline

E4-R4 có output C2a/C3 giống nhau 17/20 dưới sampling; E4-B1 thay cohort development, dùng greedy và ba định dạng. F1 vẫn sinh câu mặc định trên 17/24, trong khi F2/F3 tạo nhiều text khác nhau hơn. Khác cohort và decoding nên không lấy chênh lệch này làm hiệu quả can thiệp có kiểm định.

## 14. Problems Encountered

Full run gắn với phiên Codex được ngắt theo yêu cầu người dùng để chuyển sang `tmux`; cửa sổ GPU được đóng, 30 commits được giữ và run nối tiếp cùng prompt hashes/cấu hình. Labeler tính xong nhưng mã provenance ban đầu dùng `text=True`, không hỗ trợ Python 3.6; sửa thành `universal_newlines=True`, kiểm tra file input/output đã có và ghi manifest mà không chấm lại. Không có OOM.

## 15. Conclusion

Theo quyết định đã chốt trước chạy, E4-B1 là `E4_B1_PROMPT_ROUTE_REJECTED` **trong thiết lập này**, vì không format nào vượt shuffled ≥20 điểm phần trăm về oracle following. F2/F3 làm output bớt collapse và thay đổi khi đổi evidence, nên không kết luận CXR-LLaVA hoàn toàn không nhạy với text. Labeler và reference-report oracle không xác nhận factuality trên ảnh, CLEAR benefit hoặc giá trị lâm sàng. Giữ `HOLD_MORE_EVIDENCE` và `HOLD_COMPLEX_FUSION`; không khởi chạy E4-B2.

## 16. Next Experiment

Chỉ khuyến nghị: nếu thiết kế E4-B2, cần ít nhất 15 ca reference-positive cho mỗi finding; so với pilot này còn thiếu 9–11 positives tùy finding, chưa thể suy ra tổng N vì overlap/split. Cần protocol development mới để chọn format và bảo đảm quyền dữ liệu, rồi mới ước lượng riêng ngân sách; không tự động chạy cohort lớn.

## 17. Reproducibility Checklist

Protocol, seed, cohort/score/checkpoint hashes, prompt render, `input_ids`, output token/stop reason, phiên bản môi trường, GPU UUID, thời gian, interrupt, raw nhãn và các tiêu chí đều đã lưu. Sáu unit tests và synthetic dry run trong `synthetic/` đã qua. Run manifest và artifact index là điểm vào để kiểm tra lại; bản commit local sẽ được ghi sau khi hoàn tất kiểm toán báo cáo.

## 18. Short Lab Update

E4-B1 đã sinh đủ 264 báo cáo trong 6.563 giây GPU wall cộng dồn trên 4 K80. Cách viết evidence bằng câu văn tránh collapse của nhánh matched nhưng không đạt mention coverage hoặc lợi thế oracle đã chốt. Kết luận nghiên cứu là prompt-evidence route chưa qua cổng development này; chưa có bằng chứng CLEAR cải thiện báo cáo và chưa có đánh giá lâm sàng.
