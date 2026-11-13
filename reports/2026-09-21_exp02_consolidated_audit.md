# Experiment Analysis Log — timeline hợp nhất và audit EXP-02

## 1. Experiment Information

- **Date:** 2026-09-21; **Project:** CXR_LLaVA_Improvement.
- **Experiment ID:** EXP02-CONSOLIDATED-AUDIT.
- **Status:** hoàn thành tái tính offline và review kỹ thuật; runtime prompt audit, cohort phân tầng và inference đối chứng chưa hoàn tất.
- **Phân loại:** audit kỹ thuật / descriptive subset evaluation, không phải benchmark hoặc xác nhận lâm sàng.
- **Run mới:** `20260921T081624Z_560fa793` dưới `result/CXR_LLaVA_EXP02_Indiana_External_Eval_v2/`, ký hiệu **A/** bên dưới.
- **Raw lịch sử:** `result/CXR_LLaVA_EXP02_Indiana_External_Eval/`, ký hiệu **R/**. Không sửa raw CSV/labels.

## 2. Objective

Hợp nhất ba log 16/09, 17/09 và 21/09 thành một lịch sử có mức bằng chứng rõ; truy ROUGE-L; kiểm tra prompt/revision trước chẩn đoán output rỗng; sửa evaluator; review mismatch/control/masked findings; thiết kế cohort phân tầng và đối chứng trên cùng ảnh.

## 3. Survey / Technical Motivation

Nguồn: quy định CXR-local; notebook 00–03; model helper local và remote tại revision đã pin; CSV predictions/input/row map/labels/metrics; ba báo cáo liên kết bên dưới. Báo cáo dùng cấu trúc `experiment_analysis_template.md`.

CNN chọn từ [TorchXRayVision chính thức](https://github.com/mlmed/torchxrayvision): DenseNet121 CheXpert, bảy target tương ứng có trong [metadata checkpoint](https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/models.py). Đây là comparator classification riêng, không thay CheXpert text labeler.

## 4. Environment

Tái tính bằng Python/pandas/numpy local, phiên bản trong `A/run_config.json`. Không load CXR-LLaVA, không chạy inference hay labeler. ROUGE tái tính bằng `rouge-score==0.1.2`, `RougeScorer(['rougeL'], use_stemmer=True)`.

Log lịch sử `R/logs/experiment_environment.txt` ghi `2026-09-18T10:31:02Z`, Tesla T4, Python 3.13.15, torch 2.11.0+cu128, Transformers 4.46.3, NF4/FP16. Timestamp này là lúc ghi environment, không xác định ngày chạy riêng của 10/50/100.

## 5. Dependency Setup

Audit nhẹ chỉ bổ sung `python -m pip install rouge-score==0.1.2`. Không cài CNN hoặc thay stack model local. Notebook recovery vẫn yêu cầu Colab GPU và load NF4/FP16; giữ vision tower/mm_projector/lm_head khỏi quantization và `device_map={'': 0}`.

## 6. Code / Notebook Changes

- `scripts/exp02_reevaluate.py`: giữ canonical cuối mỗi `image_id`, kiểm tra GT nhất quán giữa attempts và subset, đọc ID bằng string. Dùng F1 trực tiếp từ counts và cùng policy trong bootstrap đã có; kiểm chứng lại bằng tests/artifacts mới.
- Notebook 02: resume tính completed IDs sau dedup; label input lấy canonical success không rỗng; evaluator dùng module chung, lưu run v2 mới, không cần model hoặc rerun labeler. Cleanup không ghi đè raw nữa.
- `scripts/exp02_provenance.py`: pin model/code/tokenizer/generation-config; capture chat thực từ `write_radiologic_report` trước generation, lưu rendered prompt/token IDs/source hashes. Capture luôn khôi phục method gốc bằng context manager.
- Notebook 03: bỏ loader trùng, dùng lại compatibility patch `LlamaTokenizer` của 00/01 thay patch `AutoTokenizer`; provenance flag mặc định False, legacy fallback 00/01; chặn gate inference đến khi cohort đã qua quota và GT/view review. Không đánh inference-only thành gate đầy đủ.
- Thêm `exp02_stratify.py`, `exp02_cnn_control.py`, tests và protocol cohort/đối chứng lịch sử (file plan đã xóa ngày 18/10/2026). Generator notebook cũ từ chối ghi đè notebook đã sửa tay.

## 7. Experiment Configuration

Offline: các subset lịch sử 10, 50, 100; seed42; bootstrap 1.000 lần/bệnh; joint-definite GT/pred trong {0,1}. Blank và −1 không tự chuyển negative. Chỉ canonical success có report không rỗng được dùng cho label metrics; failure denominator là số ảnh yêu cầu.

Revision pin cho run mới: `b2224786bb90d54b1e1291171866706cfbb44e2b`, xác minh qua Hugging Face API ngày 21/09. Đây **không phải bằng chứng revision đã tạo output lịch sử** (chỉ ghi `main`). Source remote tại SHA này được lưu trong `A/audit/pinned_remote_model_source.py`.

## 8. Code Executed

```powershell
python -m unittest discover -s tests -p 'test_exp02*.py'
python scripts/exp02_reevaluate.py --source result/CXR_LLaVA_EXP02_Indiana_External_Eval --output result/CXR_LLaVA_EXP02_Indiana_External_Eval_v2/NEW_UNIQUE_RUN_ID --sizes 10 50 100 --bootstrap-iterations 1000 --seed 42
```

Thay `NEW_UNIQUE_RUN_ID` bằng UTC+UUID chưa tồn tại; script từ chối ghi đè. Ngoài evaluator: tái tính ROUGE trên 24 predictions, render fallback bằng Jinja từ code notebook + helper remote pinned, kiểm kê ảnh/XML, review 45 cặp và kiểm tra quota từ GT đã lưu. Không chạy notebook `Run all`.

## 9. Output — timeline hợp nhất

| Mốc có bằng chứng | Nội dung | Trạng thái hiện tại |
|---|---|---|
| 16/09 — baseline đa dataset (báo cáo nháp đã xóa) | padChest 24 reports; CSV ROUGE-L 0,0156677119; CXR8/Brixia là task khác | Giữ kết quả CSV; không gộp thành metric Indiana |
| 17/09 — EXP-02 (báo cáo tiến độ đã xóa) | Ghi tiến độ 10-image/labeler và các gate còn mở | **SUPERSEDED** về trạng thái; bảo toàn nội dung lịch sử |
| 18/09 10:31:02 UTC | Environment log chung của artifacts Indiana | Chỉ chứng minh timestamp log; không dựng thêm thời gian inference |
| 21/09 — phân tích 10/50/100 (báo cáo trung gian đã xóa) | 10/10, 50/50, 98/100; lỗi dedup/F1/bootstrap; review còn trống | Giữ làm phân tích trước sửa evaluator |
| 21/09 — v2 `20260921T031045Z_a2ca54af` đã có khi bắt đầu | Evaluator và review queue sơ bộ | Không coi queue là review hoàn tất |
| 21/09 — audit hiện tại `20260921T081624Z_560fa793` | Tái tính offline, truy ROUGE, render prompt static, review 45 ảnh, kiểm tra quota | Nguồn trạng thái mới; các phần chưa chạy được nêu riêng |

Không có timestamp theo ảnh để xác định thứ tự chính xác các lượt inference. 10 ⊂ 50 ⊂ 100, không phải ba thử nghiệm độc lập.

## 10. Output Analysis — ROUGE-L và prompt

### ROUGE-L 0,0157 vs 0,0209

| Nguồn | Số mẫu | ROUGE-L |
|---|---:|---:|
| `result/padchest_datasample_baseline_metrics.csv` | 24 | 0,015667711922180482 |
| Tái tính từ `result/padchest_datasample_baseline_predictions.csv` | 24 | 0,01566771192218049 |
| Saved output notebook 01, code cell index20 | 24 theo output cell16/18 | 0,020943 (độ chính xác được lưu) |

Từng dòng CSV khớp tái tính, sai số tối đa 6,94×10⁻¹⁷; xem `A/audit/padchest_rouge_recomputed.csv`. **0,0157 là điểm tái lập được từ raw CSV hiện có; 0,0209 là aggregate của snapshot output notebook khác.**

Bằng chứng snapshot khác: preview notebook cell18 có latency hàng đầu 8,000139 giây, CSV có 7,055006; generation_tokens hàng thứ hai notebook 68, CSV 69; hàng thứ năm notebook 88, CSV 67. Notebook lưu đường dẫn `results/`, còn bản CSV local ở `result/`. Chưa có toàn bộ predictions/metrics của snapshot 0,020943 để xác định các dòng đóng góp hoặc nguyên nhân generation thay đổi. Không quy chênh lệch cho rounding, evaluator bug, prompt hoặc model improvement khi chưa có raw snapshot tương ứng.

Reference padChest là chuỗi tiếng Tây Ban Nha đã xử lý, generated report là tiếng Anh; lexical ROUGE trên cặp này không đo trực tiếp chất lượng lâm sàng. Cả hai giá trị đều không phải ROUGE Indiana.

### Prompt 02 so với 00/01

`A/audit/00_rendered_report_prompt_static.txt`, `01_...`, `02_...` và `prompt_comparison.json` chứa render đầy đủ từ template fallback hiện tại và chat của helper remote pinned.

- 00/01: `<s>[INST] <<SYS>> ... <</SYS>> ... [/INST]`.
- 02: nối thẳng system content và user content, không `[INST]`, không system delimiters; cuối system text liền với `<image>`.
- Hai render 00/01 giống nhau; 02 khác. Helper yêu cầu sáu finding gồm atelectasis nhưng không hỏi tường minh Lung Opacity/Pneumonia trong bảy bệnh đánh giá.

Đây là **static reconstruction khi fallback được dùng**, không phải log actual tokenizer của lần chạy tháng 9. Notebook 02 không lưu runtime output để biết fallback có thực sự được kích hoạt. Runtime logger mới capture đúng chat của helper, thay cho synthetic chat ngắn không đại diện; cần chạy trên Colab sau preflight để đóng phần runtime provenance. Chưa có cơ sở kết luận template gây hai output rỗng.

## 11. Quantitative Results

| Subset | Canonical success/failed | F1 Cardiomegaly | Consolidation | Lung Opacity | Pleural Effusion | Pneumonia |
|---|---|---:|---:|---:|---:|---:|
| 10 | 10 / 0 | 0 | 0 | 1 | undefined | undefined |
| 50 | 50 / 0 | 0,545455 | 0 | 0,666667 | 0 | undefined |
| 100 | 98 / 2 | 0,583333 | 0 | 0,727273 | 0 | 0 |

Edema/Pneumothorax undefined ở cả ba subset. Không xuất macro-F1 bảy bệnh. Confusion counts 21 tổ hợp giữ nguyên; F1 sửa bảy ô từ NaN thành 0. Failure bộ100 = **2/100 = 2%**, không phải 4/102. Bootstrap dùng cùng hàm F1, loại undefined khỏi percentile và ghi số resample undefined; CI có điều kiện trên resample defined, không ngụ ý độ chắc chắn khi support thấp.

Nguồn: `A/metrics/*_pathology_metrics_v2.csv`, `*_bootstrap_ci_v2.csv`, `*_old_new_metric_diff.csv`, `system_metrics_v2.csv`. Alignment kiểm tra ID, thứ tự và exact report text giữa canonical, row map, input không header và label Reports; toàn bộ PASS.

## 12. Qualitative Results

Đã đọc toàn bộ GT/generated của **45 ảnh độc lập**: 23 mismatch (28 records), 10 control không joint-definite mismatch và 12 ảnh bổ sung từ omission queue. Có 21 masked-positive records trong queue, chồng lấp với mismatch/control; không cộng các nhóm records như ảnh độc lập.

Ảnh được xem qua năm contact sheets (preview tối đa390×380/ảnh), ghi image SHA256 và view; không dùng preview làm chẩn đoán. Năm ảnh lateral: CXR1072, CXR1244, CXR1961, CXR2111, CXR2122. Các ảnh khác trong review là frontal theo preview, chưa adjudicate AP/PA.

Artifacts: `A/review/manual_image_review.csv` (45 nhận xét riêng), `review_completed.csv` (mỗi mismatch/control/omission có full text evidence và notes), `all_masked_findings_reviewed_images.csv` (mọi cặp bị mask trên các ảnh đã review). Giữ raw labels để không thay kết quả hậu nghiệm.

Các phát hiện cần xử lý:

- CXR1019: GT và GEN đều nói heart size normal nhưng có Cardiomegaly mismatch; đồng thời effusion-versus-scarring thành khẳng định effusion. Hai vấn đề phải tách riêng.
- CXR1115 (control): GEN vừa khẳng định consolidation vừa phủ định consolidation. Không mismatch không có nghĩa output đúng.
- CXR1222 (control): GT clear lungs/no active disease, GEN pneumothorax/fluid/subcutaneous emphysema; mask bỏ qua khác biệt lớn trong văn bản.
- CXR1798: cả GT và GEN phủ định pneumonia nhưng nhãn mismatch; cần kiểm tra negation labeler.
- CXR2067: GT “No focal airspace consolidations” nhưng GT label positive; queue tự động không chứng minh model bỏ sót consolidation.
- CXR1244/CXR1489/CXR2167: “no large effusion” không phủ định “small effusion”; FP nhị phân cần review qualifier.
- CXR2111 lateral: GT bệnh trên lateral và emphysema không được mô tả trong GEN; ảnh F1 không bảo đảm frontal.

Tất cả review có `clinical_validation_status=not_clinically_validated`. Nhận xét là text/reference discrepancy hoặc ứng viên lỗi labeler, không tự xác nhận chẩn đoán từ ảnh.

## 13. Comparison with Baseline

Evaluator sửa nghĩa của F1/denominator, không cải thiện mô hình. Không so điểm padChest với Indiana. Không so enriched cohort tương lai với random subset cũ như paired improvement. CNN đã có runner/preflight nhưng chưa có predictions trên cohort mới; không có kết quả đối chứng để báo cáo.

## 14. Problems Encountered

1. **Thiếu provenance lịch sử:** chưa biết resolved revision/actual rendered prompt của output cũ. Pin mới và static comparison không lấp được bằng chứng thiếu này.
2. **GT label artifacts:** nhiều normal/negated report bị positive; phải audit trước stratification, không lấy quota từ labels chưa review rồi coi là ca bệnh xác nhận.
3. **Thiếu support:** GT-positive thô trên98 ảnh = Cardiomegaly15, Consolidation1, Edema1, Lung Opacity15, Pleural Effusion0, Pneumonia2, Pneumothorax0. Không đủ quota20–30/bệnh; `A/audit/stratification_feasibility.csv` lưu cả negative/uncertain/unmentioned.
4. **View:** F1 chứa lateral; report có thể tổng hợp hai view. Cohort chính mới cần confirmed frontal và một study/ảnh.
5. **Hai empty outputs:** giữ nguyên failed; chưa retry hoặc kết luận nguyên nhân.

## 15. Conclusion

CSV ROUGE-L0,0156677 đã được tái lập; 0,020943 thuộc saved notebook snapshot chưa có full raw tương ứng. Tái tính metric offline sửa zero-policy và tỷ lệ failed, giữ nguyên confusion counts. Review đã chuyển từ scaffold sang nhận xét riêng45 ảnh và cho thấy cả labeler artifacts lẫn lỗi văn bản bị mask. Chưa đủ nhãn GT tin cậy để chốt cohort mới hoặc chạy đối chứng.

## 16. Next Experiment

Theo protocol đề xuất tại thời điểm báo cáo (file plan đã xóa ngày 18/10/2026): actual prompt audit trên Colab, label/review candidate GT và view, freeze cohort target25 dương/25 âm mỗi bệnh, gate10, rồi CXR-LLaVA và CNN cùng cohort/hash. So sánh trên common mask và báo abstention riêng; không âm thầm map blank thành negative.

Notebook03: chạy config/path/preflight; bật `RUN_MODEL_PROVENANCE` để load đúng một lần và lưu actual report prompt. Giữ retry/gate flags False đến khi điều kiện tương ứng thỏa. Không chạy lại cell cài dependency ở mỗi vòng sau restart; bỏ qua khi stack đã được kiểm tra.

## 17. Reproducibility Checklist

- [x] Raw labels/predictions giữ nguyên; source hashes và alignment lưu.
- [x] 13 unit tests cho F1/bootstrap/dedup/alignment/stratification PASS.
- [x] Recomputed ROUGE từng dòng khớp CSV.
- [x] Static rendered prompt00/01/02 và pinned remote source lưu.
- [x] Review45 ảnh có notes/evidence/image hash, nêu giới hạn preview.
- [ ] Historical resolved revision và raw snapshot ROUGE0,020943.
- [ ] Actual runtime prompt/model load xác nhận trên Colab.
- [ ] Reviewed GT pool đủ quota và frozen cohort.
- [ ] CNN/VLM inference và paired comparison trên cohort mới.

## 18. Short Lab Update

Đã hợp nhất lịch sử, đánh dấu log17/09 superseded, tái lập ROUGE0,0156677 từ CSV và xác định0,020943 là snapshot notebook khác. Evaluator offline sửa F1/zero-policy/dedup; bộ100 vẫn98 success,2 failed. Review45 ảnh cho thấy lateral trong F1, labeler artifacts và lỗi bị mask ở cả control. Đã thêm thiết kế cohort25 positive/bệnh và CNN comparator, nhưng cần GT/view review đủ pool và runtime prompt audit trước inference mới.
