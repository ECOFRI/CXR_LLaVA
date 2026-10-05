# Plan: đánh giá CXR-LLaVA trên cohort Indiana frontal đầy đủ, chạy K80

- **Ngày:** 2026-10-04.
- **Trạng thái:** In progress — inventory, NLM-view selection, frozen cohort và GT labeling hoàn tất; K80 preflight và gate 10/10 PASS; full inference đang chạy.
- **Mục tiêu:** tái thực hiện phương pháp đánh giá external report generation của paper bằng checkpoint phát hành, trên cohort Indiana được dựng lại từ dữ liệu local.
- **Phần paper được đối chiếu:** Table 4, CXR-LLaVA trên Indiana. Không bao gồm training reproduction, MIMIC Table 2, CheXpert classification Table 3, các model đối chứng hoặc đánh giá radiologist.
- **Tên kết quả dự kiến:** “Indiana full-cohort evaluation of the released CXR-LLaVA checkpoint on K80”. Chỉ dùng “exact benchmark reproduction” nếu xác minh được manifest, checkpoint và phương pháp đánh giá gốc.

## 1. Cơ sở và trạng thái hiện tại

### Dữ liệu đã kiểm tra

| Nguồn | Hiện trạng local | Ý nghĩa |
|---|---|---|
| Indiana/OpenI | 3.955 XML, 7.470 PNG | Nguồn cho experiment chính |
| XML có ảnh tồn tại | 3.851 báo cáo | Chưa áp điều kiện reference/view |
| Có ảnh và FINDINGS hoặc IMPRESSION không rỗng | 3.826 báo cáo | Candidate pool; chưa phải cohort frontal cuối cùng |
| Có ảnh và cả hai section không rỗng | 3.331 báo cáo | Chỉ là thống kê; không tự đặt điều kiện bắt buộc cả hai section |
| Có nhiều ảnh tồn tại | 3.405 báo cáo | Cần chọn view, không mặc định ảnh đầu tiên là frontal |
| NIH/CXR8 | 4.999 ảnh; metadata 112.120 dòng | Có thể làm thí nghiệm bổ sung, không thay thế test set của paper |
| PadChest sample | 24 ảnh/báo cáo | Không đủ để thay benchmark; có lateral/costal |
| MIMIC | Có danh sách test tác giả trong repo; chưa có ảnh/report tương ứng trong datasets | Chưa thể chạy Table 2 |
| CheXpert test | Chưa có bộ 518 ảnh/nhãn chuẩn trong datasets | Chưa thể chạy Table 3 |

Dataset root mặc định: `../datasets/NLMCRX` tính từ project; ảnh ở `NLMCXR_png/`, XML ở `NLMCXR_reports/ecgen-radiology/`. Mọi script mới cần nhận đường dẫn qua CLI/env.

### Execution log — 2026-10-04

- Phase A đã chạy bằng `scripts/inventory_indiana.py`. Inventory đầy đủ nằm trong `result/indiana_full_k80_20261004_inventory02/` (được `.gitignore` loại khỏi Git).
- Đếm lại: 3.955 XML/report rows; 7.470 image references, tất cả file tồn tại và decode được; 3.927 reports có ID và reference không rỗng; 3.405 reports có nhiều image references; 104 reports không có image reference; 28 reports thiếu reference text.
- Không có duplicate report IDs. Có 3 nhóm duplicate image SHA-256 được ghi trong `inventory/summary.json`; chúng cần audit, không tự gộp.
- Có thêm [`frontal_final.csv`](https://data.lhncbc.nlm.nih.gov/public/chest-xray/frontal_final.csv) và [`lateral_final.csv`](https://data.lhncbc.nlm.nih.gov/public/chest-xray/lateral_final.csv) do NLM công bố; [trang nguồn NLM](https://lhncbc.nlm.nih.gov/CHRB/CHRB-resources.html) nói nhãn được tạo bằng cách xem và phân loại từng ảnh. Chúng là external human labels, không phải review thủ công mới của dự án.
- NLM liệt kê 3.864 frontal và 3.689 lateral; so với local, 3.818 ảnh chỉ ở frontal, 3.644 chỉ ở lateral, 4 ở cả hai danh sách, 4 không có nhãn. Tám ảnh mâu thuẫn/thiếu nhãn bị loại, không đoán view từ suffix/caption. PA/AP không được suy từ nhãn frontal của NLM.
- `scripts/select_indiana_nlm_views.py` chọn 3.667 report/ảnh frontal hợp lệ; 3.794 frontal candidates, 120 report có nhiều frontal, chọn image ID tăng dần và ghi 127 frontal không chọn. 288 report bị loại, tất cả có lý do. Cohort + source hashes + annotation audit ở `result/indiana_full_k80_20261004_inventory02/nlm_review_01/`.
- Pairing lấy trực tiếp từ `parentImage` của XML và mọi image ID đều có prefix đúng report UID; `manual_pairing_visual_review=false` được ghi trong config. Reference gốc có thể mô tả toàn study gồm nhiều view. Vì vậy kết quả là reconstructed cohort theo source view labels, không phải exact cohort paper hoặc new human report-pairing review.
- Official CheXpert sample gate chạy trên pinned revisions và khớp `labeled_reports.csv` (4 rows × 15 columns).
- `scripts/finalize_indiana_cohort.py` giữ vai trò cho nhánh local reviewer; nhánh NLM dùng source labels có provenance riêng và đã tạo frozen manifest.
- Official CheXpert GT-only labeling hoàn tất cho 3.667 references: sample gate PASS, row/ID/text alignment PASS. `gt_labeling/support.csv` và `provenance.json` ghi raw labels, hash và support. Support khác supplementary paper rõ rệt, ví dụ Cardiomegaly local 566 positive/1.824 negative so với paper 371/727. Nguyên nhân chính xác chưa xác định; cohort và cách lấy FINDINGS+IMPRESSION khác paper có thể góp phần. Không sửa nhãn/cohort để ép khớp.
- `bash scripts/run_indiana_full_k80.sh --check` PASS: 4 K80 còn khoảng 11,11 GiB/GPU, đúng model/labeler revisions, cohort và GT provenance.
- Gate trên manifest mới hoàn tất 10/10 ca; dừng sau ca đầu rồi resume không ghi đè checkpoint; seed khác bị từ chối. Full inference đang chạy trong `result/indiana_full_k80_20261004_031940_1252388/`. Đây là job dài, chưa có metrics cuối.

### Baseline đã chạy

- Run 10 và 50 ảnh đã hoàn tất inference, official CheXpert labeling, alignment và metrics.
- Run 50: `result/indiana_k80_50_20261004_013132_1238264/`.
- 50/50 thành công; latency trung bình 21,99 giây/ảnh; peak allocated VRAM max-device 8,87 GiB.
- Macro F1 0,3949 trên 5/7 bệnh lý defined; coverage 77/350 = 22%; 12 FP và 13 omission entries.
- Cohort cũ chọn ảnh đầu tiên theo XML, chưa xác minh view. Không dùng cohort này như frontal benchmark đã được review.

Chi tiết: [báo cáo 50 ảnh](../reports/2026-10-04_k80_indiana50_paper_pipeline_analysis.md).

## 2. Phạm vi và nguyên tắc thiết kế

1. Cohort chính lấy **toàn bộ báo cáo đủ điều kiện với ảnh frontal đã xác minh**, không cân bằng bệnh bằng quota hoặc chọn theo output model.
2. Không ép N bằng 3.689 để khớp paper; mọi khác biệt N phải có bảng exclusions và lý do.
3. Một ảnh cho mỗi report/study theo protocol local được công bố. Không suy patient ID từ tên file nếu không có metadata đáng tin cậy.
4. Giữ reference gốc; không viết lại bằng LLM hoặc sửa để khớp ảnh/model. Ghi rõ reference có thể mô tả nhiều view.
5. Chọn ảnh, protocol và config trước inference mới; không tune trên cohort test sau khi xem điểm.
6. Giữ raw labels riêng với nhãn được review. Bảng chính dùng official raw labels; mọi phân tích adjudicated phải là bảng phụ, có evidence và phiên bản riêng.
7. Run 10/50 là bằng chứng kỹ thuật và phân tích khám phá. Không ghép output cũ vào cohort mới khi khác config, selection hoặc seed policy.

Plan này là nhánh **cohort tự nhiên phục vụ đánh giá external**. Protocol [cohort phân tầng/CNN](CXR_LLaVA_Stratified_Control_Protocol.md) vẫn là experiment riêng; quota 25 positive/25 negative không áp vào cohort chính của plan này.

## 3. Cấu hình model dự kiến đóng băng

| Thành phần | Cấu hình |
|---|---|
| Checkpoint | ECOFRI/CXR-LLAVA-v2 |
| Revision | `b2224786bb90d54b1e1291171866706cfbb44e2b` |
| Architecture | Llama-2 7B + ViT-L/16 của bản phát hành |
| Inference env | cxr-llava-k80, Python 3.10, torch 1.12.1+cu102, transformers 4.36.2 |
| Precision | FP32, không quantization |
| GPU | 4 K80; CUDA_VISIBLE_DEVICES cấu hình được |
| Dispatch | 8 lớp Llama/GPU; vision/projector/embed/head ở GPU đầu |
| Preprocessing | Grayscale 512×512, normalization config gốc |
| Prompt/template | Official report helper; lưu rendered prompt/token IDs/template/source hashes |
| Generation | Helper defaults temperature=0.2, top_p=0.8; tối đa 512 new tokens theo context |
| Official labeler revision | `44ddeb363149aa657296237f18b5472a73c1756f` |
| NegBio revision | `073199e2792824740e89844a59c13d3d40ce4d23` |

FP32, multi-GPU dispatch, synchronous generation và cách giữ KV cache là compatibility adaptations đã có. Defaults generation lấy từ code phát hành, chưa xác minh là cấu hình dùng cho Table 4. Lưu khác biệt này trong mọi báo cáo.

## 4. Phase A — Inventory và manifest ứng viên

**Triển khai:** mở rộng hoặc tái sử dụng `scripts/prepare_local_indiana.py` để xuất tất cả image candidates của mỗi report trước khi chọn view. Chế độ `--all` hiện tại vẫn chọn ảnh đầu tiên, nên chỉ dùng làm inventory tham khảo, chưa đủ cho phase này.

**Các trường cần có:**

```text
report_id,image_id,image_path,report_xml_path,figure_id,caption,
findings,impression,ground_truth_report,reference_policy,
image_exists,image_mode,image_width,image_height,
image_sha256,xml_sha256,view_label,view_evidence,selection_status,exclusion_reason
```

- Giữ quan hệ image/report từ XML; phân biệt ảnh thiếu, report rỗng, XML lỗi, ảnh decode lỗi và image mode không hỗ trợ.
- Kiểm tra duplicate IDs và image hashes; ghi các trường hợp nghi duplicate để review, không tự gộp report khác nhau chỉ vì text giống nhau.
- Ghi thống kê candidate pool và các section reference hiện có. Chính sách reference phải cố định và giữ được text provenance.

**Đầu ra:** `inventory/images.csv`, `inventory/reports.csv`, `inventory/summary.json`, `inventory/exclusions.csv`.

**Hoàn tất khi:** tái giải thích được số lượng local, mapping một-nhiều của report/image và mọi trường hợp không đủ điều kiện. Không có image/report được ghép bằng phỏng đoán tên file ngoài quan hệ XML.

## 5. Phase B — Xác minh view và chọn ảnh

**Triển khai:** công cụ review local xuất contact sheets hoặc HTML và CSV annotation cho `frontal_PA`, `frontal_AP`, `frontal_unspecified`, `lateral`, `other`, `unknown`.

- Metadata chỉ được coi là evidence nếu mô tả riêng ảnh đó. Caption “PA and Lateral”, suffix `1001` hay figure `F1` không tự chứng minh frontal.
- Ca unknown/mixed cần review hình ảnh; lưu reviewer, thời điểm, evidence, phương pháp và mức chắc chắn. Machine-assisted annotation phải được ghi đúng nguồn, không ghi thành human review.
- Nếu một report có nhiều frontal hợp lệ, áp quy tắc lựa chọn ổn định đã định nghĩa trước khi đọc prediction, ví dụ image ID tăng dần trong các ảnh đã được xác minh. Ghi các ảnh không được chọn và lý do.
- Lateral/other không vào primary cohort. Nếu vẫn unknown sau review, ghi exclusion rõ ràng thay vì chuyển thành frontal.
- Xem lại các report/image pairing bất thường và reference mô tả nhiều view.

**Đầu ra:** `review/view_annotations.csv`, `review/pairing_review.csv`, `review/view_evidence/`, `selection/frontal_candidates.csv`.

**Hoàn tất khi:** mỗi ảnh được chọn có evidence frontal và report pairing; các ambiguity/exclusions đều có audit trail. Không tự tạo giá trị PASS khi review chưa diễn ra.

## 6. Phase C — Gán nhãn reference trước inference

**Khoảng trống code hiện tại:** `scripts/label_cohort.py` yêu cầu inference hoàn tất. Cần thêm workflow GT-only, dùng lại phần pin revisions, official sample gate, subprocess logging và alignment; không thay implementation official labeler.

1. Chạy official sample và đối chiếu reference labels.
2. Xuất GT headerless CSV đã quote, row map `row_index,report_id` và hash text. Label một lần/report; không nhân support do report có nhiều ảnh.
3. Label GT trong môi trường legacy riêng, giữ cột `Reports` và raw labels.
4. Kiểm tra số dòng, thứ tự, text và mapping trước join.
5. Báo positive/negative/uncertain/unmentioned cho bảy target; đối chiếu phân bố với supplementary paper và giải thích khác cohort/text policy nếu có. Không điều chỉnh nhãn để làm phân bố giống paper.
6. Review negation/positive/uncertain đáng ngờ. Lưu nhận xét riêng; mọi thay đổi ở bảng adjudicated phải có nguồn và không ghi đè raw labels.

**Đầu ra:** `gt_labeling/inputs/`, `gt_labeling/raw_labels.csv`, `gt_labeling/row_map.csv`, `gt_labeling/support.csv`, `gt_labeling/logs/`, `review/gt_review.csv`.

**Hoàn tất khi:** official sample PASS, toàn bộ selected reports có nhãn aligned và support rõ ràng. Bệnh hiếm vẫn được giữ và báo low support, không enrichment để đạt quota trong primary cohort.

## 7. Phase D — Đóng băng cohort và phương pháp

- Manifest cuối tương thích runner: `report_id,image_id,image_path,ground_truth_report`, kèm view/evidence/hashes.
- Lưu `cohort_sha256`, N thực tế, config JSON, reference policy, selection/exclusion policy và review evidence.
- Ghi số lượng qua từng bước: all XML → ảnh tồn tại → reference hợp lệ → frontal verified → dedup/review → final cohort.
- Không thêm/bớt ca dựa trên điểm model. Mọi thay đổi manifest/config tạo experiment ID mới.
- Lập bảng “paper vs local”: cohort IDs, N, view rule, reference, checkpoint, prompt, decoding, dtype, labeler, masking, aggregation, bootstrap. Đánh dấu verified/adapted/unresolved.

**Hoàn tất khi:** cohort/config đóng băng có hashes và evidence; thống kê exclusions tái lập được. Exact paper split vẫn là unresolved nếu chưa có danh sách tác giả, nhưng có thể tiếp tục dưới classification reconstructed-cohort evaluation.

## 8. Phase E — Runner dài và phục hồi khi gián đoạn

**Triển khai:** `scripts/run_indiana_full_k80.sh` đã được tạo dựa trên runner 50 ca, dùng cohort đóng băng, output root và `CUDA_VISIBLE_DEVICES`; gate và full execution đang chờ chạy.

- Preflight checkpoint, dependencies, GPU RAM, Java/labeler, paths, hashes, view/review evidence và N.
- Lưu full configs, prompt audit, package freezes, git commit/status/diff, scripts snapshot và model/source hashes.
- Load model một lần, batch=1; lưu mỗi prediction cùng status, latency và VRAM. Dùng ghi atomic/checkpoint để tránh CSV hỏng khi mất kết nối hoặc process dừng.
- Hỗ trợ resume có kiểm tra cohort/config/model hashes; không tự trộn kết quả từ attempt khác hoặc ghi đè success cũ.
- Vì generation có sampling, phải lưu/restore RNG state tại ranh giới mỗi ca để resume giữ nguyên luồng RNG. Nếu chọn deterministic seed riêng cho từng ca thay thế, công bố policy mới và kiểm tra từ gate; không gọi là tương đương run 50 dùng global seed stream.
- Ghi failures và attempts riêng; retry là thao tác có scope rõ ràng, không âm thầm retry chỉ để tăng completion.
- Khóa run để tránh hai process cùng ghi; không để hai job cùng sửa compatibility model copy hoặc chiếm cùng bốn GPU.
- Background/log/status như run 50; nohup không tự phục hồi sau server reboot.

**Gate kỹ thuật:** chạy một cohort nhỏ lấy từ final manifest trước, kiểm tra pipeline và khả năng resume/không duplicate. Run 10/50 cũ xác nhận stack nhưng không thay cho gate trên manifest/config mới.

**Hoàn tất khi:** gate, output integrity, failure handling và resume checks đều qua; output directory của full run đã được xác định.

## 9. Phase F — Full inference và official labeling

1. Chạy tất cả N ca của manifest đóng băng; theo dõi completion, errors, latency, VRAM và thời gian từng stage.
2. Chỉ cache/reuse GT labels nếu hashes/text/row map hoàn toàn khớp với final manifest. Nếu khác thì label lại.
3. Label generated reports bằng đúng official labeler/revision; giữ sample-gate evidence, logs, package freeze và input/output hashes.
4. Nếu inference có failures, ghi completion trên toàn N và denominator của phần được đánh giá; không bỏ ca lỗi khỏi báo cáo tổng thể.
5. Dừng evaluation khi text/order/ID alignment sai. Không sửa nhãn bằng positional join không được kiểm tra.

**Đầu ra:** predictions, GT/generated label inputs, labels, row maps, logs/provenance và run status.

## 10. Phase G — Metrics, coverage và so sánh Table 4

### Bảng chính theo bệnh lý

Cardiomegaly, Consolidation, Edema, Lung Opacity, Pleural Effusion, Pneumonia, Pneumothorax.

- Giữ joint-definite policy hiện tại và mô tả chính xác: GT/pred đều 0 hoặc 1; -1/NaN không đổi thành negative.
- Xuất support, coverage, TP/FP/TN/FN, precision/recall/F1 cho từng bệnh lý.
- Bảng riêng trên tất cả GT-definite pairs phải báo positive omissions, negative unknowns, uncertainty và generation failures; không diễn giải recall trên mask chọn lọc là sensitivity trên tất cả ca.
- F1 không có positive reference/prediction là undefined. Macro bỏ undefined phải kèm số bệnh lý defined; không ghi “macro bảy bệnh” khi thiếu giá trị.
- Bootstrap 1.000 lần, seed cố định; nêu resampling unit, eligibility mask và số resamples undefined. Nếu đổi từ implementation conditional từng bệnh sang bootstrap theo study, báo thành phương pháp riêng và giữ kết quả cũ để audit.

### Đối chiếu paper

- Bảng so sánh từng bệnh: F1/CI paper, local F1/CI, support, coverage, delta và các khác biệt phương pháp.
- Không coi delta là significant improvement nếu cohort/protocol chưa tương đương.
- “Average F1” trong Table 4 không bằng trung bình cộng bảy số được in. Cần xác minh aggregation từ tác giả/code; không đoán micro/weighted hoặc chọn công thức chỉ vì gần 0,62.
- Nếu xuất thêm micro/weighted metrics, đặt tên/formula rõ ràng như phân tích bổ sung, không gọi là paper average đã tái lập.

**Hoàn tất khi:** source hashes và alignment PASS, metric arithmetic đã test, bảng coverage/omissions/CI đầy đủ và mọi phương pháp chưa xác minh được nêu rõ.

## 11. Phase H — Review lỗi và báo cáo

- Review FP/FN và omissions có cùng ảnh/reference/prediction/raw labels; ghi evidence, không coi output model là ground truth.
- Dùng [experiment_analysis_template.md](../experiment_analysis_template.md), viết báo cáo mới trong `reports/`.
- Báo throughput/inference latency/labeling time/whole-pipeline time riêng; VRAM max-device không gọi là tổng bộ nhớ bốn GPU.
- So với run 10/50 chỉ để kiểm tra execution và support; không suy model improvement từ thay đổi cohort.
- Final status: completed reconstructed-cohort evaluation, partial evaluation hoặc failed kèm nguyên nhân; không tự nâng thành exact reproduction.

## 12. Cấu trúc đầu ra dự kiến

```text
result/indiana_full_k80_<run_id>/
  inventory/
  review/
  selection/
    cohort.csv
    exclusions.csv
    cohort_config.json
  gt_labeling/
  gate/
  inference/
  metrics/
  logs/
  snapshots/
  RESULTS.md
  status.txt
reports/<date>_k80_indiana_full_cohort_analysis.md
```

Tên file/interface trên là thiết kế; chưa giả định các artifact đã tồn tại.

## 13. Công việc code và kiểm chứng cần hoàn thành

| Ưu tiên | Hạng mục | Tái sử dụng | Kiểm chứng cần có |
|---|---|---|---|
| P0 | Manifest mọi candidate ảnh/report, exclusions | prepare_local_indiana.py | Missing/duplicate/empty report, multi-image mapping, CSV quoting |
| P0 | View review/export và selection có evidence | XML metadata, image viewer/contact sheets | F1/1001 không tự được coi frontal; ambiguous chưa được PASS |
| P0 | GT-only official labeling | label_cohort.py | Sample match, order/text/hash mismatch phải fail |
| P0 | Freeze manifest/config và review evidence | sha256 helpers | Không đổi cohort sau freeze mà dùng cùng run ID |
| P1 | Full runner, checkpoint/resume | k80_original.py, run_indiana50_k80.sh | Simulated interruption, RNG continuity, config mismatch, duplicate success |
| P1 | Coverage/omission/aggregation audit | exp02_reevaluate.py | Undefined/no positives, masked pairs, known confusion counts, bootstrap reproducibility |
| P1 | One-command workflow và tổng hợp output | bash runner hiện tại | Stage failure/status/logs, fresh output, env paths |
| P2 | Bảng Table 4 comparison và report | reports/template | Source linkage, method differences, no unsupported benchmark claim |

## 14. Tài nguyên và trình tự thực hiện

- Bốn K80 đã đủ chạy unquantized FP32 theo run 50. Không tự chạy nhiều model copies đồng thời.
- Ước tính từ mean 21,99 giây/ảnh: nếu N=3.826 thì khoảng 23,4 giờ riêng inference; N frontal thực tế có thể thấp hơn. Đây là ngoại suy, chưa gồm labeler/I/O/retries.
- Kiểm tra disk/RAM và lưu output liên tục trước khi khởi chạy dài.
- Thứ tự: **inventory → review views → GT labels/support → freeze → gate/resume checks → full inference → generated labels → metrics → report**.
- Không khởi chạy full inference chỉ để đủ số ảnh trước khi giải quyết selection/view và output integrity.

## 15. Điều kiện kết thúc và giới hạn còn lại

- [ ] Cohort frontal đã review, nguyên tắc chọn/loại rõ ràng, N thực tế được ghi.
- [ ] GT labels đã aligned, phân bố/support được kiểm tra.
- [ ] Config/model/prompt/reference policy đóng băng và có hashes.
- [ ] Gate kỹ thuật và resume test PASS.
- [ ] Full inference có completion/failures rõ ràng, labels official và alignment PASS.
- [ ] Metrics từng bệnh, CI, coverage/omissions và Table 4 comparison hoàn tất.
- [ ] Báo cáo theo template có nguồn evidence và hạn chế đầy đủ.
- [ ] Nếu tuyên bố exact reproduction: phải có thêm xác minh official split, checkpoint, decoding và average-F1/CI method gốc.

MIMIC hiện chưa có ảnh/report local; CheXpert test đã có sau khi plan Indiana được lập nhưng là experiment riêng. NIH/PadChest không thay thế chúng. Reproduce training hay radiologist evaluation không thuộc scope plan này.

### Follow-on Table 3 evaluation (separate experiment, 2026-10-05)

CheXlocalize test images are now available under `../datasets/CheXpert_test_labels/CheXpert/`: 668 images, of which 518 are frontal and 150 lateral. The 518 frontal image labels agree with the 500-study public `groundtruth.csv` and match the paper's supplementary positive/negative support. This is a distinct CheXpert experiment run by `scripts/chexpert_table3_k80.py` and `scripts/run_chexpert_table3_k80.sh`; it does not change Indiana's frozen cohort. The paper's exact direct-label prompt/decoding code remains unavailable, so local yes/no questions are documented as a protocol adaptation.

## 16. Tài liệu tham chiếu

- [CXR-LLaVA paper v3 và supplementary](https://arxiv.org/pdf/2310.18341v3).
- [Repo tác giả](https://github.com/ECOFRI/CXR_LLAVA).
- [Official checkpoint](https://huggingface.co/ECOFRI/CXR-LLAVA-v2).
- [Official CheXpert labeler](https://github.com/stanfordmlgroup/chexpert-labeler).
- [K80 setup đã kiểm chứng](K80_Unquantized_Setup.md).
- [Subset evaluation commands](K80_Paper_Subset_Evaluation.md).
- [Báo cáo 10 ảnh](../reports/2026-10-04_k80_indiana10_paper_pipeline_analysis.md).
- [Báo cáo 50 ảnh](../reports/2026-10-04_k80_indiana50_paper_pipeline_analysis.md).
