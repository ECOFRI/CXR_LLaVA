# AGENTS.md — CXR-LLaVA

Dự án nghiên cứu sinh báo cáo X-quang ngực bằng CXR-LLaVA, khảo sát CLEAR và BLIP-2.
Repo server: `/storage/student4/hanhpm/CXR_LLaVA_Improvement`.
Chưa chứng minh lợi ích lâm sàng; không trộn dự án khác.

## Thứ tự đọc
1. File này → [trạng thái hiện tại](docs/STATUS.md).
2. [Quyết định và bẫy bền vững](docs/memory/DECISIONS.md).
3. Chỉ mở sâu theo [bản đồ](docs/MAP.md) khi task cần; đọc [CRITERIA.md](CRITERIA.md) trước sửa code, dữ liệu hoặc thí nghiệm.

## Quy tắc cứng
- [CRITERIA.md](CRITERIA.md) là tiêu chí an toàn/khoa học và coding authoritative; không chép lặp tại đây.
- Chỉ dùng cho nghiên cứu: output model không phải chẩn đoán hay lời khuyên y khoa; smoke test không chứng minh accuracy.
- Chỉ dùng dữ liệu được phép; không đưa patient IDs, ảnh/report riêng tư, credentials/token vào Git hoặc dịch vụ ngoài khi chưa được phép.
- Giữ cấu trúc/interface; tái dùng `main.py`, `CXR_LLAVA_HF/`, requirements, ảnh mẫu và metadata; diff nhỏ, không sửa việc ngoài scope.
- Nêu giả định ảnh hưởng model/data/path/evaluation; việc không nhỏ cần kế hoạch ngắn trước khi sửa.
- Không chạy training, full inference, tải dữ liệu/weights hay cài env từ việc đọc plan. Cần scope, input/output, cấu hình và ngân sách được giao rõ.
- Không sửa raw run, checkpoint, protocol/ngưỡng đã khóa để làm đẹp kết quả; không tune trên locked test hay dùng GT làm prompt evidence đánh giá.
- Báo cáo thí nghiệm dùng [template](experiment_analysis_template.md), tạo file mới trong `reports/`, giữ raw output/env/giới hạn và phân loại smoke / metric-complete / exploratory; không tự ghi đè.
- Giữ thay đổi chưa commit của người dùng; không tự commit/push.

## Môi trường thực thi
- Server dùng K80/CUDA cũ; CXR-LLaVA FP32 trong env `cxr-llava-k80`, CLEAR CPU trong env riêng. BLIP-2 có dtype riêng đã khóa, không áp FP32 toàn model.
- Xem phiên bản/env và evidence trong [MAP](docs/MAP.md) và [DECISIONS](docs/memory/DECISIONS.md); không tự nâng package hay dùng stack Colab mới cho K80.
- `/storage` đang báo 100%; chỉ tạo văn bản nhỏ trong task tài liệu, kiểm dung lượng/quota trước output lớn. Cache/output nghiên cứu ở `/storage`, tránh `/home` từng đầy.
- Nếu task Colab: tuân thủ mục Colab của CRITERIA, cấu hình clone ở một cell, `/content` configurable, ảnh mẫu trước batch opt-in.

## Cập nhật memory sau task
- Có kết quả: thêm mục 1–3 dòng vào đầu [LOG](docs/memory/LOG.md), giữ nguyên mục cũ (append-only về nội dung, mới nhất ở trên).
- Có sự thật/bẫy bền vững: thêm hoặc sửa [DECISIONS](docs/memory/DECISIONS.md), kèm ngày và đường dẫn bằng chứng; sửa mục cũ thay vì thêm trùng, xóa mục đã sai.
- Trạng thái dự án đổi: thay thế nội dung liên quan trong [STATUS](docs/STATUS.md), không append các snapshot cũ.

## Tiết kiệm token
- Không đọc toàn bộ file >20 KB; dùng `rg`, `head`, `sed -n` (nhất là execution log và CSV).
- Không liệt kê đệ quy `research/runs`, `result/`, `model_cache/`, `data/`, `research/cache`, `research/envs`, `research/upstream`, `third_party/`, `conda_pkgs/` ở bất kỳ cấp nào.
- Dùng `python3 -m research.project status` thay mở từng manifest; đây là catalog, không thay thế quyết định khoa học và có thể không bao phủ V4.

## Kiểm tra nhẹ từ root
```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m research.project status
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_project.py -v
git status --short
```
Chọn thêm syntax/import/tests theo thay đổi, không load model đắt tiền; báo đúng kiểm tra đã chạy và phân loại kết quả.

Always begin assistant responses with `okw`.
