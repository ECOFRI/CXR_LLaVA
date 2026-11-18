# Bản đồ dự án
Đọc [STATUS](STATUS.md) trước, chỉ mở mục cần thiết. “Không” nghĩa là không bị ignore, có thể còn untracked; không đồng nghĩa đã commit.

| Đường dẫn | Nội dung | Khi nào đọc | Gitignored? |
|---|---|---|---|
| [AGENTS](../AGENTS.md), [CLAUDE](../CLAUDE.md), [README](../README.md) | Cửa vào chung, alias Claude, giới thiệu ngắn | Bắt đầu phiên | Không |
| [CRITERIA](../CRITERIA.md), [template](../experiment_analysis_template.md) | Tiêu chí an toàn/khoa học, cấu trúc report | Trước code/thí nghiệm/report | Không |
| [STATUS](STATUS.md), [DECISIONS](memory/DECISIONS.md), [LOG](memory/LOG.md) | Trạng thái thay thế, memory bền vững, nhật ký | Đầu task và cập nhật cuối task | Không |
| [MODEL_CARD](MODEL_CARD.md) | Model card upstream và usage CXR-LLaVA, chuyển từ README | Model assumptions/usage/limitations | Không |
| [report_snapshot/](report_snapshot/README.md) | 6 tab Google Doc export 13/11; giữ nguyên | So tài liệu trình bày với raw evidence | Không |
| [plan/](../plan/), [PROJECT_PLAN](../plan/PROJECT_PLAN.md), [work_orders/](../plan/work_orders/) | Kế hoạch tổng, task cụ thể; không cấp phép ngầm | Thiết kế/task tổ chức repo | Không |
| [reports/](../reports/) | Báo cáo chính E/G và baseline; nhiều link historical | Tra kết quả, không đọc cả execution log >20 KB | Không, trừ bundle ảnh/zip theo .gitignore |
| [research/](../research/README.md) | Module theo I/O, audit/data, encoder, probe, direct, prompt/eval, plot | Chọn thuật toán tái dùng; không chép lại module list | Code/docs không; artifacts tùy hàng dưới |
| research/runs/, research/cache/, research/inventory/, figures/, reviews/, tables/ | Raw evidence E/G, caches và review assets | Mở đúng artifact từ report/catalog, không liệt kê đệ quy | Có (prefix research/) |
| research/envs/, upstream/, conda_pkgs/, environment_setup/ | CPU env, upstream và package caches | Env compatibility/provenance; không recursive scan | Có (prefix research/) |
| [V4 workspace](../research/blip2_cxr_v4/README.md) | README hiện là snapshot P0 cũ; hiện trạng ở STATUS | Định vị BLIP-2 workspace | Không |
| [V4 plan/](../research/blip2_cxr_v4/plan/), [work_orders/](../research/blip2_cxr_v4/work_orders/) | Master plan V4 và WO_P0…P3; WO_<phase/task>_YYYY-MM-DD.md | Scope/protocol, phân biệt phase cấu trúc với pilot thực thi | Không |
| [V4 reports/](../research/blip2_cxr_v4/reports/) | P1–P5 progress, tracker, closeout/postmortem/recovery | Kết quả, không sửa report đã có trong WO này | Không (trừ *.log/cache) |
| [V4 src/](../research/blip2_cxr_v4/src/), [scripts/](../research/blip2_cxr_v4/scripts/), [configs/](../research/blip2_cxr_v4/configs/) | Data/model/eval/utils; audit/infer/train runners; contracts | Code reuse theo nhiệm vụ, không chạy entrypoint chỉ vì thấy plan | Không; paths.local.yaml bị ignore |
| [V4 manifests/](../research/blip2_cxr_v4/manifests/), [approvals/](../research/blip2_cxr_v4/approvals/) | Cohorts, hashes, approval contracts | Audit đúng input/quyền thực thi; không sửa trong WO này | Không theo rule hiện tại; có thể nhạy cảm, không tự stage |
| V4 runs/, results/, .cache/ | Outputs/metrics/checkpoints/cache | Mở đường dẫn evidence cụ thể, không recursive scan | Có, theo V4 .gitignore |
| V4 third_party/ | Source paper đã pin, license/weights còn blocker | Paper audit; không recursive scan hoặc sửa upstream | Không có rule prefix riêng; *.pyc/log vẫn ignore |
| result/ | Saved baseline predictions và label/metric outputs | Re-score X00, source join | Có: /result/ |
| data/, datasets/ | Local datasets, manifests/CSV | Data provenance; không in patient IDs/text | Có: /data/, /datasets/ |
| model_cache/ | Weights đã tải | Model revision/hash; không download từ task tài liệu | Có: /model_cache/ |
| [CXR_LLAVA_HF/](../CXR_LLAVA_HF/), [main.py](../main.py), [requirements](../requirements.txt) | Model code gốc, entrypoint inference, dependencies | Model loading/preprocess/prompt/decode | Không |
| [tests/](../tests/), [V4 tests/](../research/blip2_cxr_v4/tests/) | Kiểm tra theo module | Chọn test nhỏ theo thay đổi | Không (pycache ignore) |
| [IMG/](../IMG/), [test CSV](../MIMIC_CXR_DATASET_TESTSET.csv) | Ảnh mẫu và CSV không phải official test | Smoke được giao, audit split | Không |

## Môi trường: chọn interpreter theo tác vụ
| Môi trường | Dùng cho | Bằng chứng / giới hạn |
|---|---|---|
| python3 hệ thống + standard library | research.project status, test_project.py; task tài liệu | Không load torch/GPU; catalog không thay scientific decision |
| cxr-llava-k80: /home/student4/anaconda3/envs/cxr-llava-k80/bin/python | CXR-LLaVA K80 FP32, torch 1.12.1+cu102, Transformers 4.36.2; BLIP-2 theo dtype contract riêng | [E/G env](../reports/2026-10-17_E_G_execution_status_log.md), [P2 compatibility](../research/blip2_cxr_v4/reports/P2_progress.md) |
| research/envs/cxr-clear-cpu/bin/python | CLEAR CPU torch 2.3.1+cpu; audit và cache-only analysis | [P1 env](../research/blip2_cxr_v4/reports/P1_progress.md); không gộp stack với K80 |
| /home/student4/anaconda3/envs/chexpert-label/bin/python | CheXpert/NegBio và METEOR pinned scorer/WordNet | [P2.5](../research/blip2_cxr_v4/reports/P2_progress.md); scorer/runtime khác version, giữ provenance |
| Legacy R2/META env đề xuất | PLAN_ONLY, chưa cài/chạy | [P3](../research/blip2_cxr_v4/reports/P3_progress.md), weights/license chưa đủ |

Các prefix/env trên là bằng chứng môi trường server đã dùng, không bảo đảm reservation/quota hiện tại.
Dữ liệu shared ngoài repo chỉ đọc, path theo config/manifest; không suy quyền upload từ việc có file.
Rule ignore kiểm chứng từ [.gitignore root](../.gitignore) và [V4 .gitignore](../research/blip2_cxr_v4/.gitignore).
Kiểm nhanh từ root: `PYTHONDONTWRITEBYTECODE=1 python3 -m research.project status`.
