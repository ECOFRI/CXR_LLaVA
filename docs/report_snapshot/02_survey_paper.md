<!-- Snapshot of Google Doc 'VLM Medical' tab 'survey paper' (t.2wwre276gft2), exported 2026-11-13. Read-only copy; source of truth is the Google Doc. -->

# Survey paper: CXR report generation (CLEAR × CXR-LLaVA)
Cập nhật 23/10/2026. Ô để trống là chờ chạy thực nghiệm; ô ghi "lấy từ paper" là số cần chép từ PDF gốc sau khi kiểm tra cùng protocol; ô ghi "Cần kiểm tra" là thông tin chưa xác minh từ bản PDF đầy đủ.
## Mục tiêu và setup
Thử ghép visual encoder của CLEAR với LLM của CXR-LLaVA, so với baseline CXR-LLaVA nguyên bản trên MIMIC-CXR findings, dùng bộ metric chung để mượn bảng eval của các paper khác.
| Thành phần | CLEAR | CXR-LLaVA |
|---|---|---|
| Vai trò trong hệ thống mới | Visual encoder | LLM + projector (khung LLaVA) |
| Image encoder | DINOv2 ViT-B/14 (with registers), contrastive image–text | ViT pre-train trên ảnh có nhãn bất thường, rồi align kiểu CLIP |
| Embedding | 768-d; concept score với 368,294 quan sát, chiếu qua SFR-Embedding-Mistral | Visual tokens qua projector vào LLaMA-2 |
| Dữ liệu train | Report MIMIC (concept bank trích từ report) | 592,580 CXR: 374,881 có nhãn, 217,699 có report |
| Eval gốc | Zero-shot, linear probe, concept bottleneck | F1 phát hiện bệnh, so GPT-4V, Gemini-Pro-Vision |
| Code / weight | HF peterhan91/CLEAR | GitHub ECOFRI/CXR_LLaVA |

Quyết định thiết kế cần ablation:
Input cho LLM: patch tokens DINOv2, concept-based embedding (vector concept score), hoặc cả hai.
Connector: MLP 2 lớp (kiểu LLaVA) hay Q-Former (kiểu BLIP-2).
Freeze: encoder frozen hay mở một phần; LLM frozen, LoRA hay full fine-tune.
Input view: chỉ frontal hay frontal + lateral.
## Bảng tổng hợp paper
13 paper chia ba nhóm: A là baseline có bảng MIMIC-CXR để mượn, B là BLIP-2/Q-Former cho CXR, C là evaluation.
| # | Paper | Nhóm | Kiến trúc | Dataset / split | Metric báo cáo | Điểm yếu (từ eval) |
|---|---|---|---|---|---|---|
| 1 | LLaVA-Rad (Nat. Comms 2025) | A | BiomedCLIP-CXR + MLP + Vicuna-7B (LLaVA v1.5) | 697K cặp ảnh–text, 7 nguồn; MIMIC-CXR findings | CheXprompt (GPT-4), RadGraph, CheXbert, NLG | Tác giả nói chưa sẵn sàng triển khai lâm sàng; CheXprompt phụ thuộc GPT-4 |
| 2 | MAIRA-1 | A | Rad-DINO (CXR encoder) + Vicuna-7B, text augmentation | MIMIC-CXR findings, frontal | BLEU-4, ROUGE-L, RadGraph F1, CheXbert, RadCliQ, Macro/Micro-F1-14 | Manual review thấy failure mode mà metric tự động không bắt được |
| 3 | MAIRA-2 | A | MAIRA-1 + lateral, prior image/report, Indication/Technique/Comparison | MIMIC-CXR + nguồn khác | RadFact (LLM-based), metric chuẩn | Cần input phong phú (prior), khó so với model chỉ có 1 ảnh |
| 4 | RayDINO | A | DINO self-supervised encoder | MIMIC-CXR test, 500 bootstrap, 95% CI | Như MAIRA-1 | Cần kiểm tra: decoder dùng cho report generation |
| 5 | R2GenGPT | A | Visual alignment module + LLaMA frozen, chỉ train 5M tham số | IU-Xray, MIMIC-CXR | Chủ yếu NLG (BLEU, METEOR, ROUGE-L, CIDEr) + CE | Cần kiểm tra: mức CE so với model fine-tune LLM |
| 6 | XrayGPT | A | MedCLIP frozen + linear + Vicuna fine-tune | MIMIC-CXR, OpenI | Bác sĩ chấm subset (>70% chính xác, 4/5) + ROUGE | Eval tự động yếu, khó so sánh trực tiếp |
| 7 | CXR-LLaVA (Eur. Radiol. 2025) | A (baseline) | ViT pre-train + LLaVA projector + LLaMA-2 | 592,580 CXR; test nội bộ + ngoại (gồm nhi) | F1 phát hiện bệnh, so GPT-4V/Gemini | Không theo protocol MIMIC findings + RadGraph/CheXbert; tác giả thừa nhận giới hạn performance |
| 8 | ClinicalBLIP (JMIR 2024) | B | InstructBLIP (Q-Former), fine-tune nhiều giai đoạn | IU X-Ray, MIMIC-CXR | Cần kiểm tra | Cần kiểm tra: có báo CE metric không |
| 9 | Vargas et al. (BRACIS 2024) | B | BLIP-2: CXAS U-Net + BioGPT frozen, chỉ train alignment | MIMIC-CXR | NLG + CE | CE cạnh tranh nhưng NLG thấp; cho rằng NLG không phải proxy tốt |
| 10 | ELIXR (Google) | B | SupCon CXR encoder + adapter kiểu BLIP-2 + PaLM 2 | CXR nội bộ + MIMIC | Zero-shot classification, VQA, QA accuracy | Không phải report generation thuần; LLM đóng |
| 11 | Yu et al., Patterns 2023 (ReXVal) | C | Metric study | 50 study MIMIC-CXR, 6 radiologist | BLEU, BERTScore, CheXbert sim, RadGraph F1, RadCliQ | Mẫu nhỏ (50 study) |
| 12 | ReXrank | C | Leaderboard | ReXGradient 10K + MIMIC-CXR, IU-Xray, CheXpert Plus | 8 metric: BLEU-2, BERTScore, SembScore, RadGraph-F1, RadCliQ, RaTEScore, GREEN, FineRadScore | Tách findings-only và findings+impression |
| 13 | GREEN | C | LM-based metric | MIMIC-CXR | Đếm lỗi có ý nghĩa lâm sàng + giải thích | Phụ thuộc LM chấm |

## Bảng eval MIMIC-CXR dùng chung
Protocol: MIMIC-CXR official test split, sinh section findings, ảnh frontal. Số của MAIRA-1, Refers, CheXzero lấy từ bảng S1 của RayDINO (median, 500 bootstrap). Các hàng Baseline và Ours chờ chạy thực nghiệm.
| Model | BLEU-4 | ROUGE-L | RadGraph F1 | CheXbert score | RadCliQ ↓ | Macro-F1-14 | Micro-F1-14 | GREEN |
|---|---|---|---|---|---|---|---|---|
| MAIRA-1† (GPT-4 augmented) | 14.2 | 28.9 | 24.3 | 44.0 | 3.10 | 38.6 | 55.7 |  |
| MAIRA-1 | 14.0 | 28.5 | 22.8 | 42.8 | 3.16 | 36.7 | 54.3 |  |
| Refers | 10.8 | 27.3 | 19.8 | 36.7 | 3.36 | 23.2 | 40.1 |  |
| CheXzero | 10.5 | 27.6 | 20.5 | 38.8 | 3.30 | 26.4 | 43.3 |  |
| LLaVA-Rad | lấy từ paper | lấy từ paper | lấy từ paper | lấy từ paper | lấy từ paper | lấy từ paper | lấy từ paper |  |
| R2GenGPT | lấy từ paper | lấy từ paper | lấy từ paper | lấy từ paper | lấy từ paper | lấy từ paper | lấy từ paper |  |
| Baseline: CXR-LLaVA (nguyên bản, zero-shot) |  |  |  |  |  |  |  |  |
| Baseline: CXR-LLaVA (fine-tune MIMIC findings) |  |  |  |  |  |  |  |  |
| Ours: CLEAR patch tokens + MLP + LLaMA-2 |  |  |  |  |  |  |  |  |
| Ours: CLEAR concept embedding + MLP + LLaMA-2 |  |  |  |  |  |  |  |  |
| Ours: CLEAR + Q-Former + LLaMA-2 |  |  |  |  |  |  |  |  |

Để đối chiếu thêm, nộp kết quả lên ReXrank sẽ có cùng 8 metric trên MIMIC-CXR, IU-Xray, CheXpert Plus và ReXGradient.
## Điểm yếu rút ra từ evaluation
Điểm yếu chung: phần lớn paper không cùng protocol (split, findings hay findings+impression, số view, có prior hay không), nên bảng không so sánh trực tiếp được.
CXR-LLaVA (baseline): chỉ báo F1 phát hiện 6 bệnh chính và so với model tổng quát; thiếu RadGraph F1, CheXbert, RadCliQ trên MIMIC findings. Đây là khoảng trống đầu tiên cần đo cho baseline.
CLEAR (encoder): reviewer lo ngại concept bank do LLM trích xuất chưa được kiểm chứng; chưa được thử làm encoder cho report generation. Cần đo xem concept embedding có giúp các bệnh hiếm (Macro-F1-14) không.
MAIRA-1: metric tự động bỏ sót failure mode mà manual review thấy; dùng làm tham chiếu chính vì có đủ bộ metric radiologist-aligned.
MAIRA-2: cải thiện nhờ thêm lateral và prior; khi so, phải ghi rõ model mình chỉ dùng một ảnh.
LLaVA-Rad: metric CheXprompt phụ thuộc GPT-4, khó tái lập với chi phí thấp; vẫn là baseline kiến trúc gần nhất.
R2GenGPT, XrayGPT: chủ yếu NLG hoặc bác sĩ chấm subset, ít CE; điểm NLG cao không bảo đảm đúng lâm sàng.
Vargas et al. (BLIP-2): CE cạnh tranh nhưng NLG thấp, gợi ý ưu tiên RadGraph/CheXbert/GREEN thay vì BLEU.
ClinicalBLIP, ELIXR: cần đọc PDF để xác định có báo CE metric và protocol nào trước khi đưa vào bảng.
Câu hỏi mở:
CXR-LLaVA sinh cả findings lẫn impression hay chỉ một phần? Ảnh hưởng cách tách section khi eval.
Rad-DINO (MAIRA) và DINOv2 của CLEAR khác nhau ở đâu về pre-training? Có thể giải thích kết quả.
Concept bank của CLEAR trích từ MIMIC report: có rò rỉ test set không?
## Kế hoạch thực nghiệm
Chạy baseline trước để có điểm yếu đo được, rồi mới đổi encoder; mỗi bước điền kết quả vào bảng eval ở trên.
Chuẩn bị MIMIC-CXR official split, trích section findings, chỉ ảnh frontal
Cài bộ metric theo ReXrank: RadGraph F1, CheXbert/SembScore, RadCliQ, GREEN, BLEU, BERTScore
Tái lập 1 số liệu đã công bố (ví dụ MAIRA-1 hoặc LLaVA-Rad) để kiểm tra pipeline eval
Baseline 1: CXR-LLaVA zero-shot trên test set
Baseline 2: CXR-LLaVA fine-tune trên MIMIC findings
Phân tích lỗi baseline theo 6 loại lỗi của ReXVal và theo từng bệnh (Macro-F1-14)
Ours 1: CLEAR patch tokens + MLP + LLaMA-2
Ours 2: CLEAR concept embedding + MLP + LLaMA-2
Ours 3: CLEAR + Q-Former + LLaMA-2
Ablation freeze (encoder frozen hoặc mở một phần; LLM LoRA hoặc frozen)
Bootstrap 95% CI cho mọi số để so với bảng RayDINO
| Thực nghiệm | Cấu hình | GPU-giờ | Ghi chú kết quả |
|---|---|---|---|
| Baseline 1 |  |  |  |
| Baseline 2 |  |  |  |
| Ours 1 |  |  |  |
| Ours 2 |  |  |  |
| Ours 3 |  |  |  |

## Sources
CLEAR (medRxiv) · CLEAR (HuggingFace) · CLEAR peer review
CXR-LLaVA (arXiv) · CXR-LLaVA (European Radiology)
LLaVA-Rad · MAIRA-1 · MAIRA-2 · RayDINO · R2GenGPT · XrayGPT
ClinicalBLIP · Vargas et al. · ELIXR
ReXVal (Patterns) · ReXrank · GREEN

