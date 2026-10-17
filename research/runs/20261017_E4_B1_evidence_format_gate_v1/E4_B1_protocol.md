# E4-B1 evidence-format gate — frozen development protocol

**Classification:** development pilot for paper planning; not a clinical evaluation, independent benchmark, or estimate of CLEAR benefit. This protocol is frozen and SHA-256 recorded before CLEAR/CXR-LLaVA inference. No wording, selection or threshold changes after gate results.

## Cohort and selection

Source: the 3,667-image local Indiana frontal baseline manifest `result/indiana_full_k80_20261004_031940_1252388/inference/cohort.csv` and its aligned reference-report CheXpert labels (`raw_labels.csv`, SHA-256 `e96c71d6d488c4329c968c0096723b16c8d2da49db28da26da07d64fbf485673`). Exclude the 20 E4-R4 image IDs. Selection seed `2026101701`. Shuffle eligible rows, greedily take a seeded random choice among rows maximizing still-needed positive findings, target at least four positives for each of six findings; then fill to 24 with seeded random rows having no positive among the six. The selected cohort is immutable `E4_B1_cohort.csv`, SHA-256 `74f913413fbedca41c8c6592d79105946e2f72221d88be046fcfd9e5602fdce7`. It contains IDs, six reference-label states, image hashes and selection reasons. Selected support: Atelectasis 4, Cardiomegaly 5, Consolidation 4, Edema 4, Pleural Effusion 6, Pneumothorax 4; 13 neutral-fill images. There is no four-positive shortfall. Each candidate image hash and report-to-label alignment is checked. Indiana was used in the baseline and is reported to have appeared in CLEAR's published evaluation; this cohort is development material, not independent testing. The locked CheXpert test and prior E4 20 images are excluded.

## Evidence and seeded control

CLEAR scoring uses the existing local `peterhan91/CLEAR` checkpoint (`best_model.pt`, expected SHA-256 `da1617a7d47970667a7a158eabfa4c4f3798d5d03b7f98caec150d8ee6ab6a03`), CPU FP32, official 448×448 preprocess, and E2's lower-case positive prompt / `no ` + lower-case negative prompt. Five scored findings, in order: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion. For each image/finding save positive cosine, negative cosine and unscaled paired softmax. Direction is `favors present` iff positive cosine > negative cosine; equality favors absent. These are uncalibrated similarities, not probabilities or image truth. No CLEAR score exists for Pneumothorax; that sixth finding is always marked unavailable in matched/shuffled evidence.

Derangement seed `2026101702`: shuffle the sorted 24 image IDs with `random.Random(seed)`, assign each shuffled ID the next in cyclic order as its donor. This is a bijection with no self-donor. Store donor IDs and number of differing **five-finding directions** per image. Use the exact donor's scores; never use labels to set the permutation.

Oracle states come only from the image's Indiana **reference report**: Positive, Negative, Uncertain, Not mentioned for all six findings. Oracle is a text-leakage diagnostic upper bound and never a model capability or performance result. Reference report may describe lateral views absent from the frontal input.

## Original chat and exact frozen templates

System message, unchanged:

```text
You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides.
```

User message base, unchanged:

```text
<image>
Write a radiologic report on the given chest radiograph, including information about atelectasis, cardiomegaly, consolidation, pulmonary edema, pleural effusion, and pneumothorax.
```

Append exactly `\n<context>\n` + the specified block + `\n</context>` in the same user turn; no neutral `x` padding. C0 uses the base alone. N1 uses this fixed natural-language block, with no target pathology terms:

```text
Please write a clear and complete description of the supplied image. Use ordinary prose and a consistent order. Describe what is visible without adding assumptions. Keep each sentence readable and avoid repetition. If an area is difficult to assess, say so plainly. Do not use information outside this image. Finish with a concise summary in the same style. Make the opening sentence direct and keep the remaining description orderly. Prefer ordinary wording that a reader can follow without ambiguity. If the image is unclear, describe that limitation plainly.
```

Tokenizer-only preflight estimated N1 near a five-present F2 prompt (about 221 input tokens); actual token counts are persisted per arm and reported, rather than claiming exact per-case equality.

F1-M/F1-S reproduce the **E4-R4 numeric block exactly**, with no padding:

```text
Auxiliary CLEAR paired similarity scores; not calibrated disease probabilities. Use the image; these scores do not establish diagnoses.
finding | positive cosine | negative cosine | paired positive score
Atelectasis | {exact positive cosine CSV string} | {exact negative cosine CSV string} | {exact paired softmax CSV string}
Cardiomegaly | {same fields}
Consolidation | {same fields}
Edema | {same fields}
Pleural Effusion | {same fields}
```

F1-O uses the same `<context>` table presentation but **categorical oracle states**, because fabricating numeric cosine values from report labels is prohibited:

```text
Reference-report labels for a diagnostic prompt test; they are not image truth. Use the image if they conflict with it.
finding | reference-report state
Atelectasis | {Positive / Negative / Uncertain / Not mentioned}
Cardiomegaly | {state}
Consolidation | {state}
Edema | {state}
Pleural Effusion | {state}
Pneumothorax | {state}
```

This F1-O schema differs from F1-M/S in numeric versus categorical content; F1 oracle responsiveness cannot isolate numeric formatting alone. It is reported as a limitation.

F2-M/F2-S block:

```text
Auxiliary evidence is uncertain and is not a diagnosis. Use the image as primary source.
{one sentence per scored finding, in fixed order}
No auxiliary estimate is available for pneumothorax; assess the image directly.
```

Each scored finding uses exactly `Auxiliary evidence suggests {lower-case finding} may be present.` for `favors present`, or `Auxiliary evidence suggests no {lower-case finding}.` for `favors absent`. F2-O uses the same preamble and one sentence per **six** findings: `The reference report states {lower-case finding} is present.` for Positive; `The reference report states no {lower-case finding}.` for Negative; `The reference report is uncertain about {lower-case finding}.` for Uncertain; `The reference report does not mention {lower-case finding}.` for Not mentioned. The oracle contains no invented states.

F3-M/S/O is exactly its F2 counterpart followed, inside the same context block, by:

```text
State your own assessment of atelectasis, cardiomegaly, consolidation, edema, pleural effusion, and pneumothorax explicitly; rely on the image if auxiliary evidence conflicts with it.
```

Arms per image: C0, N1, F1-M, F1-S, F1-O, F2-M, F2-S, F2-O, F3-M, F3-S, F3-O (11×24=264 full-run reports). Preserve the same frontal image in all arms. Smoke reports are separate and **not reused** in gate metrics.

## Decoding and exact records

Use existing `cxr-llava-k80` environment, original FP32 checkpoint revision `b2224786bb90d54b1e1291171866706cfbb44e2b`, one four-device map matching E4-R4, batch 1. Native `generate_cxr_repsonse(chat,image,temperature=0,top_p=0.8)` sets `do_sample=False`; assert effective generation kwargs before each call and record `generation_config`. No manual seed is needed for greedy; record seed policy `none`. Keep native 512 maximum new tokens, `</s>` keyword stop and EOS behavior. Add a read-only wrapper around `model.generate` to capture output token IDs and whether native keyword stop fired; do not alter model math. For every attempt save image/arm, exact chat, rendered prompt, `input_ids`, input count, output text and token IDs/count, stop reason (`EOS`, `STOP_KEYWORD`, `MAX_TOKENS`, or `UNDETERMINED` if artifacts cannot distinguish), wall time, errors and hardware telemetry. Save records atomically. If exact stop reason cannot be determined, mark it `UNDETERMINED`, never infer it from short text. Keep model loaded for sequential calls; no parallel process.

## Gate metrics and precommitted decision

Run the installed offline rule-based CheXpert labeler on full-run output texts only; preserve Positive, Negative, Uncertain, Not mentioned, Extraction failure for six findings. No blank→Negative conversion. Generic normal exact strings after trim are `No acute cardiopulmonary process.` and `No evidence of acute cardiopulmonary disease.`. Collapse is output under 60 characters **or** exactly one of those strings. Per arm report collapse count/24 and mean count of findings per image with state Positive, Negative or Uncertain (Extraction failure is not a valid mention). For each format, restrict M/S sensitivity to images with at least one differing scored CLEAR direction; numerator is number with any differing six-finding labeler state, denominator is eligible images. Oracle following compares O and S on the same image–finding cells whose oracle reference state is Positive/Negative; report numerator/denominator and percentage-point difference. Oracle missing/uncertain states are excluded, never negative.

Format passes **only if all four**: O and M collapse rates each ≤20%; M mention mean ≥4/6; O oracle following exceeds S by ≥20 percentage points with ≥20 eligible cells; M/S evidence sensitivity ≥30% with ≥10 eligible images. If several pass, choose highest O oracle following, then lower O collapse, then F2 before F3 before F1. These are development selection thresholds, not clinical evidence. Report raw counts only; no CIs or significance tests.

Order of outcomes: integrity/determinism failure→`E4_B1_BLOCKED`; no format passes and no format reaches the oracle 20-point condition→`E4_B1_PROMPT_ROUTE_REJECTED` **in this setup only**; no format passes but some reaches oracle condition→`E4_B1_ORACLE_RESPONSIVE_CLEAR_NOT_USABLE`; else→`E4_B1_FORMAT_SELECTED_<F#>`. Keep `HOLD_MORE_EVIDENCE` and `HOLD_COMPLEX_FUSION`. Do not launch E4-B2. No image-grounded correctness is inferred from labeler output.

## Staging, resource bounds and stopping

First score the 24 dev images with CLEAR CPU so two smoke images have their matched/shuffled blocks; this dependency occurs after protocol freeze. Smoke: first two selected IDs in sorted order, C0 twice each with byte-identical greedy output required; then 11 arms on those two images. Record all fields and OOM checks. Smoke outputs are stored separately. Project the 264-call wall time using measured per-report time plus model load, labeler and safety margin. If projected total CXR-LLaVA wall time including smoke exceeds four hours, stop before full run. Recheck `nvidia-smi` before GPU launch; choose only four devices with no other processes. One process at a time. Full run may resume only with identical hashes/config, recording all interruptions; cumulative CXR-LLaVA wall time including smoke ≤14,400 s. Stop on mismatch, OOM, unavailable device or time cap. No inference on E4-R4 images or locked CheXpert test. Qualitative display uses the first five cohort IDs in sorted order, fixed here.

Verify cohort, score, checkpoint and protocol hashes; use unit tests for templates, derangement, oracle mapping, five-state label handling and denominators; run isolated synthetic end-to-end validation. Keep every artifact under `/storage/student4/hanhpm/`, including caches and temporary files. No download or package installation. Record GPU UUIDs, peak VRAM, CPU/GPU wall use and actual local Git HEAD in manifest. An E4-B2 recommendation may include the count of additional reference-positive cases needed for 15 per finding, arms C0/N1/selected M/S (oracle optional) and measured K80 time; it is never execution authorization.
