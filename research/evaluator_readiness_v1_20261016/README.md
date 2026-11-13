# G2-C Evaluator Adapter — standalone package

**Scope:** Synthetic smoke testing for the CXR-LLaVA × CLEAR E4 four-arm evaluation workflow. This package does not load checkpoints, produce radiology reports, or run clinical validation.

## Included tools

- `research/evaluation/g2c_adapter_v1/e4_labeler_bridge.py`: converts a **wide CheXpert-style label matrix** (one labeled report per row) into normalized long records through an **explicit report-key mapping**. Does not rely on row order.
- `research/evaluation/g2c_adapter_v1/e4_adapter.py`: validates exact cohort, reference, report and prediction grids, then calculates per-arm/per-finding metrics and coverage.
- `tests/`: 12 Python unit tests and a complete synthetic wide→long→metrics smoke run.

The code requires Python 3.8+ standard library only. It does not install or modify the existing NLP evaluator.

## Run locally or on an authorized research server

From the **root of this package**:

```bash
python -m unittest discover -s tests -v
python tests/smoke_end_to_end.py synthetic_bridge_smoke
python research/evaluation/g2c_adapter_v1/e4_adapter.py \
  --mode synthetic \
  --cohort synthetic_input/cohort.csv \
  --findings synthetic_input/findings.txt \
  --references synthetic_input/references.csv \
  --predictions synthetic_input/predictions.csv \
  --reports synthetic_input/reports.csv \
  --outdir synthetic_output
```

Expected results: 12/12 tests pass; 80 synthetic reports, 100 reference-label rows and 400 predicted-label rows; exactly 4 arms × 5 findings × 20 images.

## Data contract for actual pinned labeler outputs

| File | Required columns | Expected unique keys |
| --- | --- | --- |
| cohort CSV | `image_id` | 20 `image_id` values |
| findings TXT | one finding per line | exactly five frozen finding strings |
| reports CSV | `image_id`, `arm_id`, `report_text` | 80 `(image_id, arm_id)` |
| reference labels CSV | `image_id`, `finding`, `reference_label` | 100 `(image_id, finding)` |
| prediction labels CSV | `image_id`, `arm_id`, `finding`, `prediction_label` | 400 `(image_id, arm_id, finding)` |

Arms are exactly `C0`, `C1`, `C2a`, `C3`. Reject missing, duplicate, or extra keyed rows, missing reports and unknown label values.

Label semantics:

- Positive: `1`, `1.0`, `positive`, `pos`, `p`, `true`
- Negative: `0`, `0.0`, `negative`, `neg`, `n`, `false`
- Uncertain: `-1`, `-1.0`, `uncertain`, `uncertainty`, `u`
- Blank/unmentioned: empty, `nan`, `none`, `null`, `blank`, `unmentioned`, `not mentioned`, `missing`

The distinction between uncertain and blank is preserved in `normalized_labels.csv`. Both are **non-definite** for the primary metric denominator. Undefined precision/recall/F1 are emitted as blank CSV values, not zero.

### Wide labeler adapter

A *wide* labeler CSV must contain a source key column (`report_key` by default), and five exact finding columns. A companion mapping CSV must contain `report_key,image_id` for reference labels, or `report_key,image_id,arm_id` for predictions. The source key must uniquely join the labeler output and mapping, with no additional/missing keys. Use the actual column names and parser rules after inspecting the pinned evaluator; do **not** assume its output is already compatible.

```bash
python research/evaluation/g2c_adapter_v1/e4_labeler_bridge.py \
  --kind prediction \
  --wide-csv labeler_predictions_wide.csv \
  --mapping-csv prediction_report_key_mapping.csv \
  --findings five_findings.txt \
  --output prediction_labels_long.csv
```

For references, use `--kind reference`, a distinct mapping and output file.

## Defined metrics

Per arm and finding: TP/TN/FP/FN **only on jointly definite pairs**, precision, recall, F1, joint-definite coverage, reference/prediction unknown counts, missing-positive cases, and omission lower/upper bounds. FP describes **disagreement with report-derived reference**, not a clinically proven hallucination. Upper omission bound treats *unresolved* positives as possible omission, not definite false negatives. An additional macro table contains only defined per-finding metrics with the number of defined labels.

Synthetic data use generic findings `F01`–`F05`. **Never substitute those for the real frozen five finding names**. For actual integration, read the names from G2-C and G2-B2 artifacts and use a real 20-image cohort manifest.

## Limits of the current evidence

Local tests verify adapter logic and simulated wide labeler input only. The original CheXpert-Labeler execution, its pinned environment, its actual input/output format, and installation inside the research server have **not been independently verified** by this package. At the project gate level, leave evaluator readiness in `HOLD_SERVER_LABELER_RUNTIME_VERIFICATION` until Sol Light confirms those steps on the authorized host.

Research restrictions: no E4 inference, no new data downloads, no training, no locked-test tuning, no MIMIC access, no checkpoint changes, no claiming patient independence or clinical validity. The user has approved access to up to four K80 GPUs for a later E4 job, but this package does not allocate GPUs and does not confirm scheduler/quota availability.
