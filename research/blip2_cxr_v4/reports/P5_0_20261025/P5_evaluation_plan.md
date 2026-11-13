# P5.0 preregistered evaluation plan — 2026-10-25

Status: **planned, no P5 predictions or metrics**. Only evaluate if future P5.2 checkpoint passes integrity/reload and P5.3 obtains separate GPU approval.

## Primary comparison

On each frozen cohort separately, compare future Q-Former arm (M3) against saved P4 projection arm (M1) using **paired corpus BLEU-4 difference M3−M1**, same 500 study/image/reference IDs. No pooling of MIMIC and Indiana. Reuse saved P4 M1, M0 and Indiana X00/M2 reports; do not regenerate them. The P4 text tokenizer is `lowercase_ascii_alnum_regex_v1`, BLEU smoothing none. A 10-case M3 output validity gate must check nonempty text, exact stable IDs/reference joins, unique rows, identical prompt/decode, and image loading before full inference.

Secondary lexical metrics: BLEU-1/2/3, ROUGE-L, METEOR, report length, repetition, empty rate and 128-token truncation rate. Also report latency/peak memory separately; these are performance metrics, not clinical quality. All saved outputs require run ID, model ID, study ID, image ID and generated report text.

## Clinical and error analyses

Use exactly the P4 CheXpert/NegBio revisions and seven-label ontology. Report per-class TP/FP/FN/TN, precision, recall, F1, positive support, label coverage, and prediction/reference blank/uncertain rates. Primary historical policy retains jointly definite GT/prediction pairs. Prespecified sensitivity retains GT-definite cases and maps missing/uncertain prediction to negative, leaving GT untouched. Undefined F1 is N/A, never zero. Different model-specific historical masks must be shown, not hidden in a headline F1.

Prepare a balanced **blinded** review sheet with deidentified paired reference and M1/M3 texts, random A/B arm order, study/image IDs held in a separate key, and candidate plus control cases selected by a fixed seed. Stratify by cohort and by labeler disagreements, including unsupported-positive, omitted-positive, incorrect-negation, within-report contradiction, generic/short output, certainty, location and severity. Ask reviewers to assess the actual image and reference where authorized; without radiologists/image adjudication, report only text/labeler-derived candidates, never confirmed hallucinations or omissions. Do not export identifying data.

## Statistics and claims

Use 1,000 paired percentile bootstrap resamples, seed 42. Resample patient clusters for MIMIC (`subject_id`), study clusters for Indiana because X00 patient IDs are unavailable. Present 95% CI for the primary BLEU-4 delta on each dataset; show absolute scores and paired delta. Secondary metrics and error categories are exploratory with unadjusted intervals and multiple-comparison caution. Pretraining overlap/contamination, source-order cohort selection, labeler uncertainty, differing M1/M3 truncation, and Indiana's different X00 protocol constrain interpretation. Increased BLEU does not establish clinical improvement; Q-Former benefit requires both rigorous lexical evidence and a separate clinical review claim, if such review becomes possible.
