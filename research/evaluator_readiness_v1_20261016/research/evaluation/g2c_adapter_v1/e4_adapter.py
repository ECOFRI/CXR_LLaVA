from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ARMS = ("C0", "C1", "C2a", "C3")
DEFINITE = frozenset(("positive", "negative"))
LABEL_ALIASES = {
    "1": "positive",
    "1.0": "positive",
    "positive": "positive",
    "pos": "positive",
    "p": "positive",
    "true": "positive",
    "0": "negative",
    "0.0": "negative",
    "negative": "negative",
    "neg": "negative",
    "n": "negative",
    "false": "negative",
    "-1": "uncertain",
    "-1.0": "uncertain",
    "uncertain": "uncertain",
    "uncertainty": "uncertain",
    "u": "uncertain",
    "": "blank",
    "nan": "blank",
    "none": "blank",
    "null": "blank",
    "blank": "blank",
    "unmentioned": "blank",
    "not mentioned": "blank",
    "missing": "blank",
}


class AdapterError(ValueError):
    pass


def normalize_label(value):
    text = "" if value is None else str(value).strip().lower()
    if text not in LABEL_ALIASES:
        raise AdapterError(f"Unrecognized label: {value!r}")
    return LABEL_ALIASES[text]


def read_csv(path, required):
    path = Path(path)
    if not path.is_file():
        raise AdapterError(f"Input file missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise AdapterError(f"Empty CSV header: {path}")
        missing = set(required) - set(reader.fieldnames)
        if missing:
            raise AdapterError(f"Columns missing in {path}: {sorted(missing)}")
        return list(reader)


def distinct_records(rows, keys, dataset):
    index = {}
    for number, row in enumerate(rows, start=2):
        key = tuple((row.get(column) or "").strip() for column in keys)
        if any(not item for item in key):
            raise AdapterError(f"Blank identity in {dataset} row {number}: {key}")
        if key in index:
            raise AdapterError(f"Duplicate {dataset} key: {key}")
        index[key] = row
    return index


def require_exact_keys(observed, expected, name):
    missing = expected - set(observed)
    extra = set(observed) - expected
    if missing or extra:
        raise AdapterError(
            f"{name} grid mismatch: missing={len(missing)} {sorted(missing)[:3]}; "
            f"extra={len(extra)} {sorted(extra)[:3]}"
        )


def parse_findings(path):
    findings = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    findings = [finding for finding in findings if finding]
    if len(findings) != 5 or len(set(findings)) != 5:
        raise AdapterError("Expected exactly five unique findings from the frozen protocol")
    return findings


def load_inputs(cohort_path, findings_path, references_path, predictions_path, reports_path, id_column="image_id"):
    cohort_rows = read_csv(cohort_path, (id_column,))
    cohort_index = distinct_records(cohort_rows, (id_column,), "cohort")
    image_ids = [key[0] for key in cohort_index]
    if len(image_ids) != 20:
        raise AdapterError(f"Expected exactly 20 locked image IDs, got {len(image_ids)}")
    findings = parse_findings(findings_path)

    reference_rows = read_csv(references_path, ("image_id", "finding", "reference_label"))
    prediction_rows = read_csv(predictions_path, ("image_id", "arm_id", "finding", "prediction_label"))
    report_rows = read_csv(reports_path, ("image_id", "arm_id", "report_text"))
    references = distinct_records(reference_rows, ("image_id", "finding"), "reference labels")
    predictions = distinct_records(prediction_rows, ("image_id", "arm_id", "finding"), "prediction labels")
    reports = distinct_records(report_rows, ("image_id", "arm_id"), "reports")

    expected_references = {(image_id, finding) for image_id in image_ids for finding in findings}
    expected_predictions = {
        (image_id, arm, finding) for image_id in image_ids for arm in ARMS for finding in findings
    }
    expected_reports = {(image_id, arm) for image_id in image_ids for arm in ARMS}
    require_exact_keys(references, expected_references, "reference")
    require_exact_keys(predictions, expected_predictions, "prediction")
    require_exact_keys(reports, expected_reports, "report")

    for key, row in reports.items():
        if not (row["report_text"] or "").strip():
            raise AdapterError(f"Empty report for {key}")

    canonical_refs = {key: normalize_label(row["reference_label"]) for key, row in references.items()}
    canonical_preds = {key: normalize_label(row["prediction_label"]) for key, row in predictions.items()}
    evaluated = []
    for image_id in image_ids:
        for arm in ARMS:
            for finding in findings:
                reference_label = canonical_refs[(image_id, finding)]
                prediction_label = canonical_preds[(image_id, arm, finding)]
                evaluated.append({
                    "image_id": image_id,
                    "arm_id": arm,
                    "finding": finding,
                    "reference_label": reference_label,
                    "prediction_label": prediction_label,
                    "joint_definite": int(reference_label in DEFINITE and prediction_label in DEFINITE),
                })
    return image_ids, findings, evaluated


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_group(rows, cohort_size):
    count = len(rows)
    if count != cohort_size:
        raise AdapterError(f"Expected {cohort_size} rows per arm/finding, got {count}")

    n_tp = n_tn = n_fp = n_fn = 0
    unknown_reference = unknown_prediction = unknown_both = 0
    ref_positive = ref_negative = 0
    pred_positive = pred_negative = 0
    ref_positive_pred_unknown = 0
    pred_positive_ref_unknown = 0

    for row in rows:
        ref = row["reference_label"]
        pred = row["prediction_label"]
        if ref == "positive":
            ref_positive += 1
        elif ref == "negative":
            ref_negative += 1
        else:
            unknown_reference += 1
        if pred == "positive":
            pred_positive += 1
        elif pred == "negative":
            pred_negative += 1
        else:
            unknown_prediction += 1
        if ref not in DEFINITE and pred not in DEFINITE:
            unknown_both += 1
        if ref == "positive" and pred not in DEFINITE:
            ref_positive_pred_unknown += 1
        if pred == "positive" and ref not in DEFINITE:
            pred_positive_ref_unknown += 1
        if ref in DEFINITE and pred in DEFINITE:
            if ref == "positive" and pred == "positive":
                n_tp += 1
            elif ref == "negative" and pred == "negative":
                n_tn += 1
            elif ref == "negative":
                n_fp += 1
            else:
                n_fn += 1

    joint = n_tp + n_tn + n_fp + n_fn
    if joint != sum(row["joint_definite"] for row in rows):
        raise AdapterError("Joint-definite accounting mismatch")
    if ref_positive + ref_negative + unknown_reference != count:
        raise AdapterError("Reference accounting mismatch")
    if pred_positive + pred_negative + unknown_prediction != count:
        raise AdapterError("Prediction accounting mismatch")

    return {
        "total": count,
        "joint_definite": joint,
        "coverage": safe_ratio(joint, count),
        "reference_definite": ref_positive + ref_negative,
        "prediction_definite": pred_positive + pred_negative,
        "reference_unknown": unknown_reference,
        "prediction_unknown": unknown_prediction,
        "both_unknown": unknown_both,
        "reference_positive": ref_positive,
        "reference_negative": ref_negative,
        "prediction_positive": pred_positive,
        "prediction_negative": pred_negative,
        "reference_positive_prediction_unknown": ref_positive_pred_unknown,
        "prediction_positive_reference_unknown": pred_positive_ref_unknown,
        "tp": n_tp, "tn": n_tn, "fp": n_fp, "fn": n_fn,
        "precision": safe_ratio(n_tp, n_tp + n_fp),
        "recall": safe_ratio(n_tp, n_tp + n_fn),
        "f1": safe_ratio(2 * n_tp, 2 * n_tp + n_fp + n_fn),
        "confirmed_omission_rate_on_evaluable_positives": safe_ratio(n_fn, n_tp + n_fn),
        "omission_lower_bound_on_all_reference_positives": safe_ratio(n_fn, ref_positive),
        "omission_upper_bound_on_all_reference_positives": safe_ratio(n_fn + ref_positive_pred_unknown, ref_positive),
        "discordant_positive_rate_on_evaluable_predicted_positives": safe_ratio(n_fp, n_tp + n_fp),
    }


def compute_metrics(evaluated, image_ids, findings):
    groups = defaultdict(list)
    for row in evaluated:
        groups[(row["arm_id"], row["finding"])].append(row)
    per_finding = []
    for arm in ARMS:
        for finding in findings:
            item = {"arm_id": arm, "finding": finding}
            item.update(summarize_group(groups[(arm, finding)], len(image_ids)))
            per_finding.append(item)

    macro = []
    for arm in ARMS:
        matched = [row for row in per_finding if row["arm_id"] == arm]
        entry = {"arm_id": arm}
        for name in ("precision", "recall", "f1", "coverage"):
            valid = [row[name] for row in matched if row[name] is not None]
            entry[f"macro_{name}"] = sum(valid) / len(valid) if valid else None
            entry[f"defined_{name}_findings"] = len(valid)
        entry["total_image_finding_pairs"] = len(image_ids) * len(findings)
        entry["joint_definite_pairs"] = sum(row["joint_definite"] for row in matched)
        entry["missing_or_uncertain_pairs"] = entry["total_image_finding_pairs"] - entry["joint_definite_pairs"]
        macro.append(entry)
    return per_finding, macro


def write_csv(path, rows):
    if not rows:
        raise AdapterError(f"Cannot write empty CSV: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    images, findings, evaluated = load_inputs(
        args.cohort, args.findings, args.references, args.predictions, args.reports,
        args.cohort_id_column,
    )
    per_finding, macro = compute_metrics(evaluated, images, findings)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "normalized_labels.csv", evaluated)
    write_csv(outdir / "per_finding_metrics.csv", per_finding)
    write_csv(outdir / "macro_metrics.csv", macro)
    summary = {
        "mode": args.mode,
        "status": "PASS_SYNTHETIC_GRID" if args.mode == "synthetic" else "PASS_STRUCTURAL_VALIDATION_ONLY",
        "images": len(images),
        "arms": list(ARMS),
        "findings": findings,
        "reports": len(images) * len(ARMS),
        "reference_labels": len(images) * len(findings),
        "prediction_labels": len(evaluated),
        "joint_definite_policy": "score only pairs where reference and prediction are definite",
        "missing_policy": "blank and uncertain remain non-definite, never coerced to negative",
        "unsupported_finding_policy": "FP indicates disagreement with report-derived reference, not clinical hallucination",
        "validation_note": "Not a clinical validation or original labeler runtime validation",
    }
    (outdir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Four-arm E4 labeler-output evaluator adapter")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--cohort-id-column", default="image_id")
    parser.add_argument("--findings", required=True, help="Five lines from pinned G2-C protocol")
    parser.add_argument("--references", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--reports", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--mode", choices=("synthetic", "integration"), default="synthetic")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except (AdapterError, OSError) as error:
        parser.exit(2, f"BLOCKED: {error}\n")


if __name__ == "__main__":
    main()
