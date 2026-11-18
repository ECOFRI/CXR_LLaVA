#!/usr/bin/env python
"""Build the historical EXP-02 review queue from offline reevaluation output."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


KEYWORDS = {
    "Cardiomegaly": ["cardiomegaly", "cardiac", "heart size", "cardiomediastinal"],
    "Consolidation": ["consolidation", "consolidative", "airspace"],
    "Edema": ["edema", "vascular congestion", "pulmonary vascularity"],
    "Lung Opacity": ["opacity", "opacities", "airspace", "infiltrate"],
    "Pleural Effusion": ["effusion", "costophrenic"],
    "Pneumonia": ["pneumonia"],
    "Pneumothorax": ["pneumothorax"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_span(text: str, pathology: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    terms = KEYWORDS.get(pathology, [pathology.lower()])
    for sentence in sentences:
        lower = sentence.lower()
        if any(term in lower for term in terms):
            return sentence.strip()
    return ""


def build_review_queue(run: Path, output: Path, seed: int) -> None:
    errors = pd.read_csv(run / "review" / "indiana_f1_subset_100_label_errors_v2.csv")
    omissions = pd.read_csv(run / "review" / "indiana_f1_subset_100_omission_queue.csv")
    canonical = pd.read_csv(run / "canonical_predictions" / "indiana_f1_subset_100_canonical_predictions.csv")

    errors = errors.copy()
    errors["review_group"] = "mismatch"
    errors["valid_mask_reason"] = "joint_definite_mismatch"

    mismatch_ids = set(errors["image_id"].astype(str))
    success_ids = set(canonical.loc[canonical["canonical_success"], "image_id"].astype(str))
    candidates = sorted(success_ids - mismatch_ids)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(candidates, size=min(10, len(candidates)), replace=False).tolist())

    control_base = canonical.loc[canonical["image_id"].astype(str).isin(chosen)].copy()
    control_rows = []
    for _, row in control_base.iterrows():
        control_rows.append(
            {
                "image_id": row["image_id"],
                "report_id": row.get("report_id", ""),
                "pathology": "CONTROL_NO_JOINT_DEFINITE_MISMATCH",
                "gt_label": "",
                "pred_label": "",
                "error_type": "control",
                "ground_truth_report": row.get("ground_truth_report", ""),
                "generated_report": row.get("generated_report", ""),
                "review_group": "control",
                "valid_mask_reason": "sampled_success_without_joint_definite_mismatch",
            }
        )

    omissions = omissions.copy()
    if not omissions.empty:
        omissions["report_id"] = ""
        omissions["gt_label"] = omissions["gt_state"]
        omissions["pred_label"] = omissions["pred_state"]
        omissions["error_type"] = "omission_candidate_masked_by_label_policy"
        omissions["review_group"] = "omission"
        omissions["valid_mask_reason"] = "gt_positive_pred_uncertain_or_unmentioned"
        omissions = omissions[
            [
                "image_id",
                "report_id",
                "pathology",
                "gt_label",
                "pred_label",
                "error_type",
                "ground_truth_report",
                "generated_report",
                "review_group",
                "valid_mask_reason",
            ]
        ]

    queue = pd.concat([errors, pd.DataFrame(control_rows), omissions], ignore_index=True, sort=False)
    queue["reviewer_type"] = "agent_text_review"
    queue["reviewer"] = "Codex"
    queue["review_date_utc"] = utc_now()
    queue["image_viewed"] = False
    queue["clinical_validation_status"] = "not_clinically_validated"
    queue["gt_evidence_span"] = [
        find_span(text, pathology) for text, pathology in zip(queue["ground_truth_report"], queue["pathology"])
    ]
    queue["generated_evidence_span"] = [
        find_span(text, pathology) for text, pathology in zip(queue["generated_report"], queue["pathology"])
    ]
    queue["omission_candidate"] = queue["review_group"].eq("omission")
    queue["uncertainty_shift"] = queue["valid_mask_reason"].astype(str).str.contains("uncertain", case=False, na=False)
    queue["laterality_location_mismatch"] = ""
    queue["unsupported_statement_candidate"] = queue["error_type"].eq("FP")
    queue["resolution"] = np.where(
        queue["review_group"].eq("control"),
        "control_text_pair_sampled_for_review",
        "needs_text_review_or_clinical_review",
    )
    queue["notes"] = np.where(
        queue["review_group"].eq("omission"),
        "GT positive but generated label was uncertain or unmentioned under main metric mask.",
        "",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output, index=False)

    summary = {
        "review_date_utc": utc_now(),
        "seed": seed,
        "mismatch_rows": int(len(errors)),
        "mismatch_unique_images": int(errors["image_id"].nunique()),
        "control_rows": int(len(control_rows)),
        "omission_rows": int(len(omissions)),
        "total_rows": int(len(queue)),
        "image_viewed": False,
        "clinical_validation_status": "not_clinically_validated",
    }
    pd.DataFrame([summary]).to_csv(output.with_name(output.stem + "_summary.csv"), index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_review_queue(args.run, args.output, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
