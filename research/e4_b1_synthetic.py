"""Isolated synthetic dry run of the E4-B1 metric and decision pipeline."""

import json
from pathlib import Path

from research.e4_b1_analyze import gate, label_state
from research.e4_b1_prepare import ARMS, OUT, SIX


def main():
    folder = OUT / "synthetic"
    folder.mkdir(exist_ok=True)
    cohort = {f"SYN{i:02d}": {f: "Positive" if i < 5 else "Negative" for f in SIX} for i in range(24)}
    records = [{"image_id": image, "arm": arm,
                "generated_report": "No acute cardiopulmonary process." if arm == "C0" else "A complete synthetic report sentence with sufficient length to pass the short-output check."}
               for image in sorted(cohort) for arm in ARMS]
    states = {(image, arm, f): (cohort[image][f] if arm.endswith("-O") else "Negative")
              for image in cohort for arm in ARMS for f in SIX}
    different = {image: 2 for image in cohort}
    arm_rows, format_rows, decision = gate(records, states, cohort, different)
    assert len(arm_rows) == 11 and len(format_rows) == 3 and label_state("") == "Not mentioned"
    assert all(r["oracle_eligible_cells"] == 144 for r in format_rows)
    result = {"synthetic_only": True, "image_count": 24, "report_count": len(records),
              "oracle_eligible_cells": [r["oracle_eligible_cells"] for r in format_rows],
              "decision": decision, "format_metrics": format_rows}
    (folder / "synthetic_inputs.json").write_text(json.dumps({"cohort":cohort,"records":records,"different":different})+"\n")
    (folder / "dry_run_result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"decision":decision,"reports":len(records),"eligible":result["oracle_eligible_cells"]}))


if __name__ == "__main__":
    main()
