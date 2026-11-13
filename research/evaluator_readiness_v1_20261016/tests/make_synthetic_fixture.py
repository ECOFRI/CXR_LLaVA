from __future__ import annotations

import csv
from pathlib import Path

ARMS = ("C0", "C1", "C2a", "C3")
FINDINGS = ("F01", "F02", "F03", "F04", "F05")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate(base):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    images = [f"SYNTH_{number:03}" for number in range(1, 21)]
    write_csv(base / "cohort.csv", [{"image_id": image} for image in images])
    (base / "findings.txt").write_text("\n".join(FINDINGS) + "\n", encoding="utf-8")
    references = []
    predictions = []
    reports = []
    for index, image_id in enumerate(images, start=1):
        for finding in FINDINGS:
            reference_label = "negative"
            if finding == "F01":
                reference_label = {1: "positive", 2: "uncertain", 3: "blank", 4: "positive"}.get(index, "negative")
            references.append({"image_id": image_id, "finding": finding, "reference_label": reference_label})
        for arm in ARMS:
            reports.append({"image_id": image_id, "arm_id": arm, "report_text": "Synthetic report for evaluator only."})
            for finding in FINDINGS:
                prediction_label = "negative"
                if finding == "F01":
                    if arm == "C0":
                        prediction_label = {4: "blank", 5: "positive", 6: "uncertain"}.get(index, "negative")
                    elif arm == "C1":
                        prediction_label = {1: "positive", 4: "blank", 5: "positive"}.get(index, "negative")
                    elif arm == "C2a":
                        prediction_label = {1: "positive", 4: "positive"}.get(index, "negative")
                    else:
                        prediction_label = {1: "uncertain", 4: "negative", 6: "positive"}.get(index, "negative")
                predictions.append({
                    "image_id": image_id,
                    "arm_id": arm,
                    "finding": finding,
                    "prediction_label": prediction_label,
                })
    write_csv(base / "references.csv", references)
    write_csv(base / "predictions.csv", predictions)
    write_csv(base / "reports.csv", reports)


if __name__ == "__main__":
    import sys
    generate(sys.argv[1])
