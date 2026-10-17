"""Read-only E4-B1 integrity check; writes a compact validation result."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/runs/20261017_E4_B1_evidence_format_gate_v1"
R4 = ROOT / "research/runs/20261017_E4_R4_controlled_resume_v1/E4_R4_generated_reports.jsonl"
EXPECTED_PROTOCOL = "a6eb9661672b65816630866ec2ad8a9676179562202c4e0b2966cebb0b599269"
ARMS = ("C0", "N1", "F1-M", "F1-S", "F1-O", "F2-M", "F2-S", "F2-O", "F3-M", "F3-S", "F3-O")
FIVE = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    cohort = list(csv.DictReader((OUT / "E4_B1_cohort.csv").open(newline="", encoding="utf-8")))
    scores = list(csv.DictReader((OUT / "E4_B1_clear_scores.csv").open(newline="", encoding="utf-8")))
    prompts = jsonl(OUT / "E4_B1_frozen_prompts.jsonl")
    reports = jsonl(OUT / "E4_B1_reports.jsonl")
    labels = list(csv.DictReader((OUT / "E4_B1_labeler_output.csv").open(newline="", encoding="utf-8")))
    windows = jsonl(OUT / "E4_B1_gpu_windows.jsonl")
    r4_ids = {r["image_id"] for r in jsonl(R4)}
    ids = {r["image_id"] for r in cohort}
    expected = {(i, a) for i in ids for a in ARMS}
    clear_manifest = json.loads((OUT / "E4_B1_clear_compute.json").read_text())
    label_manifest = json.loads((OUT / "E4_B1_labeler_manifest.json").read_text())
    model_manifest = json.loads((OUT / "E4_B1_model_integrity.json").read_text())
    smoke = json.loads((OUT / "E4_B1_smoke_result.json").read_text())
    background = json.loads((OUT / "E4_B1_background_status.json").read_text())
    starts = {r["window_id"] for r in windows if r["event"] == "start"}
    ends = {r["window_id"] for r in windows if r["event"] == "end"}
    report_keys = [(r["image_id"], r["arm"]) for r in reports]
    checks = {
        "protocol_hash": sha(OUT / "E4_B1_protocol.md") == EXPECTED_PROTOCOL,
        "cohort_hash": sha(OUT / "E4_B1_cohort.csv") == clear_manifest["cohort_sha256"],
        "clear_scores_hash": sha(OUT / "E4_B1_clear_scores.csv") == clear_manifest["scores_sha256"],
        "checkpoint_integrity": model_manifest["status"] == "PASS" and all(x["matches_R4"] for x in model_manifest["files"].values()),
        "clear_checkpoint_hash": clear_manifest["checkpoint_sha256"] == "da1617a7d47970667a7a158eabfa4c4f3798d5d03b7f98caec150d8ee6ab6a03",
        "development_cohort_24_and_disjoint": len(cohort) == len(ids) == 24 and ids.isdisjoint(r4_ids),
        "clear_grid_120": len(scores) == len({(r["image_id"], r["finding"]) for r in scores}) == 120 and {(r["image_id"], r["finding"]) for r in scores} == {(i, f) for i in ids for f in FIVE},
        "full_24_by_11_unique": len(reports) == len(set(report_keys)) == 264 and set(report_keys) == expected,
        "all_reports_success": all(r["status"] == "success" for r in reports),
        "prompts_match_runtime": len(prompts) == 264 and all((r["image_id"], r["arm"]) == (p["image_id"], p["arm"]) and r["rendered_prompt"] == p["rendered_prompt"] and r["input_ids"] == p["input_ids"] for r, p in zip(reports, prompts)),
        "labels_align": len(labels) == 264 and all(l["Reports"] == r["generated_report"] for l, r in zip(labels, reports)) and label_manifest["status"] == "PASS" and sha(OUT / "E4_B1_labeler_output.csv") == label_manifest["output_sha256"],
        "smoke_determinism": smoke["status"] == "PASS" and smoke["greedy_C0_byte_identical"],
        "gpu_windows_closed_under_cap": starts == ends and sum(r["wall_sec"] for r in windows if r["event"] == "end") < 14400,
        "background_completed": background["status"] == "COMPLETE" and background["exit_code"] == 0,
        "synthetic_dry_run": json.loads((OUT / "synthetic/dry_run_result.json").read_text())["synthetic_only"],
        "six_unit_tests": "Ran 6 tests" in (OUT / "E4_B1_tests.log").read_text() and "OK" in (OUT / "E4_B1_tests.log").read_text(),
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "images": len(ids), "reports": len(reports), "clear_score_rows": len(scores),
              "gpu_wall_sec": sum(r["wall_sec"] for r in windows if r["event"] == "end"),
              "protocol_sha256": EXPECTED_PROTOCOL,
              "reports_sha256": sha(OUT / "E4_B1_reports.jsonl"),
              "labels_sha256": sha(OUT / "E4_B1_labeler_output.csv")}
    (OUT / "E4_B1_final_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("E4-B1 final validation failed")


if __name__ == "__main__":
    main()
