"""Synthetic-only checks for E4-A3A state and pairing rules."""

import json
import tempfile
import unittest
from pathlib import Path

from research.e4_a3a import ARMS, agreement, check_grid, coverage_status, paired_image_score, state


class E4A3ATests(unittest.TestCase):
    def test_label_states_remain_distinct(self):
        self.assertEqual([state(x) for x in ("0.0", "-1.0", "blank", None, "bad")], [
            "Negative", "Uncertain", "Not mentioned", "Extraction failure", "Extraction failure"
        ])
        for x in ("Not mentioned", "Uncertain", "Extraction failure"):
            self.assertIsNone(agreement(x, "Negative"))
        self.assertEqual(agreement("Negative", "Negative"), 1)

    def test_duplicate_text_does_not_collapse_report_grid(self):
        reports = [{"image_id": f"I{i}", "arm_id": a, "generated_report": "same text"}
                   for i in range(20) for a in ARMS]
        prompts = [{"image_id": f"I{i}", "arm": a} for i in range(20) for a in ARMS]
        self.assertEqual(len(check_grid(reports, prompts)), 20)
        with self.assertRaises(ValueError):
            check_grid(reports[:-1] + [reports[0]], prompts)

    def test_paired_denominator_excludes_missing_and_uncertain(self):
        refs = {("I0", "Cardiomegaly"): "Positive", ("I0", "Edema"): "Negative"}
        labels = {("I0", "C2a", "Cardiomegaly"): "Positive", ("I0", "C3", "Cardiomegaly"): "Negative",
                  ("I0", "C2a", "Edema"): "Uncertain", ("I0", "C3", "Edema"): "Negative"}
        self.assertEqual(paired_image_score("I0", "C2a", "C3", refs, labels), ("improved", 1, 1, 0))
        self.assertEqual(paired_image_score("I1", "C2a", "C3", refs, labels), ("non-evaluable", 0, 0, 0))
        self.assertEqual(coverage_status(4), "INSUFFICIENT_COVERAGE")
        self.assertEqual(coverage_status(5), "DESCRIPTIVE_ONLY")

    def test_synthetic_dry_run_file(self):
        with tempfile.TemporaryDirectory() as base:
            sample = Path(base) / "synthetic"
            sample.mkdir()
            rows = [{"image_id": f"I{i}", "arm_id": a, "generated_report": "same text"}
                    for i in range(20) for a in ARMS]
            (sample / "reports.json").write_text(json.dumps(rows))
            restored = json.loads((sample / "reports.json").read_text())
            self.assertEqual(len(check_grid(restored, [{"image_id": r["image_id"], "arm": r["arm_id"]} for r in restored])), 20)
