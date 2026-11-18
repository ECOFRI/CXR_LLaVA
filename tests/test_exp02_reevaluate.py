import math
import unittest

import numpy as np
import pandas as pd

from scripts.exp02_reevaluate import (
    align_labels,
    bootstrap_f1,
    canonicalize_predictions,
    f1_from_counts,
)


class Exp02ReevaluateTests(unittest.TestCase):
    def test_f1_zero_tp_cases(self):
        self.assertEqual(f1_from_counts(0, 1, 0), 0)
        self.assertEqual(f1_from_counts(0, 0, 1), 0)
        self.assertEqual(f1_from_counts(0, 1, 1), 0)

    def test_f1_undefined_when_no_positive_reference_or_prediction(self):
        self.assertTrue(math.isnan(f1_from_counts(0, 0, 0)))

    def test_f1_known_value(self):
        self.assertAlmostEqual(f1_from_counts(7, 7, 3), 14 / 24)

    def test_success_with_whitespace_text_becomes_failed(self):
        predictions = pd.DataFrame(
            [
                {
                    "image_id": "a",
                    "ground_truth_report": "gt",
                    "generated_report": "   ",
                    "status": "success",
                }
            ]
        )
        subset = pd.DataFrame([{"image_id": "a"}])
        canonical, _ = canonicalize_predictions(predictions, subset)
        self.assertEqual(canonical.loc[0, "canonical_status"], "failed")

    def test_two_failed_records_same_id_collapse_to_one_failed(self):
        predictions = pd.DataFrame(
            [
                {"image_id": "a", "ground_truth_report": "gt", "generated_report": "", "status": "failed"},
                {"image_id": "a", "ground_truth_report": "gt", "generated_report": "", "status": "failed"},
            ]
        )
        subset = pd.DataFrame([{"image_id": "a"}])
        canonical, conflicts = canonicalize_predictions(predictions, subset)
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical.loc[0, "canonical_status"], "failed")
        self.assertEqual(len(conflicts), 1)

    def test_alignment_fails_when_reports_are_swapped(self):
        canonical = pd.DataFrame(
            [
                {
                    "image_id": "a",
                    "report_id": "1",
                    "ground_truth_report": "gt a",
                    "generated_report": "gen a",
                    "canonical_success": True,
                },
                {
                    "image_id": "b",
                    "report_id": "2",
                    "ground_truth_report": "gt b",
                    "generated_report": "gen b",
                    "canonical_success": True,
                },
            ]
        )
        row_map = pd.DataFrame(
            [
                {"chexpert_row": 0, "report_id": "1", "image_id": "a"},
                {"chexpert_row": 1, "report_id": "2", "image_id": "b"},
            ]
        )
        gt_labels = pd.DataFrame(
            {
                "Reports": ["gt b", "gt a"],
                "Cardiomegaly": [0, 1],
                "Consolidation": [0, 1],
                "Edema": [0, 1],
                "Lung Opacity": [0, 1],
                "Pleural Effusion": [0, 1],
                "Pneumonia": [0, 1],
                "Pneumothorax": [0, 1],
            }
        )
        pred_labels = gt_labels.copy()
        with self.assertRaises(ValueError):
            align_labels(
                canonical,
                row_map,
                ["gt a", "gt b"],
                ["gen a", "gen b"],
                gt_labels,
                pred_labels.assign(Reports=["gen a", "gen b"]),
            )

    def test_bootstrap_is_repeatable(self):
        gt = np.array([1, 1, 0, 0])
        pred = np.array([1, 0, 1, 0])
        self.assertEqual(bootstrap_f1(gt, pred, 100, 42), bootstrap_f1(gt, pred, 100, 42))

    def test_all_negative_labels_have_undefined_ci(self):
        gt = np.array([0, 0, 0])
        pred = np.array([0, 0, 0])
        result = bootstrap_f1(gt, pred, 1000, 42)
        self.assertTrue(math.isnan(result["ci_low_95"]))
        self.assertEqual(result["bootstrap_undefined"], 1000)

    def test_empty_valid_pairs_do_not_resample(self):
        result = bootstrap_f1(np.array([]), np.array([]), 1000, 42)
        self.assertTrue(math.isnan(result["ci_low_95"]))
        self.assertEqual(result["ci_reason"], "no_valid_pairs")


if __name__ == "__main__":
    unittest.main()
