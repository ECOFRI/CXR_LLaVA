import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "evaluation" / "g2c_adapter_v1"))
from e4_adapter import AdapterError, compute_metrics, load_inputs, normalize_label
from make_synthetic_fixture import generate


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        generate(self.base)

    def load(self):
        return load_inputs(
            self.base / "cohort.csv", self.base / "findings.txt",
            self.base / "references.csv", self.base / "predictions.csv",
            self.base / "reports.csv",
        )

    def update_csv(self, filename, callback):
        path = self.base / filename
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames
            rows = list(reader)
        callback(rows)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    def test_grid_and_reference_integrity(self):
        images, findings, rows = self.load()
        self.assertEqual(len(images), 20)
        self.assertEqual(len(findings), 5)
        self.assertEqual(len(rows), 400)
        self.assertEqual(len({(r['image_id'], r['arm_id'], r['finding']) for r in rows}), 400)

    def test_blank_uncertain_are_not_negative(self):
        self.assertEqual(normalize_label("-1"), "uncertain")
        self.assertEqual(normalize_label("NaN"), "blank")
        self.assertEqual(normalize_label("0"), "negative")
        self.assertEqual(normalize_label("1"), "positive")
        with self.assertRaises(AdapterError):
            normalize_label("probably")

    def test_confusion_and_missing_bounds(self):
        images, findings, rows = self.load()
        per_finding, macro = compute_metrics(rows, images, findings)
        c0 = next(item for item in per_finding if item['arm_id'] == 'C0' and item['finding'] == 'F01')
        self.assertEqual((c0['tp'], c0['tn'], c0['fp'], c0['fn']), (0, 14, 1, 1))
        self.assertEqual(c0['joint_definite'], 16)
        self.assertEqual(c0['coverage'], 0.8)
        self.assertEqual(c0['reference_positive_prediction_unknown'], 1)
        self.assertEqual(c0['omission_lower_bound_on_all_reference_positives'], 0.5)
        self.assertEqual(c0['omission_upper_bound_on_all_reference_positives'], 1.0)
        c2a = next(item for item in per_finding if item['arm_id'] == 'C2a' and item['finding'] == 'F01')
        self.assertEqual((c2a['tp'], c2a['tn'], c2a['fp'], c2a['fn']), (2, 16, 0, 0))
        self.assertEqual(c2a['joint_definite'], 18)
        self.assertEqual(c2a['f1'], 1.0)
        self.assertEqual(len(macro), 4)

    def test_duplicate_prediction_is_blocked(self):
        self.update_csv('predictions.csv', lambda rows: rows.append(dict(rows[0])))
        with self.assertRaisesRegex(AdapterError, 'Duplicate'):
            self.load()

    def test_missing_prediction_is_blocked(self):
        self.update_csv('predictions.csv', lambda rows: rows.pop())
        with self.assertRaisesRegex(AdapterError, 'grid mismatch'):
            self.load()

    def test_missing_report_is_blocked(self):
        self.update_csv('reports.csv', lambda rows: rows.pop())
        with self.assertRaisesRegex(AdapterError, 'grid mismatch'):
            self.load()

    def test_unknown_label_is_blocked(self):
        self.update_csv('references.csv', lambda rows: rows[0].update(reference_label='suspicious'))
        with self.assertRaisesRegex(AdapterError, 'Unrecognized'):
            self.load()

    def test_duplicate_image_is_blocked(self):
        self.update_csv('cohort.csv', lambda rows: rows.append(dict(rows[0])))
        with self.assertRaisesRegex(AdapterError, 'Duplicate'):
            self.load()

    def test_undefined_metrics_not_imputed_as_zero(self):
        images, findings, rows = self.load()
        per_finding, macro = compute_metrics(rows, images, findings)
        c0_f02 = next(item for item in per_finding if item['arm_id'] == 'C0' and item['finding'] == 'F02')
        self.assertIsNone(c0_f02['precision'])
        self.assertIsNone(c0_f02['recall'])
        self.assertIsNone(c0_f02['f1'])
        self.assertEqual(next(item for item in macro if item['arm_id'] == 'C0')['defined_f1_findings'], 1)


if __name__ == '__main__':
    unittest.main()
