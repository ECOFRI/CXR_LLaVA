import unittest
import pandas as pd

from scripts.exp02_stratify import select_cohort
from scripts.exp02_reevaluate import TARGET_PATHOLOGIES, canonicalize_predictions


class CohortTests(unittest.TestCase):
    def pool(self):
        return pd.DataFrame([
            dict(image_id=str(i), report_id=str(i), image_path=str(i), ground_truth_report='gt',
                 **{p: int(i < 30) for p in TARGET_PATHOLOGIES}) for i in range(60)
        ])

    def test_multilabel_support_and_repeatability(self):
        a, _ = select_cohort(self.pool())
        b, _ = select_cohort(self.pool())
        self.assertTrue(a.equals(b))
        self.assertTrue(a.image_id.is_unique)
        for p in TARGET_PATHOLOGIES:
            self.assertGreaterEqual(a[p].eq(1).sum(), 25)
            self.assertGreaterEqual(a[p].eq(0).sum(), 25)

    def test_shortfall_does_not_create_smaller_cohort(self):
        pool = self.pool(); pool['Pneumonia'] = 0
        selected, support = select_cohort(pool)
        self.assertIsNone(selected)
        self.assertEqual(support.loc[support.pathology.eq('Pneumonia'), 'positive'].iloc[0], 0)

    def test_gt_conflict_stops_dedup(self):
        rows = pd.DataFrame([
            dict(image_id='a', ground_truth_report=t, generated_report='x', status='success')
            for t in ['first', 'different']
        ])
        with self.assertRaisesRegex(ValueError, 'Conflicting GT'):
            canonicalize_predictions(rows, pd.DataFrame([dict(image_id='a')]))

    def test_latest_failure_is_not_hidden_by_earlier_success(self):
        rows = pd.DataFrame([
            dict(image_id='a', ground_truth_report='gt', generated_report='ok', status='success'),
            dict(image_id='a', ground_truth_report='gt', generated_report='', status='failed'),
        ])
        canonical, _ = canonicalize_predictions(rows, pd.DataFrame([dict(image_id='a')]))
        self.assertFalse(canonical.canonical_success.iloc[0])


if __name__ == '__main__':
    unittest.main()
