"""Guard metric denominators and missing-output semantics independently of model code."""
import unittest

import numpy as np

from research.e_audit import parse, statistics


class AuditPolicyTests(unittest.TestCase):
    def test_missing_prediction_excluded_from_baseline_not_negative(self):
        gt = np.array([1., 1., 0., 0.])
        pred = np.array([1., np.nan, 1., 0.])
        baseline = statistics(gt, pred)
        sensitivity = statistics(gt, np.nan_to_num(pred, nan=0))
        self.assertEqual((baseline['covered'], baseline['fn']), (3, 0))
        self.assertEqual((sensitivity['covered'], sensitivity['fn']), (4, 1))
        self.assertAlmostEqual(baseline['f1'], 2 / 3)
        self.assertAlmostEqual(sensitivity['f1'], .5)

    def test_undefined_does_not_become_zero(self):
        metrics = statistics(np.array([0., 0.]), np.array([0., 0.]))
        self.assertTrue(np.isnan(metrics['precision']))
        self.assertTrue(np.isnan(metrics['recall']))
        self.assertTrue(np.isnan(metrics['f1']))

    def test_f1_zero_when_errors_have_nonzero_denominator(self):
        metrics = statistics(np.array([1., 0.]), np.array([0., 0.]))
        self.assertEqual(metrics['f1'], 0.)

    def test_uncertain_gt_not_evaluated(self):
        metrics = statistics(np.array([-1., np.nan, 1.]), np.array([1., 1., 1.]))
        self.assertEqual(metrics['covered'], 1)
        self.assertEqual(metrics['tp'], 1)

    def test_only_explicit_leading_binary_token(self):
        self.assertEqual(parse('  YES, finding present.'), 1)
        self.assertEqual(parse('No, not seen.'), 0)
        self.assertIsNone(parse('There is no evidence.'))
        self.assertIsNone(parse('Nobody knows.'))
        self.assertIsNone(parse(''))


if __name__ == '__main__':
    unittest.main()
