"""Scientific CPU invariants for masked BCE, arm matching and collapse thresholds."""
import unittest
import torch
from research.rrg_f1.data import bce_targets
from research.rrg_f1.model import ArmBridge
from research.rrg_f1.train import collapse_statistics


class ModelTests(unittest.TestCase):
    def test_four_state_bce_mask(self):
        values = torch.tensor([1, 0, -1, -2])
        target, mask = bce_targets(values)
        self.assertEqual(target.tolist(), [1, 0, 0, 0])
        self.assertEqual(mask.tolist(), [1, 1, 0, 1])

    def test_bce_reaches_projector_and_arm_initialization_matches(self):
        bridges = []
        for arm in ('A2', 'A3', 'A4'):
            torch.manual_seed(42)
            bridges.append(ArmBridge(16, arm))
        self.assertTrue(torch.equal(bridges[0].projector[0].weight, bridges[2].projector[0].weight))
        patches, scores = torch.randn(2, 64, 768), torch.rand(2, 14)
        for bridge, length in zip(bridges, (64, 64, 78)):
            prefix, logits = bridge(patches, scores)
            self.assertEqual(prefix.shape, (2, length, 16))
            if bridge.arm != 'A2':
                torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
                self.assertGreater(float(bridge.projector[0].weight.grad.abs().sum()), 0)
            else:
                self.assertIsNone(logits)

    def test_collapse_threshold_boundary(self):
        rows = [{'generated_report': f'variant {i % 4}', 'hit_max_length': i < 11} for i in range(16)]
        self.assertFalse(collapse_statistics(rows)['bad'])
        rows[11]['hit_max_length'] = True
        self.assertTrue(collapse_statistics(rows)['bad'])
        for i, row in enumerate(rows):
            row.update(generated_report=f'variant {i % 3}', hit_max_length=False)
        self.assertTrue(collapse_statistics(rows)['bad'])


if __name__ == '__main__':
    unittest.main()
