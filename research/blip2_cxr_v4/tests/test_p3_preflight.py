"""CPU-only checks of P3 refusal and synthetic bridge behavior."""
import sys
import unittest
from pathlib import Path

VR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VR / 'src'))

from vlm_medical.models.bridges import LoRALinear, make_bridge
from vlm_medical.models.registry import MissingCheckpointError, resolve_checkpoint
from vlm_medical.train.common import require_approval


class P3PreflightTests(unittest.TestCase):
    def test_missing_checkpoint(self):
        with self.assertRaises(MissingCheckpointError):
            resolve_checkpoint(VR / '.cache' / 'definitely_missing')

    def test_training_refused_without_approval(self):
        with self.assertRaises(PermissionError):
            require_approval(VR, 'X06_nonexistent_approval_test')

    def test_synthetic_bridges_never_initialize_cuda(self):
        import torch
        for kind in ('linear', 'query'):
            bridge = make_bridge(kind, 8, 16, 2)
            self.assertEqual(tuple(bridge(torch.zeros(1, 4, 8)).shape), (1, 2, 16))
        self.assertEqual(tuple(LoRALinear(torch.nn.Linear(8, 8))(torch.zeros(1, 8)).shape), (1, 8))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
