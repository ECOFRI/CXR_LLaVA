"""CPU resume and checkpoint identity checks."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

VR = Path(__file__).resolve().parents[1]
mod_path = VR / 'src/vlm_medical/train/p4_checkpoint.py'
spec = importlib.util.spec_from_file_location('p4_checkpoint', mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_projection = nn.Module()
        self.language_projection.layer = nn.Linear(3, 2)
        self.frozen = nn.Linear(2, 2)
        self.frozen.requires_grad_(False)


class CheckpointTests(unittest.TestCase):
    def test_exact_resume_and_identity(self):
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.language_projection.parameters(), lr=.01)
        loss = model.language_projection.layer(torch.ones(1, 3)).sum()
        loss.backward()
        optimizer.step()
        order = [2, 0, 1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            mod.save_state(path, model, optimizer, 'config', 'manifest', 1, 1, order, 12.5)
            other = TinyModel()
            other_optimizer = torch.optim.AdamW(other.language_projection.parameters(), lr=.01)
            position, elapsed = mod.resume_state(path, other, other_optimizer, 'config', 'manifest', order)
            self.assertEqual((position, elapsed), (1, 12.5))
            for name in mod.ALLOWLIST:
                self.assertTrue(torch.equal(dict(model.named_parameters())[name], dict(other.named_parameters())[name]))
            self.assertEqual(len(other_optimizer.state), 2)
            with self.assertRaises(ValueError):
                mod.resume_state(path, other, other_optimizer, 'wrong', 'manifest', order)
            with self.assertRaises(ValueError):
                mod.resume_state(path, other, other_optimizer, 'config', 'manifest', [0, 2, 1])
            payload = torch.load(path, map_location='cpu')
            payload['adapter']['unexpected'] = torch.zeros(1)
            with self.assertRaises(ValueError):
                mod.load_adapter(other, payload, 'config', 'manifest')
            payload['adapter'].pop('unexpected')
            payload['stage'] = 'P4.1'
            with self.assertRaises(ValueError):
                mod.load_adapter(other, payload, 'config', 'manifest')


if __name__ == '__main__':
    unittest.main()
