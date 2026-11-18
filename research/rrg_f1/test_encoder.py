"""CPU fake-backbone check of shared F0/F1 pooling and paired score axes."""
import types
import unittest
import torch
from research.rrg_f0.encoder import FrozenClear


class EncoderTest(unittest.TestCase):
    def test_batch_pooling_and_direct_scores(self):
        encoder = object.__new__(FrozenClear)
        torch.manual_seed(5)
        patch = torch.randn(2, 1024, 768)
        cls = torch.randn(2, 768)
        encoder.model = types.SimpleNamespace(visual=types.SimpleNamespace(
            backbone=types.SimpleNamespace(forward_features=lambda _: {'x_norm_patchtokens': patch, 'x_norm_clstoken': cls}),
            projection=torch.nn.Identity()))
        encoder.device = 'cpu'
        encoder.text = torch.nn.functional.normalize(torch.randn(28, 768), dim=-1)
        pooled, scores = encoder.encode(torch.zeros(2, 3, 448, 448))
        expected = patch.reshape(2, 16, 2, 16, 2, 768).mean((2, 4)).reshape(2, 256, 768).half()
        self.assertTrue(torch.equal(pooled, expected))
        cosine = torch.nn.functional.normalize(cls, dim=-1) @ encoder.text.T
        wanted = torch.stack([cosine[:, 2*j:2*j+2].softmax(-1)[:, 0] for j in range(14)], dim=1)
        self.assertTrue(torch.equal(scores, wanted))


if __name__ == '__main__':
    unittest.main()
