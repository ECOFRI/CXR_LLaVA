"""CPU-only spatial pooling checks; synthetic tensors, no MIMIC access."""
import unittest

from research.rrg_f0.core import pool_grid


class PoolTests(unittest.TestCase):
    def test_disk_then_loader_matches_direct_pool(self):
        import torch
        values = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 1024, 1).expand(1, 1024, 3)
        disk = pool_grid(values).half()
        loaded = pool_grid(disk)
        self.assertEqual(disk.shape, (1, 256, 3))
        self.assertEqual(loaded.shape, (1, 64, 3))
        self.assertTrue(torch.equal(loaded.float(), pool_grid(values, factor=4)))
        self.assertEqual(float(loaded[0, 0, 0]), 49.5)
        self.assertEqual(float(loaded[0, 1, 0]), 53.5)

    def test_nonsquare_rejected(self):
        import torch
        with self.assertRaises(ValueError):
            pool_grid(torch.zeros(1, 5, 3))

    def test_gradient_through_pool(self):
        import torch
        values = torch.ones(1, 256, 3, requires_grad=True)
        pool_grid(values).sum().backward()
        self.assertTrue(torch.equal(values.grad, torch.full_like(values, .25)))


class BridgeTests(unittest.TestCase):
    def test_target_mask_and_frozen_llm_gradient_path(self):
        import types
        import torch
        from torch import nn
        from research.rrg_f0.model import RRG

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(hidden_size=8)
                self.embedding = nn.Embedding(16, 8)
                self.head = nn.Linear(8, 16)
                self.requires_grad_(False)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, inputs_embeds, attention_mask, labels, use_cache):
                self.labels = labels
                self.mask = attention_mask
                logits = self.head(inputs_embeds.cumsum(1))
                loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1), ignore_index=-100)
                return types.SimpleNamespace(loss=loss)

        llm = TinyLM()
        model = RRG(llm)
        ids = torch.tensor([[1, 2, 3, 4]])
        loss, _, _ = model(torch.randn(1, 64, 768), torch.rand(1, 14), ids,
                           torch.ones_like(ids), torch.tensor([[-100, -100, 3, 4]]), torch.zeros(1, 14))
        loss.backward()
        self.assertEqual(llm.labels.shape, (1, 82))
        self.assertTrue((llm.labels[:, :80] == -100).all())
        self.assertTrue(torch.equal(llm.labels[:, 80:], torch.tensor([[3, 4]])))
        self.assertIsNone(llm.embedding.weight.grad)
        self.assertGreater(float(model.bridge.projector[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.bridge.finding_embeddings.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
