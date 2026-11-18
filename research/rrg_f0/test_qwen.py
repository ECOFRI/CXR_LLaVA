"""New-env CPU integration: tiny random Qwen, no downloaded weights or data."""
import tempfile
import unittest
from pathlib import Path


class QwenIntegration(unittest.TestCase):
    def test_lora_freeze_bridge_gradient_checkpoint_roundtrip(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from research.rrg_f0.model import RRG
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=16)
        llm = Qwen3ForCausalLM(config).requires_grad_(False)
        llm = get_peft_model(llm, LoraConfig(r=16, lora_alpha=32, target_modules='all-linear', task_type='CAUSAL_LM'))
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model = RRG(llm).train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        ids = torch.tensor([[1, 2, 3, 4]])
        loss, _, _ = model(torch.randn(1, 64, 768), torch.rand(1, 14), ids,
                           torch.ones_like(ids), torch.tensor([[-100, -100, 3, 4]]), torch.zeros(1, 14))
        loss.backward()
        self.assertGreater(float(model.bridge.projector[0].weight.grad.abs().sum()), 0)
        lora_grads = [p.grad for n, p in model.named_parameters() if 'lora_' in n]
        self.assertTrue(any(g is not None and bool(g.abs().sum() > 0) for g in lora_grads))
        self.assertTrue(all(p.grad is None for p in model.llm.parameters() if not p.requires_grad))
        optimizer.step()
        saved = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tiny.pt'
            torch.save({'trainables': saved, 'optimizer': optimizer.state_dict(), 'step': 1}, path)
            loaded = torch.load(path, map_location='cpu', weights_only=True)
            for p in model.parameters():
                if p.requires_grad:
                    p.data.zero_()
            model.load_state_dict(loaded['trainables'], strict=False)
            optimizer.load_state_dict(loaded['optimizer'])
            self.assertEqual(loaded['step'], 1)
            self.assertTrue(all(torch.equal(p, saved[n]) for n, p in model.named_parameters() if p.requires_grad))


if __name__ == '__main__':
    unittest.main()
