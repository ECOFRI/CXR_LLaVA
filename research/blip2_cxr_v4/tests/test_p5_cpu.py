"""CPU-only P5 contract, gradient, target, and deterministic resume tests."""
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

VR = Path(__file__).resolve().parents[1]
ROOT = VR.parents[1]
sys.path.insert(0, str(VR / 'scripts/p5'))
from p5_common import (contract_and_index, digest, expected_names,
                       restore_checkpoint, save_checkpoint, train_order)
from p5_evaluate import clinical


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Linear(2, 2)
        self.language_model = nn.Linear(4, 1)
        self.qformer = nn.Module()
        self.qformer.inner = nn.Linear(2, 2)
        self.query_tokens = nn.Parameter(torch.ones(1, 2))
        self.language_projection = nn.Module()
        self.language_projection.layer = nn.Linear(2, 4)
        self.vision_model.requires_grad_(False)
        self.language_model.requires_grad_(False)

    def forward(self, x):
        visual = self.vision_model(x)
        query = self.qformer.inner(visual) + self.query_tokens
        return self.language_model(self.language_projection.layer(query)).square().mean()


def step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(1, 2) + random.random() + float(np.random.rand())
    loss = model(x)
    loss.backward()
    assert model.qformer.inner.weight.grad is not None
    assert model.query_tokens.grad is not None
    assert model.language_projection.layer.weight.grad is not None
    assert all(p.grad is None for p in model.vision_model.parameters())
    assert all(p.grad is None for p in model.language_model.parameters())
    optimizer.step()


class P5CpuTests(unittest.TestCase):
    def test_contract_and_frozen_manifests(self):
        config = VR / 'configs/experiments/P5_qformer_contract.approved_20261025.json'
        index = VR / 'runs/P4_projection_pilot_20261023/manifest_index.approved.json'
        data, cohort = contract_and_index(config, index)
        self.assertEqual(data['model']['trainable_parameter_count'], 107130880)
        for name, count in [('mimic_train', 2000), ('mimic_validate', 500), ('indiana_x00', 500)]:
            item = cohort['manifests'][name]
            path = ROOT / item['path']
            self.assertEqual(digest(path), item['sha256'])
            with path.open() as stream:
                self.assertEqual(sum(1 for _ in stream), count)

    def test_actual_qformer_wrapper_small_cpu(self):
        from transformers import Blip2QFormerConfig
        from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel
        sys.path.insert(0, str(VR / 'src'))
        from vlm_medical.models.bridges import FP32QFormer, FP32Projection
        cfg = Blip2QFormerConfig(hidden_size=24, num_hidden_layers=2,
                num_attention_heads=4, intermediate_size=48, encoder_hidden_size=32,
                cross_attention_frequency=1)
        qformer = FP32QFormer(Blip2QFormerModel(cfg))
        query = nn.Parameter(torch.randn(1, 4, 24))
        projection = FP32Projection(nn.Linear(24, 40, dtype=torch.float16))
        result = qformer(query_embeds=query, encoder_hidden_states=torch.randn(1, 6, 32,
                dtype=torch.float16), encoder_attention_mask=torch.ones(1, 6, dtype=torch.long),
                return_dict=True).last_hidden_state
        output = projection(result)
        output.float().square().mean().backward()
        self.assertEqual(tuple(output.shape), (1, 4, 40))
        self.assertTrue(query.grad is not None and bool(query.grad.abs().sum() > 0))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in qformer.parameters()))
        self.assertTrue(all(p.grad is not None for p in projection.parameters()))

    def test_exact_parameter_groups_and_gradients(self):
        model = Tiny()
        self.assertEqual(expected_names(model), {'qformer.inner.weight', 'qformer.inner.bias',
                         'query_tokens', 'language_projection.layer.weight', 'language_projection.layer.bias'})
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        step(model, optimizer)
        self.assertTrue(torch.isfinite(model.qformer.inner.weight.grad).all())

    def test_resume_matches_uninterrupted(self):
        order = train_order(3, 42)
        def init():
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            model = Tiny()
            opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
            return model, opt
        reference, ref_opt = init()
        for _ in range(3):
            step(reference, ref_opt)
        interrupted, int_opt = init()
        step(interrupted, int_opt)
        with tempfile.TemporaryDirectory(dir=VR / '.cache') as tmp:
            checkpoint = Path(tmp) / 'state.pt'
            save_checkpoint(checkpoint, interrupted, int_opt, 'P5.2', 'config', 'manifest', order, 1, 1.5)
            resumed, resume_opt = init()
            pos, elapsed = restore_checkpoint(checkpoint, resumed, resume_opt,
                                              'P5.2', 'config', 'manifest', order)
            self.assertEqual((pos, elapsed), (1, 1.5))
            for _ in range(pos, 3):
                step(resumed, resume_opt)
            for name, tensor in reference.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed.state_dict()[name]), name)
            self.assertEqual(ref_opt.state_dict()['state'].keys(), resume_opt.state_dict()['state'].keys())

    def test_p5_2_never_loads_p5_1_state(self):
        model, optimizer = Tiny(), torch.optim.AdamW(Tiny().parameters(), lr=1e-4)
        # The stage marker rejects smoke-state promotion before tensor copying.
        with tempfile.TemporaryDirectory(dir=VR / '.cache') as tmp:
            path = Path(tmp) / 'gate.pt'
            save_checkpoint(path, model, optimizer, 'P5.1', 'config', 'manifest', [0, 1], 0, 0)
            with self.assertRaises(ValueError):
                restore_checkpoint(path, model, optimizer, 'P5.2', 'config', 'manifest', [0, 1])

    def test_clinical_sensitivity_preserves_gt(self):
        label_names = ('Cardiomegaly', 'Consolidation', 'Edema', 'Lung Opacity',
                       'Pleural Effusion', 'Pneumonia', 'Pneumothorax')
        gt = {name: '' for name in label_names}
        pred = {name: '' for name in label_names}
        gt['Cardiomegaly'] = '1.0'
        pred['Cardiomegaly'] = ''
        gt['Pneumothorax'] = '-1.0'
        pred['Pneumothorax'] = '1.0'
        rows = {(0, 'reference'): gt, (0, 'M1'): pred}
        historical = clinical([0], 'M1', rows, 'historical')
        alternative = clinical([0], 'M1', rows, 'alternative')
        self.assertIsNone(historical['per_class']['Cardiomegaly']['f1'])
        self.assertEqual(alternative['per_class']['Cardiomegaly']['fn'], 1)
        self.assertEqual(alternative['per_class']['Pneumothorax']['paired'], 0)
        self.assertEqual(alternative['per_class']['Cardiomegaly']['pred_blank'], 1)


if __name__ == '__main__':
    unittest.main()
