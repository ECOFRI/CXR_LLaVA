"""CPU contract checks for the bounded P4 projection gate."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

VR = Path(__file__).resolve().parents[1]
ROOT = VR.parents[1]
MODULE = VR / 'scripts/train/p4_projection_gate.py'
spec = importlib.util.spec_from_file_location('p4_projection_gate', MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class P4ProjectionGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = VR / 'configs/experiments/P4_projection_contract.approved.json'
        cls.config = mod.read_contract(cls.config_path)
        cls.index = json.loads((VR / 'runs/P4_projection_pilot_20261023/manifest_index.approved.json').read_text())

    def test_approved_cohorts_and_no_patient_overlap(self):
        records = {}
        for name, expected, split in [('mimic_train', 2000, 'train'), ('mimic_validate', 500, 'validate'),
                                      ('indiana_x00', 500, 'external_x00')]:
            item = self.index['manifests'][name]
            records[name] = mod.read_manifest(ROOT / item['path'], item['sha256'], expected, split)
            self.assertEqual(len({r['image_id'] for r in records[name]}), expected)
        self.assertFalse({r['subject_id'] for r in records['mimic_train']} &
                         {r['subject_id'] for r in records['mimic_validate']})

    def test_mask_eos_and_truncation(self):
        from transformers import Blip2Processor
        checkpoint = ROOT / self.config['model']['checkpoint_dir']
        tokenizer = Blip2Processor.from_pretrained(checkpoint, local_files_only=True, use_fast=False).tokenizer
        self.assertEqual((tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id), (2, 2, 1))
        self.assertEqual(tokenizer.encode('\n', add_special_tokens=False), [50118])
        short = 'FINDINGS.\nIMPRESSION.'
        ids, labels, before, used = mod.tokenize_training(tokenizer, self.config, short)
        self.assertEqual(ids.shape, labels.shape)
        self.assertEqual(ids[0, 0].item(), 2)
        self.assertEqual(ids[0, -1].item(), 2)
        self.assertEqual(labels[0, -1].item(), 2)
        self.assertTrue((labels[0, :-(used + 1)] == -100).all().item())
        self.assertEqual(used, before)
        ids, labels, before, used = mod.tokenize_training(tokenizer, self.config, 'opacity ' * 400)
        self.assertGreater(before, used)
        self.assertLessEqual(used + 1, 128)
        self.assertLessEqual(ids.shape[1], 256)
        self.assertEqual(labels[0, -1].item(), 2)

    def test_unapproved_contract_rejected(self):
        with self.assertRaises(ValueError):
            mod.read_contract(VR / 'configs/experiments/P4_projection_contract.proposed.json')


if __name__ == '__main__':
    unittest.main()
