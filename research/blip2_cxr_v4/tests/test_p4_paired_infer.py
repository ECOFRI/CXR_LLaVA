"""CPU stable-ID, official validation, and paired-resume checks."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

VR = Path(__file__).resolve().parents[1]
p = VR / 'scripts/experiments/p4_paired_infer.py'
spec = importlib.util.spec_from_file_location('p4_paired_infer', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class PairedInferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = json.loads((VR / 'runs/P4_projection_pilot_20261023/manifest_index.approved.json').read_text())

    def test_validate_and_indiana_cohorts(self):
        rows = mod.paired_cohort(self.index)
        self.assertEqual(len(rows), 1000)
        self.assertEqual(len({mod.stable_key(r) for r in rows}), 1000)
        self.assertEqual([r['split'] for r in rows[:5]], ['validate'] * 5)
        self.assertEqual([r['split'] for r in rows[5:10]], ['external_x00'] * 5)
        self.assertEqual(sum(r['split'] == 'validate' for r in rows), 500)
        self.assertFalse(any(r['split'] == 'test' for r in rows))

    def test_saved_pairs_require_stable_key_and_both_models(self):
        rows = mod.paired_cohort(self.index)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pairs.jsonl'
            pair = mod.paired_record(rows[0], 'run', 'c', 'a',
                {'M0': {'model_id': 'M0', 'valid_nonempty': True},
                 'M1': {'model_id': 'M1', 'valid_nonempty': True}})
            path.write_text(json.dumps(pair) + '\n')
            self.assertEqual((pair['run_id'], pair['study_id'], pair['image_id']),
                             ('run', rows[0]['study_id'], rows[0]['image_id']))
            self.assertEqual(len(mod.previous_pairs(path, rows, 'run', 'c', 'a')), 1)
            pair['outputs']['M1']['model_id'] = 'M0'
            path.write_text(json.dumps(pair) + '\n')
            with self.assertRaises(ValueError):
                mod.previous_pairs(path, rows, 'run', 'c', 'a')
            pair['outputs']['M1']['model_id'] = 'M1'
            pair['run_id'] = 'wrong-run'
            path.write_text(json.dumps(pair) + '\n')
            with self.assertRaises(ValueError):
                mod.previous_pairs(path, rows, 'run', 'c', 'a')
            pair['run_id'] = 'run'
            pair['stable_key'][-1] = 'wrong-image'
            path.write_text(json.dumps(pair) + '\n')
            with self.assertRaises(ValueError):
                mod.previous_pairs(path, rows, 'run', 'c', 'a')
            pair['stable_key'] = list(mod.stable_key(rows[0]))
            pair['outputs'].pop('M1')
            path.write_text(json.dumps(pair) + '\n')
            with self.assertRaises(ValueError):
                mod.previous_pairs(path, rows, 'run', 'c', 'a')

    def test_budget_amendment_is_bound_to_frozen_run_and_config(self):
        with tempfile.TemporaryDirectory(dir=VR / '.cache') as directory:
            path = Path(directory) / 'amendment.json'
            record = {'run_id': 'run', 'config_sha256': 'config', 'manifest_index_sha256': 'index',
                      'original_budget_seconds': 7200.0, 'additional_budget_seconds': 10800.0,
                      'new_cumulative_budget_seconds': 18000.0, 'completed_pairs_at_amendment': 474,
                      'remaining_studies_by_dataset': {'mimic': 31, 'indiana': 495},
                      'authorization': 'user 2026-10-25: hoàn tất 4.3 đi',
                      'prior_ledger_sha256': 'a' * 64, 'prior_pairs_sha256': 'b' * 64}
            path.write_text(json.dumps(record))
            self.assertEqual(mod.budget_amendment(path, 'run', 'config', 'index', 7200), record)
            with self.assertRaisesRegex(ValueError, 'identity or authorization'):
                mod.budget_amendment(path, 'run', 'other-config', 'index', 7200)
            record['remaining_studies_by_dataset']['indiana'] = 494
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'identity or authorization'):
                mod.budget_amendment(path, 'run', 'config', 'index', 7200)

    def test_single_sequence_eos_matches_native_greedy_on_cpu(self):
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList, OPTConfig, OPTForCausalLM

        class ForceToken(LogitsProcessor):
            def __init__(self, token):
                self.token = token

            def __call__(self, input_ids, scores):
                scores[:] = -float('inf')
                scores[:, self.token] = 0
                return scores

        class ForcedModel:
            def __init__(self, model, token):
                self.model = model
                self.language_model = model
                self.processor = LogitsProcessorList([ForceToken(token)])

            def generate(self, **kwargs):
                return self.model.generate(logits_processor=self.processor, **kwargs)

        config = OPTConfig(vocab_size=16, hidden_size=16, word_embed_proj_dim=16,
                           ffn_dim=32, num_hidden_layers=1, num_attention_heads=2,
                           max_position_embeddings=64, bos_token_id=2, eos_token_id=2, pad_token_id=1)
        model = OPTForCausalLM(config).eval()
        model.generation_config.eos_token_id = 7  # Checkpoint default differs from approved EOS 2.
        inputs = {'input_ids': torch.tensor([[2, 4]])}
        gen = {'batch_size': 1, 'max_new_tokens': 4, 'do_sample': False, 'num_beams': 1,
               'use_cache': True, 'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0,
               'bos_token_id': 2, 'eos_token_id': 2, 'pad_token_id': 1}
        for forced_token, expected_new in ((2, 1), (5, 4), (7, 4)):
            forced = ForcedModel(model, forced_token)
            with torch.no_grad():
                native = forced.generate(**inputs, max_new_tokens=gen['max_new_tokens'],
                    do_sample=False, num_beams=1, use_cache=True, repetition_penalty=1.0,
                    no_repeat_ngram_size=0, bos_token_id=2, eos_token_id=2, pad_token_id=1)
                with patch.object(torch.Tensor, 'prod', side_effect=AssertionError('native CUDA EOS reduction used')):
                    compatible = mod.generate_single_sequence(forced, inputs, gen)
            self.assertTrue(torch.equal(native, compatible))
            self.assertEqual(compatible.shape[1] - inputs['input_ids'].shape[1], expected_new)
            self.assertEqual(model.generation_config.eos_token_id, 7)
        with self.assertRaises(ValueError):
            mod.generate_single_sequence(forced, {'input_ids': inputs['input_ids'].repeat(2, 1)}, gen)
        self.assertEqual(model.generation_config.eos_token_id, 7)
        with patch.object(forced, 'generate', side_effect=RuntimeError('forced generation failure')):
            with self.assertRaisesRegex(RuntimeError, 'forced generation failure'):
                mod.generate_single_sequence(forced, inputs, gen)
        self.assertEqual(model.generation_config.eos_token_id, 7)

        embeds = model.get_input_embeddings()(inputs['input_ids']).detach()
        forced = ForcedModel(model, 5)
        with torch.no_grad():
            tokens = mod.generate_single_sequence(forced, {'inputs_embeds': embeds,
                'attention_mask': torch.ones_like(inputs['input_ids'])}, gen)
        continuation = mod.continuation_tokens(tokens, gen['bos_token_id'])
        self.assertEqual(tokens.shape[1], gen['max_new_tokens'] + 1)
        self.assertEqual(continuation.shape[1], gen['max_new_tokens'])
        self.assertEqual(continuation.tolist(), [[5] * gen['max_new_tokens']])
        with self.assertRaises(ValueError):
            mod.continuation_tokens(torch.tensor([[4, 5]]), gen['bos_token_id'])


if __name__ == '__main__':
    unittest.main()
