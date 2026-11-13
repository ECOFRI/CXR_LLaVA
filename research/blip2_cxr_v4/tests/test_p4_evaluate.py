"""CPU P4 metric and label-mask checks on synthetic reports."""
import importlib.util
import sys
import unittest
from pathlib import Path

VR = Path(__file__).resolve().parents[1]
p = VR / 'scripts/eval/p4_evaluate.py'
spec = importlib.util.spec_from_file_location('p4_evaluate', str(p))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from vlm_medical.eval.text_metrics import case_statistics


class EvaluateTests(unittest.TestCase):
    def test_historical_mask_and_paired_cluster_bootstrap(self):
        records = [
            {'cohort': 'mimic', 'subject_id': 'A', 'study_id': '1', 'reference': 'no edema',
             'M0': {'generated_report': 'no edema'}, 'M1': {'generated_report': 'no edema'}},
            {'cohort': 'mimic', 'subject_id': 'A', 'study_id': '2', 'reference': 'edema',
             'M0': {'generated_report': 'edema'}, 'M1': {'generated_report': 'edema'}},
        ]
        labels = {}
        for i in range(2):
            for model in ('reference', 'M0', 'M1'):
                labels[(i, model)] = {name: '' for name in mod.PATHOLOGIES}
                labels[(i, model)]['Edema'] = '0' if i == 0 else '1'
        labels[(1, 'M0')]['Edema'] = '-1'
        masked = mod.point_clinical([0, 1], 'M0', labels, 'historical')
        self.assertEqual(masked['per_class']['Edema']['paired'], 1)
        self.assertEqual(masked['per_class']['Edema']['generated_uncertain'], 1)
        self.assertIsNone(masked['macro_f1'])
        definite = mod.point_clinical([0, 1], 'M1', labels, 'historical')
        self.assertEqual(definite['per_class']['Edema']['support'], 1)
        self.assertEqual(definite['per_class']['Edema']['f1'], 1.0)
        meteor = {'M0': {0: 0.0, 1: 0.0}, 'M1': {0: 0.0, 1: 0.0}}
        stats = {model: {i: case_statistics(row['reference'], row[model]['generated_report'])
                         for i, row in enumerate(records)} for model in ('M0', 'M1')}
        ci, deltas, clusters = mod.bootstrap_dataset(records, [0, 1], ['M0', 'M1'], labels,
                                                      meteor, stats, {'evaluation': {'bootstrap_seed': 42, 'bootstrap_reps': 10}})
        self.assertEqual(clusters, 1)
        self.assertEqual(deltas['bleu4'], [0.0, 0.0])
        self.assertEqual(ci['M0']['bleu4'], ci['M1']['bleu4'])


if __name__ == '__main__':
    unittest.main()
