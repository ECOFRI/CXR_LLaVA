import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.k80_original import export_reports
from scripts.manual_experiment import write_csv
from scripts.exp02_reevaluate import TARGET_PATHOLOGIES, reevaluate


class CohortEvaluationTests(unittest.TestCase):
    def test_report_export_and_paper_pathologies(self):
        # Synthetic text/labels exercise alignment and known confusion counts;
        # these are not CheXpert outputs or an accuracy experiment.
        for dataset, count in [('indiana', 7), ('mimic', 6)]:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'source'
                for name in ['subsets', 'predictions', 'chexpert_inputs', 'chexpert_labels']:
                    (source / name).mkdir(parents=True)
                prefix = f'{dataset}_f1_subset_4'
                rows = [dict(image_id=f'00{i}', report_id=f'r{i}',
                             ground_truth_report=f'GT {i}, "quoted"\nsecond line',
                             generated_report=f'Generated {i},\nsecond line', status='success')
                        for i in range(4)]
                write_csv(source / 'subsets' / f'{prefix}_seed42.csv', rows,
                          ['report_id', 'image_id', 'ground_truth_report'])
                write_csv(source / 'predictions' / f'{prefix}_predictions.csv', rows,
                          list(rows[0]))
                export_reports(source, prefix, rows)
                for kind, key, labels in [('gt', 'ground_truth_report', [1, 0, 1, 0]),
                                          ('generated', 'generated_report', [1, 1, 0, 0])]:
                    frame = pd.DataFrame({'Reports': [row[key] for row in rows]})
                    for pathology in TARGET_PATHOLOGIES:
                        frame[pathology] = labels
                    frame.to_csv(source / 'chexpert_labels' / f'{prefix}_{kind}_chexpert_labels.csv', index=False)
                output = root / 'metrics'
                reevaluate(source, output, [4], 20, 42, dataset)
                metrics = pd.read_csv(output / 'metrics' / f'{prefix}_pathology_metrics_v2.csv')
                self.assertEqual(len(metrics), count)
                for key in ['precision', 'recall', 'f1']:
                    self.assertTrue(metrics[key].eq(0.5).all())
                summary = json.loads((output / 'metrics' / f'{prefix}_summary.json').read_text())
                self.assertEqual(summary['macro_f1'], 0.5)


if __name__ == '__main__':
    unittest.main()
