"""Synthetic CPU checks for candidate cohort identity and pairing."""

import csv
import gzip
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

VR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VR / 'src'))
SPEC = importlib.util.spec_from_file_location(
    'p4_manifest_proposal', VR / 'scripts/data/build_p4_manifest_proposal.py')
proposal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proposal)


def write_csv(path, rows, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ManifestProposalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=VR / '.cache')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def mimic_fixture(self):
        root = self.root / 'jpg'
        root.mkdir()
        records = [
            {'dicom_id': 'missing', 'subject_id': '100', 'study_id': '10', 'split': 'train'},
            {'dicom_id': 'found', 'subject_id': '100', 'study_id': '10', 'split': 'train'},
            {'dicom_id': 'other', 'subject_id': '100', 'study_id': '11', 'split': 'train'},
            {'dicom_id': 'val', 'subject_id': '200', 'study_id': '20', 'split': 'validate'},
        ]
        write_csv(root / 'mimic-cxr-2.0.0-split.csv.gz', records, compressed=True)
        metadata = [{**{k: r[k] for k in ('dicom_id', 'subject_id', 'study_id')},
                     'ViewPosition': 'AP'} for r in records]
        write_csv(root / 'mimic-cxr-2.0.0-metadata.csv.gz', metadata, compressed=True)
        reports = self.root / 'reports'
        for row in records:
            relative = Path('p' + row['subject_id'][:2]) / ('p' + row['subject_id']) / ('s' + row['study_id'])
            image = root / 'files' / relative / (row['dicom_id'] + '.jpg')
            image.parent.mkdir(parents=True, exist_ok=True)
            if row['dicom_id'] != 'missing':
                image.write_bytes(b'synthetic fixture; not decoded')
            report = reports / relative.with_suffix('.txt')
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text('FINDINGS: synthetic finding\nIMPRESSION: synthetic impression')
        return {'jpg_root': str(root), 'jpg_alt_files': str(self.root / 'alt'),
                'report_files': str(reports)}

    def test_mimic_preserves_first_available_image_and_unique_studies(self):
        config = self.mimic_fixture()
        train = proposal.mimic_candidates(config, 'train', 2)
        self.assertEqual([r['study_id'] for r in train], ['10', '11'])
        self.assertEqual([r['image_id'] for r in train], ['found', 'other'])
        self.assertTrue(all(r['view'] == 'AP' for r in train))
        self.assertEqual(train[0]['target_text_sha256'], proposal.text_sha256(
            'synthetic finding\nsynthetic impression'))
        validate = proposal.mimic_candidates(config, 'validate', 1)
        self.assertEqual(validate[0]['subject_id'], '200')
        self.assertEqual(validate[0]['split'], 'validate')

    def test_mimic_rejects_insufficient_count(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient eligible'):
            proposal.mimic_candidates(self.mimic_fixture(), 'train', 3)

    def indiana_fixture(self, wrong_reference=False):
        image = self.root / 'image.jpg'
        image.write_bytes(b'synthetic fixture; not decoded')
        cohort = self.root / 'cohort.csv'
        predictions = self.root / 'predictions.csv'
        row = {'report_id': 's1', 'image_id': 'i1', 'image_path': str(image),
               'ground_truth_report': 'synthetic reference', 'reference_policy': 'fixture',
               'image_sha256': proposal.sha256(image), 'view_label': 'frontal_unspecified'}
        write_csv(cohort, [row])
        prediction = {k: row[k] for k in ('report_id', 'image_id', 'ground_truth_report')}
        prediction.update(status='success', generated_report='synthetic output')
        if wrong_reference:
            prediction['ground_truth_report'] = 'wrong reference'
        write_csv(predictions, [prediction])
        return cohort, predictions

    def test_indiana_joins_saved_prediction_by_stable_id(self):
        cohort, predictions = self.indiana_fixture()
        rows = proposal.indiana_candidates(cohort, predictions, 1)
        self.assertEqual((rows[0]['study_id'], rows[0]['image_id']), ('s1', 'i1'))
        self.assertEqual(rows[0]['source_row'], 0)
        self.assertIsNone(rows[0]['subject_id'])
        self.assertNotIn('ground_truth_report', rows[0])

    def test_indiana_rejects_reference_mismatch(self):
        cohort, predictions = self.indiana_fixture(wrong_reference=True)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            proposal.indiana_candidates(cohort, predictions, 1)


if __name__ == '__main__':
    unittest.main()
