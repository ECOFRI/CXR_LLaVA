import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "evaluation" / "g2c_adapter_v1"))
from e4_adapter import AdapterError
from e4_labeler_bridge import convert_wide


class LabelerBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        (self.base / 'findings.txt').write_text('F01\nF02\nF03\nF04\nF05\n', encoding='utf-8')

    def write(self, name, rows):
        path = self.base / name
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_wide_prediction_with_explicit_key_mapping(self):
        mapping = self.write('mapping.csv', [
            {'report_key': 'z', 'image_id': 'IMG_02', 'arm_id': 'C1'},
            {'report_key': 'a', 'image_id': 'IMG_01', 'arm_id': 'C0'},
        ])
        wide = self.write('wide.csv', [
            {'report_key': 'a', 'F01': '1.0', 'F02': '0', 'F03': '-1.0', 'F04': '', 'F05': '0'},
            {'report_key': 'z', 'F01': '0', 'F02': '0', 'F03': '0', 'F04': '0', 'F05': '1'},
        ])
        rows = convert_wide(wide, mapping, self.base / 'findings.txt', 'prediction')
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]['image_id'], 'IMG_01')
        self.assertEqual([item['prediction_label'] for item in rows[:4]], [
            'positive', 'negative', 'uncertain', 'blank'
        ])

    def test_unmapped_source_is_blocked(self):
        mapping = self.write('mapping.csv', [{'report_key': 'a', 'image_id': 'IMG_01'}])
        wide = self.write('wide.csv', [
            {'report_key': 'b', 'F01': '0', 'F02': '0', 'F03': '0', 'F04': '0', 'F05': '0'}
        ])
        with self.assertRaisesRegex(AdapterError, 'mismatch'):
            convert_wide(wide, mapping, self.base / 'findings.txt', 'reference')

    def test_unsupported_value_is_blocked(self):
        mapping = self.write('mapping.csv', [{'report_key': 'a', 'image_id': 'IMG_01'}])
        wide = self.write('wide.csv', [
            {'report_key': 'a', 'F01': 'likely', 'F02': '0', 'F03': '0', 'F04': '0', 'F05': '0'}
        ])
        with self.assertRaisesRegex(AdapterError, 'Unrecognized'):
            convert_wide(wide, mapping, self.base / 'findings.txt', 'reference')


if __name__ == '__main__':
    unittest.main()
