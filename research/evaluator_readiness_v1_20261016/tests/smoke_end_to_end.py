from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research' / 'evaluation' / 'g2c_adapter_v1'))
from e4_adapter import load_inputs, compute_metrics, write_csv
from e4_labeler_bridge import convert_wide
from make_synthetic_fixture import generate


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('synthetic_bridge_smoke')
    source = root / 'input'
    source.mkdir(parents=True, exist_ok=True)
    generate(source)

    import csv
    def read(name):
        with (source / name).open('r', encoding='utf-8', newline='') as handle:
            return list(csv.DictReader(handle))

    references = read('references.csv')
    predictions = read('predictions.csv')
    reference_wide = defaultdict(dict)
    prediction_wide = defaultdict(dict)
    for row in references:
        key = 'REF_' + row['image_id']
        reference_wide[key]['report_key'] = key
        reference_wide[key][row['finding']] = row['reference_label']
    for row in predictions:
        key = 'PRED_' + row['arm_id'] + '_' + row['image_id']
        prediction_wide[key]['report_key'] = key
        prediction_wide[key][row['finding']] = row['prediction_label']

    write_csv(root / 'reference_wide.csv', list(reference_wide.values())[::-1])
    write_csv(root / 'prediction_wide.csv', list(prediction_wide.values())[::-1])
    write_csv(root / 'reference_map.csv', [
        {'report_key': key, 'image_id': key[4:]}
        for key in reference_wide
    ])
    write_csv(root / 'prediction_map.csv', [
        {'report_key': key, 'image_id': key.split('_', 2)[-1], 'arm_id': key.split('_', 2)[1]}
        for key in prediction_wide
    ])
    reference_long = convert_wide(
        root / 'reference_wide.csv', root / 'reference_map.csv', source / 'findings.txt', 'reference'
    )
    prediction_long = convert_wide(
        root / 'prediction_wide.csv', root / 'prediction_map.csv', source / 'findings.txt', 'prediction'
    )
    write_csv(root / 'reference_converted.csv', reference_long)
    write_csv(root / 'prediction_converted.csv', prediction_long)
    images, findings, records = load_inputs(
        source / 'cohort.csv', source / 'findings.txt',
        root / 'reference_converted.csv', root / 'prediction_converted.csv', source / 'reports.csv'
    )
    per_finding, macro = compute_metrics(records, images, findings)
    write_csv(root / 'per_finding_metrics.csv', per_finding)
    write_csv(root / 'macro_metrics.csv', macro)
    assert len(reference_long) == 100
    assert len(prediction_long) == 400
    assert len(records) == 400
    c0 = next(item for item in per_finding if item['arm_id'] == 'C0' and item['finding'] == 'F01')
    assert (c0['tp'], c0['tn'], c0['fp'], c0['fn']) == (0, 14, 1, 1)
    assert c0['joint_definite'] == 16
    print(json.dumps({
        'result': 'PASS_END_TO_END_SYNTHETIC',
        'reports': 80,
        'reference_records': len(reference_long),
        'prediction_records': len(prediction_long),
        'baseline_F01_joint_definite': c0['joint_definite'],
        'missing_values_not_forced_negative': True,
    }, indent=2))


if __name__ == '__main__':
    main()
