"""Incremental NIH split availability/overlap audit; no training or image inference."""
import csv
import itertools
import json
from collections import Counter

from research.common import RESEARCH, ROOT, now, sha, write_csv, write_json


def audit():
    dataset = ROOT.parent / 'datasets/CXR8'
    local = {p.name for p in (dataset / 'images').rglob('*') if p.is_file()}
    with (dataset / 'Data_Entry_2017_v2020.csv').open() as stream:
        metadata = {r['Image Index']: r for r in csv.DictReader(stream)}
    rows, overlaps, sources = [], [], {}
    for folder, pattern in [('PruneCXR', '*labels_*.csv'), ('LongTailCXR', '*single-label_*.csv')]:
        sets = {}
        for file in sorted((dataset / folder).glob(pattern)):
            with file.open() as stream:
                records = list(csv.DictReader(stream))
            ids = {r['id'] if r['id'].endswith('.png') else r['id'] + '.png' for r in records}
            present = ids & local
            patients = {metadata[i]['Patient ID'] for i in present if i in metadata}
            view = Counter(metadata[i]['View Position'] for i in present if i in metadata)
            sets[file.stem] = (ids, present, patients)
            labels = [k for k in records[0] if not k.lower().endswith('id')]
            values = sorted({r[k] for r in records for k in labels})
            rows.append({'family': folder, 'manifest': file.name, 'rows': len(records), 'unique_image_ids': len(ids),
                         'locally_present_images': len(present), 'missing_local_images': len(ids - local),
                         'local_patient_count': len(patients), 'missing_patient_metadata': len(present - metadata.keys()),
                         'local_view_counts': json.dumps(view), 'label_columns': ';'.join(labels), 'label_values': ';'.join(values)})
            sources[str(file)] = sha(file)
        for a, b in itertools.combinations(sets, 2):
            overlaps.append({'family': folder, 'split_a': a, 'split_b': b,
                             'manifest_image_overlap': len(sets[a][0] & sets[b][0]),
                             'local_image_overlap': len(sets[a][1] & sets[b][1]),
                             'local_patient_overlap': len(sets[a][2] & sets[b][2])})
    write_csv(RESEARCH / 'inventory/nih_split_availability.csv', rows)
    write_csv(RESEARCH / 'inventory/nih_split_overlap.csv', overlaps)
    write_json(RESEARCH / 'inventory/nih_split_audit_manifest.json', {'timestamp': now(), 'source_sha256': sources,
        'metadata_sha256': sha(dataset / 'Data_Entry_2017_v2020.csv'), 'runner_sha256': sha(__file__),
        'status': 'CANDIDATE_ONLY', 'notes': ['No labels or thresholds tuned; no patient IDs exported.',
        'Zero within-family ID overlap does not prove absence of cross-dataset duplicate pixels or encoder pretraining overlap.',
        'Balanced val/test variants may intentionally overlap unbalanced counterpart; do not mix families or variants.',
        'Independent NIH development is a domain-transfer experiment; not CheXpert validation.',
        'Label provenance and report alignment still need approval/verification before E1 or report training.']})
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    audit()
