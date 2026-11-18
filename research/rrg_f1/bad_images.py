"""Private exclusions authorized by the user; original manifests/cache indices stay fixed."""
import json
from research.rrg_f1.cohort import PRIVATE, atomic_json


def exclusions():
    path = PRIVATE / 'bad_images.json'
    return json.loads(path.read_text()) if path.exists() else []


def record_bad(family, index, row):
    entries = exclusions()
    if not any(e['family'] == family and e['index'] == index for e in entries):
        entry = {'family': family, 'index': index, 'dicom_id': row.get('dicom_id', row.get('image_id')), 'image_path': row['image_path']}
        with (PRIVATE / 'bad_images.txt').open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        entries.append(entry)
        atomic_json(PRIVATE / 'bad_images.json', entries)
    if family != 'train':
        raise RuntimeError('STOP_BAD_EVAL_IMAGE_NO_REPLACEMENT')
    if len(entries) > 300:
        raise RuntimeError('STOP_BAD_IMAGES_GT_1_PERCENT')


def active_indices(count):
    bad = {e['index'] for e in exclusions() if e['family'] == 'train'}
    return [i for i in range(count) if i not in bad]
