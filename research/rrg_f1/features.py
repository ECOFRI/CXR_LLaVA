"""Resumable CLEAR disk256 cache, F0 validation reuse, batched frozen extraction."""
import json
import os
import time
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from research.rrg_f0.encoder import FrozenClear
from research.rrg_f1.cohort import BASE, PRIVATE, atomic_json
from research.rrg_f1.data import rows_for, binding, feature_path
from research.rrg_f1.labeler import LABELS
from research.rrg_f1.bad_images import exclusions, record_bad


class Images(Dataset):
    def __init__(self, rows, indices, preprocess):
        self.rows, self.indices, self.preprocess = rows, indices, preprocess

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        try:
            with Image.open(self.rows[index]['image_path']) as im:
                value = self.preprocess(im)
        except (OSError, UnidentifiedImageError):
            return index, None
        return index, value


def image_collate(batch):
    return batch


def main():
    os.umask(0o077)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    (PRIVATE / 'features').mkdir(exist_ok=True)
    from research.rrg_f0.core import ROOT, EXPECTED_CLEAR_SHA
    old = json.loads((ROOT / 'research/rrg_f0/feature_contract.json').read_text())
    if old['status'] != 'PASS' or old['labels'] != LABELS or old['clear_sha256'] != EXPECTED_CLEAR_SHA:
        raise ValueError('F0 feature provenance mismatch')
    encoder = FrozenClear(LABELS, 'cuda')
    processed = 0
    for family in ('train', 'indiana'):
        rows = rows_for(family)
        excluded = {e['index'] for e in exclusions() if e['family'] == family}
        if len(exclusions()) > 300:
            raise RuntimeError('STOP_BAD_IMAGES_GT_1_PERCENT')
        if any(e['family'] != 'train' for e in exclusions()):
            raise RuntimeError('STOP_BAD_EVAL_IMAGE_NO_REPLACEMENT')
        missing = [i for i in range(len(rows)) if i not in excluded and not feature_path(family, i).is_file()]
        dataset = Images(rows, missing, encoder.preprocess)
        loader = DataLoader(dataset, batch_size=4, num_workers=8, collate_fn=image_collate)
        for batch in loader:
            valid = []
            for index, image in batch:
                if image is None:
                    record_bad(family, index, rows[index])
                else:
                    valid.append((index, image))
            if not valid:
                continue
            indices = [i for i, _ in valid]
            images = torch.stack([im for _, im in valid])
            patches, scores = encoder.encode(images.cuda(non_blocking=True))
            for j, index in enumerate(indices):
                path = feature_path(family, index)
                temporary = path.with_suffix('.tmp')
                torch.save({'patch_tokens': patches[j].clone(), 'scores': scores[j].clone(), 'binding': binding(rows[index])}, temporary)
                temporary.replace(path)
                processed += 1
            if processed % 100 == 0:
                atomic_json(PRIVATE / 'feature_heartbeat.json', {'status': 'RUNNING', 'family': family,
                            'new_features': processed, 'unix_time': time.time(), 'seconds_current_process': time.monotonic() - started})
                print(json.dumps({'family': family, 'new_features': processed}), flush=True)
    for i in range(500):
        if not feature_path('mimic', i).is_file():
            raise ValueError('F0 validation feature missing')
    atomic_json(BASE / 'feature_summary.json', {'status': 'PASS', 'train': 30000 - sum(e['family'] == 'train' for e in exclusions()), 'excluded_bad_images': len(exclusions()), 'indiana': 500, 'mimic_reused_F0': 500,
                'input': 448, 'disk_tokens': 256, 'load_tokens': 64, 'dtype': 'float16', 'extraction_batch': 4,
                'new_features_this_attempt': processed, 'wall_seconds_this_attempt': time.monotonic() - started})


if __name__ == '__main__':
    main()
