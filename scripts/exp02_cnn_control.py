"""Opt-in CNN comparator on the exact frozen, reviewed Indiana cohort."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scripts.exp02_reevaluate import TARGET_PATHOLOGIES, f1_from_counts

WEIGHTS = 'densenet121-res224-chex'
LABEL_MAP = {p: ('Effusion' if p == 'Pleural Effusion' else p) for p in TARGET_PATHOLOGIES}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--review-gate', type=Path, required=True,
                        help='JSON with cohort_sha256, gt_review_pass and view_review_pass')
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cohort = pd.read_csv(args.cohort, dtype={'image_id': str, 'report_id': str})
    gate = json.loads(args.review_gate.read_text(encoding='utf-8'))
    if gate.get('cohort_sha256') != sha(args.cohort) or not all(
        gate.get(k) is True for k in ['gt_review_pass', 'view_review_pass']
    ):
        raise ValueError('Frozen cohort GT/view review gate has not passed')
    if not cohort.image_id.is_unique or not cohort.report_id.is_unique:
        raise ValueError('Cohort must have unique images and studies')
    for p in TARGET_PATHOLOGIES:
        if cohort[p].eq(1).sum() < 20 or cohort[p].eq(0).sum() < 20:
            raise ValueError(f'Insufficient positive/negative support: {p}')
    files = [args.image_root / (str(i) + '.png') for i in cohort.image_id]
    for file in files:
        with Image.open(file) as im:
            im.verify()
    if not args.execute:
        print(f'Preflight PASS: {len(files)} images. Use --execute for CNN inference.')
        return
    import torch
    import torchxrayvision as xrv
    if not torch.cuda.is_available():
        raise RuntimeError('Use a CUDA runtime for this comparator')
    args.output.mkdir(parents=True, exist_ok=False)
    model = xrv.models.DenseNet(weights=WEIGHTS).eval().cuda()
    labels = list(model.pathologies)
    thresholds = model.op_threshs.detach().cpu().numpy().copy()
    # Disable XRV operating-point normalization; apply released thresholds once.
    model.op_threshs = None
    model.apply_sigmoid = True
    indices = {p: labels.index(LABEL_MAP[p]) for p in TARGET_PATHOLOGIES}
    if not all(np.isfinite(thresholds[i]) for i in indices.values()):
        raise ValueError('An evaluation target is untrained or has no released threshold')
    config = dict(model=WEIGHTS, cohort_sha256=sha(args.cohort),
                  weights_sha256=sha(model.weights_filename_local),
                  torch=torch.__version__, torchxrayvision=importlib.metadata.version('torchxrayvision'),
                  label_map=LABEL_MAP, thresholds={p: float(thresholds[i]) for p, i in indices.items()},
                  preprocessing='8-bit grayscale; normalize 255; center crop; resize 224',
                  interpretation='classification comparator; no report-generation score')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    rows = []
    for image_id, file in zip(cohort.image_id, files):
        with Image.open(file) as im:
            if im.mode not in ('L', 'RGB', 'RGBA'):
                raise ValueError(f'Unsupported bit depth/mode: {file.name}: {im.mode}')
            pixels = np.asarray(im.convert('L'))
        x = xrv.datasets.normalize(pixels, 255)[None, ...]
        x = xrv.datasets.XRayCenterCrop()(x)
        x = xrv.datasets.XRayResizer(224)(x)
        with torch.inference_mode():
            scores = model(torch.from_numpy(x).unsqueeze(0).cuda())[0].cpu().numpy()
        row = {'image_id': image_id, 'image_sha256': sha(file)}
        for p, i in indices.items():
            row['score_' + p] = float(scores[i])
            row[p] = int(scores[i] >= thresholds[i])
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output / 'cnn_predictions.csv', index=False)
    predictions = pd.DataFrame(rows).set_index('image_id').loc[cohort.image_id]
    metrics = []
    for p in TARGET_PATHOLOGIES:
        g = cohort[p].to_numpy(); y = predictions[p].to_numpy(); valid = np.isin(g, [0, 1])
        tp = int(((g == 1) & (y == 1) & valid).sum())
        fp = int(((g == 0) & (y == 1) & valid).sum())
        fn = int(((g == 1) & (y == 0) & valid).sum())
        metrics.append(dict(pathology=p, n_valid=int(valid.sum()), tp=tp, fp=fp, fn=fn,
                            f1=f1_from_counts(tp, fp, fn)))
    pd.DataFrame(metrics).to_csv(args.output / 'cnn_metrics_gt_definite.csv', index=False)


if __name__ == '__main__':
    main()
