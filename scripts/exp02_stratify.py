"""Select a frozen multilabel cohort from reviewed GT labels, never predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.exp02_reevaluate import TARGET_PATHOLOGIES


def select_cohort(pool, target=25, negatives=25, seed=42):
    required = ['image_id', 'report_id', 'image_path', 'ground_truth_report', *TARGET_PATHOLOGIES]
    if not set(required).issubset(pool):
        raise ValueError('Pool must contain IDs, image path, GT report and seven GT label columns')
    if pool.image_id.isna().any() or not pool.image_id.is_unique or not pool.report_id.is_unique:
        raise ValueError('Require one unique image per unique report/study')
    if any(not pool[p].dropna().isin([-1, 0, 1]).all() for p in TARGET_PATHOLOGIES):
        raise ValueError('Unexpected label value')
    if target < 20 or negatives < 1:
        raise ValueError('Require at least 20 positives per pathology and negative controls')
    pool = pool.sort_values('image_id').reset_index(drop=True)
    support = pd.DataFrame([
        dict(pathology=p, positive=int(pool[p].eq(1).sum()), negative=int(pool[p].eq(0).sum()),
             uncertain=int(pool[p].eq(-1).sum()), unmentioned=int(pool[p].isna().sum()))
        for p in TARGET_PATHOLOGIES
    ])
    if (support.positive < target).any() or (support.negative < negatives).any():
        return None, support
    # Greedy coverage across the 14 GT strata; deterministic randomized tie order.
    strata = np.column_stack([pool[p].eq(v).to_numpy() for p in TARGET_PATHOLOGIES for v in [1, 0]])
    quota = np.array([target, negatives] * len(TARGET_PATHOLOGIES))
    order = np.random.default_rng(seed).permutation(len(pool))
    selected = []
    counts = np.zeros(len(quota), dtype=int)
    while (counts < quota).any():
        remaining = [i for i in order if i not in selected]
        scores = (strata[remaining] * (counts < quota) / strata.sum(axis=0)).sum(axis=1)
        chosen = remaining[int(np.argmax(scores))]
        if scores.max() <= 0:
            raise ValueError('Cannot satisfy quotas')
        selected.append(chosen)
        counts += strata[chosen]
    return pool.iloc[selected].sort_values('image_id').reset_index(drop=True), support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target', default=25, type=int)
    parser.add_argument('--negatives', default=25, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pool = pd.read_csv(args.pool, dtype={'image_id': str, 'report_id': str})
    selected, support = select_cohort(pool, args.target, args.negatives, args.seed)
    support.to_csv(args.output / 'pool_support.csv', index=False)
    config = dict(target=args.target, negatives=args.negatives, seed=args.seed,
                  pool_sha256=hashlib.sha256(args.pool.read_bytes()).hexdigest(),
                  status='BLOCKED_INSUFFICIENT_SUPPORT' if selected is None else 'SELECTED_REQUIRES_GT_VIEW_REVIEW')
    if selected is not None:
        selected.to_csv(args.output / 'cohort.csv', index=False)
        config['cohort_sha256'] = hashlib.sha256((args.output / 'cohort.csv').read_bytes()).hexdigest()
    (args.output / 'selection.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
