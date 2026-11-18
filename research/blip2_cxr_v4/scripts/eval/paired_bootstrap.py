"""Paired study-grouped percentile bootstrap."""
import argparse
import json
import random
from pathlib import Path


def draws(studies, reps=1000, seed=42):
    if len(set(studies)) != len(studies):
        raise ValueError('Study keys must be unique')
    rng = random.Random(seed)
    return [[studies[rng.randrange(len(studies))] for _ in studies] for _ in range(reps)]


def interval(values):
    finite = sorted(x for x in values if x is not None)
    if not finite:
        return [None, None]
    return [finite[int(.025 * (len(finite) - 1))], finite[int(.975 * (len(finite) - 1))]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.config:
        cfg = json.loads(args.config.read_text())
        print(json.dumps({'status': 'DRY_RUN', 'seed': cfg['seed'], 'reps': cfg['bootstrap_reps']}))
    else:
        print(json.dumps({'status': 'LIBRARY_ONLY', 'seed': 42, 'reps': 1000}))


if __name__ == '__main__':
    main()
