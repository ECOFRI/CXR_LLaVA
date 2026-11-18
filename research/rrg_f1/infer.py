"""Idempotent local generation on frozen dev cohorts, plus authorized A4 controls."""
import argparse
import json
import random
import time
import torch
from research.rrg_f0.core import sha256
from research.rrg_f1.cohort import BASE, PRIVATE, atomic_json
from research.rrg_f1.data import rows_for, read_feature
from research.rrg_f1.runtime import build_model, restore_model, generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['A2', 'A3', 'A4'], required=True)
    parser.add_argument('--interventions', action='store_true')
    args = parser.parse_args()
    model, tokenizer = build_model(args.arm)
    checkpoint = restore_model(model, args.arm)
    if checkpoint['state']['status'] not in ('COMPLETE', 'COLLAPSE'):
        raise ValueError('Arm has no terminal checkpoint')
    model.eval()
    del checkpoint
    started = time.monotonic()
    cases = [(family, i) for family in ('mimic', 'indiana') for i in range(500)]
    modes = ['baseline']
    scores = None
    if args.interventions:
        if args.arm != 'A4':
            raise ValueError('Only authorized A4 token interventions')
        cases = [(family, i) for family in ('mimic', 'indiana') for i in range(50)]
        scores = torch.stack([read_feature(family, i, rows_for(family)[i])[1] for family, i in cases])
        modes = ['negative', 'shuffled']
    for mode in modes:
        directory = PRIVATE / args.arm / ('predictions' if mode == 'baseline' else 'intervention_' + mode)
        directory.mkdir(exist_ok=True)
        identity = {'checkpoint_sha256': sha256(PRIVATE / args.arm / 'last.pt'), 'arm': args.arm, 'mode': mode, 'cases': len(cases)}
        marker = directory / 'identity.json'
        if marker.exists() and json.loads(marker.read_text()) != identity:
            raise ValueError('Prediction checkpoint identity changed')
        if not marker.exists():
            atomic_json(marker, identity)
        overrides = None
        if mode == 'negative':
            overrides = torch.zeros_like(scores)
        elif mode == 'shuffled':
            order = list(range(len(cases)))
            random.Random(42).shuffle(order)
            overrides = scores[order]
        for start in range(0, len(cases), 4):
            batch = cases[start:start + 4]
            missing = [(position, case) for position, case in enumerate(batch, start) if not (directory / f'{position:04d}.json').exists()]
            if not missing:
                continue
            values = generate(model, tokenizer, [case for _, case in missing],
                              overrides[[position for position, _ in missing]] if overrides is not None else None)
            for (position, case), value in zip(missing, values):
                if (value['family'], value['index']) != case:
                    raise ValueError('Generation linkage mismatch')
                atomic_json(directory / f'{position:04d}.json', value)
            atomic_json(PRIVATE / args.arm / 'inference_heartbeat.json', {'mode': mode, 'completed_through': start + len(batch),
                        'unix_time': time.time(), 'seconds_this_attempt': time.monotonic() - started})
        outputs = [json.loads((directory / f'{i:04d}.json').read_text()) for i in range(len(cases))]
        if any((value['family'], value['index']) != case for value, case in zip(outputs, cases)):
            raise ValueError('Saved generation ordering changed')
        atomic_json(BASE / f'{args.arm}_{mode}_summary.json', {'status': 'PASS', 'reports': len(outputs),
                    'cap_count': sum(p['hit_max_length'] for p in outputs),
                    'empty_count': sum(not p['generated_report'].strip() for p in outputs),
                    'mean_tokens': sum(p['generated_tokens'] for p in outputs) / len(outputs),
                    'mode': mode, 'wall_seconds_this_attempt': time.monotonic() - started})


if __name__ == '__main__':
    main()
