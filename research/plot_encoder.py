"""Plot actual CXR encoder gate resources, without claiming CLEAR measurements."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(run):
    manifest = json.loads((run / 'run_manifest.json').read_text())
    rows = list(csv.DictReader((run / 'feature_manifest.csv').open()))
    root = Path(__file__).resolve().parents[1]
    folder = root / 'research/figures' / manifest['start_time'][:10] / manifest['run_id']
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(1, len(rows) + 1), [float(r['latency_sec']) for r in rows], marker='.')
    axes[0].set(xlabel='Technical gate image index', ylabel='Forward seconds / image', title='FP32 CXR vision + projector; GPU0')
    axes[1].plot(range(1, len(rows) + 1), [float(r['allocated_peak_gib']) for r in rows], label='Allocated')
    axes[1].plot(range(1, len(rows) + 1), [float(r['reserved_peak_gib']) for r in rows], label='Reserved')
    axes[1].set(xlabel='Technical gate image index', ylabel='Peak GiB', title='Single K80; no LLaMA loaded')
    axes[1].legend()
    fig.tight_layout()
    path = folder / 'latency_memory.png'
    fig.savefig(path, dpi=160)
    print(path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    plot(p.parse_args().run)
