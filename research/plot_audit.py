"""Plot saved audit tables using an existing Matplotlib environment."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(run):
    manifest = json.loads((run / 'run_manifest.json').read_text())
    root = Path(__file__).resolve().parents[1]
    folder = root / 'research/figures' / manifest['start_time'][:10] / manifest['run_id']
    rows = list(csv.DictReader((run / 'metrics_policies.csv').open()))
    a = [r for r in rows if r['policy'] == 'joint_definite_parsed_only']
    b = [r for r in rows if r['policy'] == 'sensitivity_missing_as_negative']
    names = [r['finding'] for r in a]
    values = np.array([float(r['f1']) for r in a])
    errors = np.array([[max(0, values[i] - float(r['ci95_low'])) for i, r in enumerate(a)],
                       [max(0, float(r['ci95_high']) - values[i]) for i, r in enumerate(a)]])
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    y = np.arange(len(a))
    axes[0].barh(y - .18, values, height=.36, xerr=errors, label='Parsed only; patient-cluster 95% CI')
    axes[0].barh(y + .18, [float(r['f1']) for r in b], height=.36, label='Sensitivity: missing as negative')
    axes[0].set(yticks=y, yticklabels=names, xlabel='F1', xlim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[1].barh(y, [100 * float(r['prediction_coverage']) for r in a])
    axes[1].set(yticks=y, yticklabels=names, xlabel='Explicit Yes/No coverage (%)', xlim=(0, 100))
    fig.suptitle('CheXpert locked test: offline audit; no CLEAR comparison or clinical adjudication')
    fig.tight_layout()
    fig.savefig(folder / 'f1_and_coverage.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(a))
    for key in ['tp', 'fp', 'tn', 'fn']:
        v = [int(r[key]) for r in a]
        ax.bar(y, v, bottom=bottom, label=key.upper())
        bottom += v
    ax.bar(y, [int(r['missing_predictions']) for r in a], bottom=bottom, label='Missing')
    ax.set(xticks=y, xticklabels=names, ylabel='Image count', title='Parsed confusion counts and missing predictions')
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / 'confusion_counts.png', dpi=160)
    print(folder)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    plot(p.parse_args().run)
