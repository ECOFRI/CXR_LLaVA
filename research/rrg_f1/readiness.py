"""CPU readiness gate: preserve P4 labeler revisions and local-only assets."""
import json
import os
import subprocess
import time
from pathlib import Path

from research.rrg_f0.core import ROOT
from research.rrg_f1.cohort import BASE, atomic_json
from research.rrg_f1.labeler import runtime_env


def check_runtime(python, parser_model):
    result = {'python_exists': python.is_file() and os.access(python, os.X_OK),
              'parser_exists': (parser_model / 'parser').is_dir() and (parser_model / 'reranker').is_dir()}
    if not result['python_exists']:
        result.update(status='STOP_MISSING_P4_LABELER_RUNTIME')
        return result
    code = "import importlib.util,json; names=['bllipparser','nltk','bioc','pandas','StanfordDependencies','ply']; modules={n:importlib.util.find_spec(n) is not None for n in names}; [__import__(n) for n in names if modules[n]]; print(json.dumps(modules))"
    probe = subprocess.run([str(python), '-c', code], capture_output=True, text=True, timeout=60,
                           env=runtime_env())
    if probe.returncode:
        result.update(status='STOP_LABELER_IMPORT_PROBE_FAILED', returncode=probe.returncode)
        return result
    result['modules'] = json.loads(probe.stdout.strip())
    if not all(result['modules'].values()) or not result['parser_exists']:
        result['status'] = 'STOP_MISSING_P4_LABELER_ASSETS'
    else:
        # Static/import readiness is deliberately not a 200-report throughput PASS.
        result['status'] = 'IMPORT_AND_ASSET_GATE_PASS'
    return result


def main():
    started = time.monotonic()
    config = json.loads((BASE / 'config.json').read_text())
    local = json.loads((BASE / 'paths.local.json').read_text())
    tools = ROOT / 'model_cache/chexpert_tools'
    revisions = {}
    for name, key in [('chexpert-labeler', 'labeler_revision'), ('NegBio', 'negbio_revision')]:
        revision = subprocess.check_output(['git', '-C', str(tools / name), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != config[key]:
            raise ValueError('STOP_LABELER_REVISION_MISMATCH')
        revisions[name] = revision
    status = check_runtime(Path(local['labeler_python']), Path(local['parser_model_dir']))
    status.update(revisions=revisions, gpu_seconds=0, wall_seconds=time.monotonic() - started,
                  benchmark_reports_completed=0, projected_labeler_seconds=None,
                  compatibility_note='P4 runtime was Python3.6/NLTK3.3; F0 rrg-qwen is Python3.11. Do not silently substitute labeler/runtime.',
                  label_alignment_note='Map label columns by name: F0 orders Lung Opacity before Lung Lesion; CheXpert labeler orders Lung Lesion before Lung Opacity.')
    atomic_json(BASE / 'readiness_summary.json', status)
    print(json.dumps(status), flush=True)
    if status['status'] != 'IMPORT_AND_ASSET_GATE_PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
