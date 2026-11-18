"""Local provenance helpers; no model loading or dataset traversal on import."""
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / 'research'
TZ = ZoneInfo('Asia/Ho_Chi_Minh')
BASELINE = ROOT / 'result/chexpert_table3_k80_20261005_full01/inference'


def now():
    return datetime.now(TZ).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pseudonym(value):
    return hashlib.sha256(('cxr-clear-local:' + str(value)).encode()).hexdigest()[:16]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command(args, cwd=ROOT):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return {'command': args, 'exit_code': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def new_run(experiment):
    run_id = datetime.now(TZ).strftime('%Y%m%dT%H%M%S') + '_' + experiment
    path = RESEARCH / 'runs' / run_id
    path.mkdir(parents=True, exist_ok=False)
    manifest = {'run_id': run_id, 'experiment': experiment, 'start_time': now(),
                'commit': command(['git', 'rev-parse', 'HEAD'])['stdout'].strip(),
                'worktree_status': command(['git', 'status', '--short'])['stdout'],
                'command': [], 'status': 'RUNNING', 'seed': 42,
                'process_environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG')},
                'actual_gpu_hours': 0, 'outputs': []}
    write_json(path / 'run_manifest.json', manifest)
    return path, manifest


def finish_run(path, manifest, status='COMPLETE'):
    # A diagnostic can finish successfully while recording an experiment blocker.
    manifest.update(status=status, end_time=now(), exit_code=0 if status in ('COMPLETE', 'BLOCKED') else 1)
    manifest['output_sha256'] = {str(f.relative_to(ROOT)): sha(f)
                               for f in path.rglob('*') if f.is_file()
                               and f.name != 'run_manifest.json'}
    write_json(path / 'run_manifest.json', manifest)
    registry = RESEARCH / 'experiment_registry.csv'
    fields = ['run_id', 'date', 'experiment', 'dataset', 'split', 'checkpoint',
              'commit', 'env', 'seed', 'precision', 'status', 'start_time',
              'end_time', 'estimated_gpu_hours', 'actual_gpu_hours',
              'metric_policy', 'artifact_path', 'blocker', 'next_action']
    with registry.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        row = {k: manifest.get(k, '') for k in fields}
        row.update(date=manifest['start_time'][:10], artifact_path=str(path))
        writer.writerow(row)
