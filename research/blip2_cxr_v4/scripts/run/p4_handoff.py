"""Detached, fail-closed P4.2 to P4.3 handoff; never runs a GPU job early."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

VR = Path(__file__).resolve().parents[2]
ROOT = VR.parents[1]
sys.path.insert(0, str(VR / 'src'))
from vlm_medical.train.p4_runtime import atomic_json, _process_start_ticks


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verdict(run, config_path, training_pid, training_start_ticks):
    if _process_start_ticks(training_pid) == training_start_ticks:
        return 'WAIT', 'P4.2 supervisor still active'
    status_path = run / 'P4.2_status.json'
    checkpoint = run / 'P4.2_last.pt'
    if not status_path.is_file():
        return 'BLOCKED', 'P4.2 status missing after supervisor exit'
    status = json.loads(status_path.read_text())
    expected_config_hash = sha256(config_path)
    if (status.get('status') != 'PASS' or status.get('completed_studies') != 2000
            or status.get('optimizer_updates') != 2000 or status.get('epochs_complete') != 1
            or status.get('final_reload_verified') is not True
            or status.get('config_sha256') != expected_config_hash
            or status.get('cumulative_charged_seconds', 14401) > 14400
            or not checkpoint.is_file() or status.get('final_checkpoint_sha256') != sha256(checkpoint)):
        return 'BLOCKED', 'P4.2 PASS, epoch, budget, or checkpoint identity criterion failed'
    return 'GO', 'P4.2 verified PASS'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--manifest-index', type=Path, required=True)
    p.add_argument('--gpu-index', type=int, required=True)
    p.add_argument('--training-supervisor-pid', type=int, required=True)
    p.add_argument('--training-supervisor-start-ticks', type=int, required=True)
    args = p.parse_args()
    run = args.output_dir.resolve()
    if not run.is_relative_to((VR / 'runs').resolve()):
        raise ValueError('Run must remain under V4 runs')
    marker = run / 'P4.3_handoff_status.json'
    if marker.exists():
        raise FileExistsError('Handoff already initialized')
    atomic_json(marker, {'status': 'WAITING_FOR_P4.2', 'training_supervisor_pid': args.training_supervisor_pid,
                         'training_supervisor_start_ticks': args.training_supervisor_start_ticks,
                         'created_utc': datetime.now(timezone.utc).isoformat()})
    while True:
        decision, reason = verdict(run, args.config, args.training_supervisor_pid, args.training_supervisor_start_ticks)
        if decision != 'WAIT':
            break
        time.sleep(30)
    if decision != 'GO':
        atomic_json(marker, {'status': 'BLOCKED', 'reason': reason,
                             'checked_utc': datetime.now(timezone.utc).isoformat()})
        return 2
    atomic_json(marker, {'status': 'LAUNCHING_P4.3', 'reason': reason,
                         'checked_utc': datetime.now(timezone.utc).isoformat()})
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    python = '/home/student4/anaconda3/envs/cxr-llava-k80/bin/python'
    command = ['timeout', '--signal=TERM', '--kill-after=20s', '7200s', python, '-u',
               str(VR / 'scripts/experiments/p4_paired_infer.py'), '--config', str(args.config),
               '--manifest-index', str(args.manifest_index), '--output-dir', str(run),
               '--gpu-index', str(args.gpu_index)]
    with (run / 'P4.3_stdout.log').open('w') as stdout, (run / 'P4.3_stderr.log').open('w') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr)
        atomic_json(marker, {'status': 'P4.3_RUNNING', 'supervisor_pid': process.pid,
                             'command': command, 'checked_utc': datetime.now(timezone.utc).isoformat()})
        exit_code = process.wait()
    status_path = run / 'P4.3_status.json'
    stage = json.loads(status_path.read_text()) if status_path.is_file() else {}
    if (stage.get('status') not in ('PASS', 'PARTIAL') or stage.get('validity_gate') != 'PASS'
            or stage.get('completed_studies', 0) == 0):
        atomic_json(marker, {'status': 'BLOCKED', 'inference_status': stage.get('status', 'MISSING'),
                             'exit_code': exit_code, 'reason': stage.get('reason', 'P4.3 validity gate failed'),
                             'completed_studies': stage.get('completed_studies', 0),
                             'checked_utc': datetime.now(timezone.utc).isoformat()})
        return exit_code or 2
    atomic_json(marker, {'status': 'EVALUATING_P4.3', 'inference_status': stage['status'],
                         'completed_studies': stage['completed_studies'],
                         'checked_utc': datetime.now(timezone.utc).isoformat()})
    evaluation_env = os.environ.copy()
    evaluation_env['NLTK_DATA'] = str(VR / '.cache/nltk')
    evaluation_env['CUDA_VISIBLE_DEVICES'] = ''
    evaluation_env['HF_HUB_OFFLINE'] = '1'
    evaluation_env['TRANSFORMERS_OFFLINE'] = '1'
    eval_command = ['timeout', '--signal=TERM', '--kill-after=20s', '7200s',
                    '/home/student4/anaconda3/envs/chexpert-label/bin/python', '-u',
                    str(VR / 'scripts/eval/p4_evaluate.py'), '--config', str(args.config),
                    '--manifest-index', str(args.manifest_index), '--run-dir', str(run)]
    with (run / 'P4.3_evaluate_stdout.log').open('w') as stdout, (run / 'P4.3_evaluate_stderr.log').open('w') as stderr:
        result = subprocess.run(eval_command, cwd=ROOT, env=evaluation_env, stdin=subprocess.DEVNULL,
                                stdout=stdout, stderr=stderr)
    metrics_path = run / 'P4.3_evaluation/metrics.json'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    complete = result.returncode == 0 and metrics.get('status') in ('PASS', 'PARTIAL')
    atomic_json(marker, {'status': metrics['status'] if complete else 'METRICS_FAIL',
                         'inference_status': stage['status'], 'inference_exit_code': exit_code,
                         'evaluation_exit_code': result.returncode,
                         'completed_studies': stage['completed_studies'],
                         'metrics_path': str(metrics_path) if complete else None,
                         'reason': 'Metric evaluation complete' if complete else 'Metric evaluator failed; see stderr/log',
                         'checked_utc': datetime.now(timezone.utc).isoformat()})
    return 0 if complete else result.returncode or 3


if __name__ == '__main__':
    sys.exit(main())
