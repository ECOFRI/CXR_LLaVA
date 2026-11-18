"""Detached F1 orchestration. No stage can bypass input gates or the GPU guard."""
import fcntl
import json
import os
import resource
import subprocess
import sys
import time
from research.rrg_f0.core import ROOT, sha256
from research.rrg_f1.cohort import BASE, PRIVATE, atomic_json
from research.rrg_f1.labeler import runtime_env


def main():
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    PRIVATE.mkdir(exist_ok=True)
    with (PRIVATE / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((BASE / 'config.json').read_text())
        scientific = ['bad_images.py','cohort.py','audit_cohort.py','labeler.py','model.py','data.py','features.py','runtime.py','train.py','infer.py','evaluate.py','intervention_analysis.py','supervisor.py','config.json','train_config.json']
        frozen = {name: sha256(BASE / name) for name in scientific}
        lock_path = PRIVATE / 'protocol_lock.json'
        if lock_path.exists() and json.loads(lock_path.read_text()) != frozen:
            raise ValueError('Protocol changed; STOP before resume')
        if not lock_path.exists():
            atomic_json(lock_path, frozen)
        if sha256(ROOT / config['work_order']) != config['work_order_sha256']:
            raise ValueError('Work order changed')
        local = json.loads((BASE / 'paths.local.json').read_text())
        python = str(ROOT / 'research/envs/rrg-qwen/bin/python')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
               'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
        status_path = PRIVATE / 'supervisor_status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {'status': 'RUNNING', 'attempts': []}
        status['status'] = 'RUNNING'
        atomic_json(status_path, status)

        def run(stage, command, seconds, marker=None, gpu=False, legacy=False):
            if marker and (BASE / marker).is_file():
                return
            attempt = {'stage': stage, 'status': 'RUNNING', 'started_unix': time.time()}
            status['attempts'].append(attempt)
            atomic_json(status_path, status)
            child_env = runtime_env() if legacy else env.copy()
            if gpu:
                child_env['CUDA_VISIBLE_DEVICES'] = '0'
                command = [sys.executable, '-m', 'research.rrg_f0.gpu_guard', '--stage', stage,
                           '--ledger', str(PRIVATE / 'gpu_budget.json'), '--limit-seconds', '50400',
                           '--seconds', str(seconds), '--stage-total-seconds', str(14400 if stage in ('A2', 'A3', 'A4') else seconds),
                           *command]
            else:
                command = ['timeout', '--signal=TERM', '--kill-after=10s', str(seconds), *command]
            log_path = PRIVATE / (stage + '_attempt_%03d.log' % len(status['attempts']))
            with log_path.open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=child_env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                attempt['pid'] = child.pid
                while child.poll() is None:
                    atomic_json(PRIVATE / 'supervisor_heartbeat.json', {'stage': stage, 'pid': child.pid, 'unix_time': time.time(),
                                'seconds_this_attempt': time.time() - attempt['started_unix']})
                    time.sleep(10)
            attempt.update(returncode=child.returncode, wall_seconds=time.time() - attempt['started_unix'],
                           status='PASS' if child.returncode == 0 else 'STOP')
            atomic_json(status_path, status)
            if child.returncode:
                raise RuntimeError('STOP_STAGE_' + stage)

        try:
            run('cohort', [sys.executable, '-m', 'research.rrg_f1.cohort', '--resume'], 7200, 'cohort_summary.json')
            run('cohort_audit', [sys.executable, '-m', 'research.rrg_f1.audit_cohort'], 600, 'cohort_audit_summary.json')
            run('labeler_readiness', [sys.executable, '-m', 'research.rrg_f1.readiness'], 120)
            run('target_cache', [sys.executable, '-m', 'research.rrg_f1.prepare_targets'], 7200)
            run('label_benchmark', [sys.executable, '-m', 'research.rrg_f1.labeler', '--stage', 'benchmark'], 600, 'label_benchmark_summary.json')
            benchmark = json.loads((BASE / 'label_benchmark_summary.json').read_text())
            if benchmark['status'] != 'PASS' or benchmark['projected_30000_seconds'] > 28800:
                raise RuntimeError('STOP_LABELER_ESTIMATE_GT_8H')
            run('train_labels', [sys.executable, '-m', 'research.rrg_f1.labeler', '--stage', 'train'], 28800, 'train_label_summary.json')
            run('encodings', [python, '-m', 'research.rrg_f1.data'], 3600, 'encoding_summary.json')
            run('features', [python, '-m', 'research.rrg_f1.features'], 12000, 'feature_summary.json', gpu=True)
            for arm in config['planned_arms']:
                run(arm, [python, '-m', 'research.rrg_f1.train', '--arm', arm], 14400, arm + '_summary.json', gpu=True)
            arms = {arm: json.loads((BASE / (arm + '_summary.json')).read_text()) for arm in config['planned_arms']}
            if all(item['status'] == 'COLLAPSE' for item in arms.values()):
                raise RuntimeError('STOP_COLLAPSE_ALL_3_ARMS')
            for arm in config['planned_arms']:
                run('eval_' + arm, [python, '-m', 'research.rrg_f1.infer', '--arm', arm], 14400, arm + '_baseline_summary.json', gpu=True)
            run('interventions', [python, '-m', 'research.rrg_f1.infer', '--arm', 'A4', '--interventions'], 3600,
                'A4_shuffled_summary.json', gpu=True)
            run('metrics', [local['labeler_python'], '-m', 'research.rrg_f1.evaluate'], 28800, 'metrics.json', legacy=True)
            run('intervention_analysis', [python, '-m', 'research.rrg_f1.intervention_analysis'], 7200, 'intervention_summary.json')
            status['status'] = 'COMPLETE'
        except (OSError, ValueError, RuntimeError) as error:
            status.update(status='STOP', reason=str(error) if str(error).startswith('STOP_') else type(error).__name__)
            raise
        finally:
            atomic_json(status_path, status)
            atomic_json(BASE / 'execution_summary.json', {'status': status['status'], 'reason': status.get('reason'),
                'stages': [{k: value for k, value in attempt.items() if k in ('stage', 'status', 'wall_seconds', 'returncode')} for attempt in status['attempts']]})


if __name__ == '__main__':
    main()
