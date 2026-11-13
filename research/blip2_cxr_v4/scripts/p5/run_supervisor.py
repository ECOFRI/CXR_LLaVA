"""Detached, bounded P5 state machine. Launch once with setsid/nohup."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from p5_common import ROOT, VR, contract_and_index, digest
from vlm_medical.train.p4_runtime import atomic_json

PYTHON = Path(sys.executable)
SCRIPT = Path(__file__).resolve().parent
RUN = VR / 'runs/P5_qformer_20261025'
CONFIG = VR / 'configs/experiments/P5_qformer_contract.approved_20261025.json'
INDEX = VR / 'runs/P4_projection_pilot_20261023/manifest_index.approved.json'
STAGES = ('PREFLIGHT', 'CODE_READY', 'P5.1', 'P5.2', 'P5.3', 'CPU_ANALYSIS', 'COMPLETE')
TERMINAL = ('COMPLETE', 'HOLD', 'BLOCKED', 'FAILED')


def _code_hashes():
    return {str(path.relative_to(ROOT)): digest(path) for path in
            [*sorted(SCRIPT.glob('*.py')), SCRIPT / 'run_all.sh', VR / 'tests/test_p5_cpu.py']}


def _status(stage, state, **data):
    existing = json.loads((RUN / 'status.json').read_text()) if (RUN / 'status.json').exists() else {}
    existing.update({'run_id': 'P5_qformer_20261025', 'stage': stage, 'status': state,
                     'heartbeat_utc_ns': time.time_ns(), 'supervisor_pid': os.getpid(),
                     'config_sha256': digest(CONFIG), 'git_commit': subprocess.check_output(
                         ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), **data})
    atomic_json(RUN / 'status.json', existing)
    return existing


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _gpu_used_mib(index=0):
    raw = subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=memory.used',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    return int(raw.splitlines()[0].strip())


def _process_usage(pid):
    status = Path('/proc') / str(pid) / 'status'
    rss = 0.0
    cpu = 0.0
    try:
        status_text = status.read_text()
    except FileNotFoundError:
        status_text = ''
    if status_text:
        for line in status_text.splitlines():
            if line.startswith('VmRSS:'):
                rss = int(line.split()[1]) / 1024
                break
    stat = Path('/proc') / str(pid) / 'stat'
    try:
        stat_text = stat.read_text()
    except FileNotFoundError:
        stat_text = ''
    if stat_text:
        fields = stat_text.rsplit(')', 1)[1].split()
        cpu = (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
    return rss, cpu


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        if not _pid_alive(pid):
            return
        time.sleep(0.5)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_child(stage, argv, timeout_seconds, gpu):
    log = RUN / (stage + '.stdout.log')
    err = RUN / (stage + '.stderr.log')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0' if gpu else ''
    env['HF_HOME'] = str(VR / '.cache/hf')
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    env['TMPDIR'] = str(VR / '.cache/tmp')
    env['XDG_CACHE_HOME'] = str(VR / '.cache')
    (VR / '.cache/tmp').mkdir(parents=True, exist_ok=True)
    budget_path = RUN / (stage + '_supervisor_budget.json') if gpu else None
    attempts = json.loads(budget_path.read_text())['attempts'] if budget_path is not None and budget_path.exists() else []
    prior_seconds = sum(x['charged_seconds'] for x in attempts)
    if gpu and prior_seconds >= timeout_seconds:
        raise RuntimeError(stage + ' cumulative supervisor wall budget exhausted')
    effective_timeout = timeout_seconds - prior_seconds if gpu else timeout_seconds
    started = time.monotonic()
    with log.open('a') as output, err.open('a') as errors:
        process = subprocess.Popen([str(PYTHON), *map(str, argv)], cwd=ROOT, env=env,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
                                   start_new_session=True)
        _status(stage, 'RUNNING', child_pid=process.pid, stdout_log=str(log), stderr_log=str(err),
                stage_started_utc_ns=time.time_ns())
        peak_used = 0
        peak_rss = 0.0
        child_cpu = 0.0
        reason = None
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= effective_timeout:
                reason = 'supervisor hard stage timeout'
                _kill_group(process.pid)
                break
            if gpu:
                try:
                    used = _gpu_used_mib()
                    peak_used = max(peak_used, used)
                    if used > 11000:
                        reason = 'nvidia-smi GPU used memory exceeded physical-capacity emergency guard'
                        _kill_group(process.pid)
                        break
                except (OSError, subprocess.CalledProcessError, ValueError) as exc:
                    reason = 'GPU monitor failed: ' + str(exc)
                    _kill_group(process.pid)
                    break
            rss, cpu = _process_usage(process.pid)
            peak_rss = max(peak_rss, rss)
            child_cpu = max(child_cpu, cpu)
            _status(stage, 'RUNNING', child_pid=process.pid, monitor_peak_gpu_used_mib=peak_used,
                    monitor_peak_child_rss_mib=peak_rss, child_cpu_seconds=child_cpu,
                    elapsed_stage_seconds=elapsed)
            time.sleep(2)
        code = process.wait(timeout=20)
    _status(stage, 'RUNNING', child_pid=None, child_exit_code=code,
            monitor_peak_gpu_used_mib=peak_used, monitor_peak_child_rss_mib=peak_rss,
            child_cpu_seconds=child_cpu,
            elapsed_stage_seconds=time.monotonic() - started)
    atomic_json(RUN / (stage + '_resource_monitor.json'), {'stage': stage, 'child_pid': process.pid,
        'exit_code': code, 'peak_gpu_used_mib_external': peak_used,
        'peak_child_rss_mib': peak_rss, 'child_cpu_seconds': child_cpu,
        'supervised_wall_seconds': time.monotonic() - started,
        'monitor_stop_reason': reason})
    if gpu:
        attempts.append({'pid': process.pid, 'charged_seconds': time.monotonic() - started,
                         'exit_code': code, 'monitor_stop_reason': reason})
        atomic_json(budget_path, {'stage': stage, 'hard_limit_seconds': timeout_seconds, 'attempts': attempts})
    if reason:
        raise RuntimeError(reason)
    if code != 0:
        raise RuntimeError(stage + ' process exit ' + str(code) + '; inspect ' + str(err))


def _preflight():
    cfg, index = contract_and_index(CONFIG, INDEX)
    RUN.mkdir(parents=True, exist_ok=True, mode=0o700)
    if _gpu_used_mib() != 0:
        raise RuntimeError('K80 GPU0 occupied at preflight')
    if not PYTHON.is_file():
        raise FileNotFoundError('Existing K80 Python environment missing')
    expected = {'config': digest(CONFIG), 'index': digest(INDEX),
                'p4_adapter': digest(VR / 'runs/P4_projection_pilot_20261023/P4.2_last.pt'),
                'p4_metrics': digest(VR / 'runs/P4_projection_pilot_20261023/P4.3_evaluation_full/metrics.json')}
    if (expected['index'] != '74965a79b31394a65af10dcb7987db0881e5083a8bb0da888bccbe120bd03175'
            or expected['p4_adapter'] != '7f833eea69fb6d2743b19bbc808be949bb1a5e014bf43a81f08090f9a5f0d8af'
            or expected['p4_metrics'] != 'ed6be843294d14deafa1fd9c9366bf302a53c08f12f6804c530e680393fb9be5'):
        raise ValueError('Immutable P4 input hash mismatch')
    for name, item in index['manifests'].items():
        if digest(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen manifest changed: ' + name)
    from safetensors import safe_open
    ckpt = ROOT / cfg['model']['checkpoint_dir']
    mapping = json.loads((ckpt / 'model.safetensors.index.json').read_text())['weight_map']
    groups = {'qformer': 0, 'query_tokens': 0, 'language_projection': 0}
    for shard in sorted(set(mapping.values())):
        with safe_open(ckpt / shard, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                group = 'qformer' if key.startswith('qformer.') else 'query_tokens' if key == 'query_tokens' else 'language_projection' if key.startswith('language_projection.') else None
                if group:
                    size = 1
                    for dim in handle.get_slice(key).get_shape():
                        size *= dim
                    groups[group] += size
    if groups != {'qformer': 105137664, 'query_tokens': 24576, 'language_projection': 1968640}:
        raise ValueError('Pinned Q-Former/query/projection parameter metadata changed')
    expected['trainable_parameter_groups'] = groups
    return expected


def _adopt_existing_child(old):
    pid = old.get('child_pid') if old else None
    if not isinstance(pid, int) or not _pid_alive(pid):
        return
    stage = old.get('stage')
    limit = {'P5.1': 900, 'P5.2': 14400, 'P5.3': 10800}.get(stage, 120)
    started_ns = old.get('stage_started_utc_ns', time.time_ns())
    while _pid_alive(pid):
        elapsed = max(0.0, (time.time_ns() - started_ns) / 1e9)
        if elapsed >= limit:
            _kill_group(pid)
            raise RuntimeError('Adopted child exceeded original ' + stage + ' time cap')
        if stage in ('P5.1', 'P5.2', 'P5.3') and _gpu_used_mib() > 11000:
            _kill_group(pid)
            raise RuntimeError('Adopted child exceeded conservative 10 GiB memory guard')
        time.sleep(2)


def _tests_and_dry_runs():
    _run_child('CPU_TESTS', ['-m', 'unittest', 'discover', '-s', VR / 'tests', '-p', 'test_p5_cpu.py', '-v'], 120, False)
    _run_child('CPU_P4_PARITY_TESTS', ['-m', 'unittest', 'discover', '-s', VR / 'tests', '-p',
                                     'test_p4_projection_gate.py', '-v'], 120, False)
    for script, stage in [('p5_train.py', 'P5.1'), ('p5_train.py', 'P5.2'),
                          ('p5_infer.py', None), ('p5_evaluate.py', None)]:
        args = [SCRIPT / script]
        if stage:
            args.extend(['--stage', stage])
        args.extend(['--config', CONFIG, '--manifest-index', INDEX, '--run-dir', RUN, '--dry-run'])
        _run_child('DRY_RUN_' + script + ('_' + stage if stage else ''), args, 90, False)
    atomic_json(RUN / 'cpu_readiness.json', {'status': 'PASS', 'tests': 'P5 6/6, P4 parity 3/3',
                'dry_runs': 'P5.1/P5.2/P5.3/evaluation PASS', 'config_sha256': digest(CONFIG),
                'code_sha256': _code_hashes(),
                'gpu_used': False})


def _train_gate(stage, limit, count):
    status = json.loads((RUN / (stage + '_status.json')).read_text())
    ledger = json.loads((RUN / (stage + '_budget.json')).read_text())
    charged = sum(x.get('charged_seconds', 0) for x in ledger['attempts'])
    supervisor_budget = RUN / (stage + '_supervisor_budget.json')
    if supervisor_budget.exists():
        charged = max(charged, sum(x['charged_seconds'] for x in json.loads(supervisor_budget.read_text())['attempts']))
    checkpoint = RUN / (stage + '_last.pt')
    updates = [json.loads(x) for x in (RUN / (stage + '_updates.jsonl')).read_text().splitlines()]
    if (status['status'] != 'PASS' or status['optimizer_updates'] != count or len(updates) != count
            or [x['position'] for x in updates] != list(range(1, count + 1))
            or charged > limit or status['peak_reserved_gib'] > 10.0
            or not status['checkpoint_reload_verified'] or status['checkpoint_sha256'] != digest(checkpoint)
            or not status['frozen_versions_unchanged']):
        raise RuntimeError(stage + ' exact update/memory/time/checkpoint gate failed')
    if stage == 'P5.2' and status.get('p5_1_checkpoint_used') is not False and status.get('resumed_from_position') is None:
        raise RuntimeError('P5.2 original initialization not verified')
    if stage == 'P5.2' and status.get('epochs_complete') != 1:
        raise RuntimeError('P5.2 one-epoch completion not verified')
    return status, charged, updates


def _feasibility(status, charged, updates):
    durations = sorted(x['step_seconds'] for x in updates)
    p90 = durations[17]
    mean = sum(durations) / len(durations)
    estimated_overhead = max(1800.0, 2 * max(0.0, charged - sum(durations)))
    forecast = 2000 * max(p90, 1.2 * mean) + estimated_overhead
    result = {'p5_1_updates': 20, 'mean_step_seconds': mean, 'p90_step_seconds': p90,
              'estimated_p5_2_overhead_seconds': estimated_overhead,
              'forecast_p5_2_seconds': forecast, 'p5_2_budget_seconds': 14400,
              'feasible': forecast <= 14400}
    atomic_json(RUN / 'P5.1_feasibility.json', result)
    if not result['feasible']:
        status.update({'status': 'HOLD', 'technical_gate': 'PASS',
                       'reason': 'P5.2 throughput forecast exceeds 4-hour cap', **result})
        atomic_json(RUN / 'P5.1_status.json', status)
        raise RuntimeError('P5.1 throughput gate HOLD: projected P5.2 >14,400 s')
    return result


def _inference_gate():
    status = json.loads((RUN / 'P5.3_status.json').read_text())
    ledger = json.loads((RUN / 'P5.3_budget.json').read_text())
    charged = sum(x.get('charged_seconds', 0) for x in ledger['attempts'])
    supervisor_budget = RUN / 'P5.3_supervisor_budget.json'
    if supervisor_budget.exists():
        charged = max(charged, sum(x['charged_seconds'] for x in json.loads(supervisor_budget.read_text())['attempts']))
    pairs = [json.loads(x) for x in (RUN / 'P5.3_predictions.jsonl').read_text().splitlines()]
    if (status['status'] not in ('PASS', 'PARTIAL') or status['validity_gate'] != 'PASS'
            or status['completed_generations'] != len(pairs) or len(pairs) > 1000
            or charged > 10800 or status['peak_reserved_gib'] > 10.0
            or len({tuple(x['stable_key']) for x in pairs}) != len(pairs)
            or any(not x['generated_report'].strip() for x in pairs)):
        raise RuntimeError('P5.3 paired output/time/memory gate failed')
    return status


def _append_once(path, marker, text):
    path = Path(path)
    old = path.read_text() if path.exists() else ''
    if marker not in old:
        with path.open('a', encoding='utf-8') as stream:
            stream.write('\n' + text + '\n')


def _report(final_state, reason=None):
    lines = ['# P5 execution progress — 2026-10-25', '',
             'Run: `P5_qformer_20261025`; scientific configuration SHA-256 `' + digest(CONFIG) + '`.',
             'Status: **' + final_state + '**. P4 artifacts remain unchanged.']
    for stage in ('P5.1', 'P5.2', 'P5.3'):
        path = RUN / (stage + '_status.json')
        if path.exists():
            d = json.loads(path.read_text())
            lines.append('- ' + stage + ': ' + d.get('status', 'UNKNOWN') +
                         '; updates/generations=' + str(d.get('optimizer_updates', d.get('completed_generations', 'N/A')))
                         + '; peak reserved GiB=' + str(d.get('peak_reserved_gib', 'N/A')))
    if reason:
        lines.append('Stop reason: ' + reason)
    metrics_path = RUN / 'P5.3_cpu_analysis/metrics.json'
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        for dataset, value in metrics['datasets'].items():
            if 'm3_minus_m1_bleu4' in value:
                delta = value['m3_minus_m1_bleu4']
                lines.append(dataset + ': paired M3−M1 BLEU-4 ' + str(delta['value']) +
                             ', 95% CI ' + str(delta['ci95']) + '; clinical proxies require caution.')
    marker = 'P5_qformer_20261025 final ' + final_state
    summary = '## ' + marker + '\n\n' + '\n'.join(lines[2:]) + '\n'
    report = VR / 'reports/P5_progress.md'
    if not report.exists():
        report.write_text('# P5 execution progress\n')
    _append_once(report, marker, summary)
    _append_once(VR / 'reports/04_experiment_tracker.md', marker, summary)
    _append_once(VR / 'reports/05_results_and_decisions.md', marker, summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    lock_path = RUN / '.supervisor.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = json.loads((RUN / 'status.json').read_text()) if (RUN / 'status.json').exists() else None
        if old and old.get('status') == 'COMPLETE':
            return
        if old and old.get('status') in ('HOLD', 'BLOCKED', 'FAILED') and not args.resume:
            raise RuntimeError('Terminal P5 run requires explicit --resume and unchanged scientific config')
        try:
            _adopt_existing_child(old)
            _status('PREFLIGHT', 'RUNNING')
            frozen = _preflight()
            plan = {'run_id': 'P5_qformer_20261025',
                'status': 'FROZEN', 'config_sha256': digest(CONFIG), 'manifest_index_sha256': digest(INDEX),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source_hashes': frozen, 'p5_2_initialization': 'fresh original pretrained BLIP-2',
                'code_sha256': _code_hashes(),
                'stage_limits': {'P5.1': [20, 900, 10.0], 'P5.2': [2000, 14400, 10.0],
                                 'P5.3': [1000, 10800, 10.0]}}
            plan_path = RUN / 'execution_plan.json'
            if plan_path.exists():
                if json.loads(plan_path.read_text()) != plan:
                    raise ValueError('Frozen P5 execution plan changed on restart')
            else:
                atomic_json(plan_path, plan)
            _status('CODE_READY', 'RUNNING')
            if not (RUN / 'cpu_readiness.json').exists():
                _tests_and_dry_runs()
            else:
                ready = json.loads((RUN / 'cpu_readiness.json').read_text())
                if (ready['status'] != 'PASS' or ready['config_sha256'] != digest(CONFIG)
                        or ready['code_sha256'] != _code_hashes()):
                    raise ValueError('CPU readiness identity mismatch')
            for stage, script, limit in [('P5.1', 'p5_train.py', 900), ('P5.2', 'p5_train.py', 14400),
                                         ('P5.3', 'p5_infer.py', 10800)]:
                prior = RUN / (stage + '_status.json')
                already = json.loads(prior.read_text()) if prior.exists() else {}
                if already.get('status') != 'PASS':
                    if stage == 'P5.2':
                        s1, charged1, updates1 = _train_gate('P5.1', 900, 20)
                        _feasibility(s1, charged1, updates1)
                    if stage == 'P5.3':
                        _train_gate('P5.2', 14400, 2000)
                    args_child = [SCRIPT / script]
                    if script == 'p5_train.py':
                        args_child.extend(['--stage', stage])
                    args_child.extend(['--config', CONFIG, '--manifest-index', INDEX, '--run-dir', RUN, '--gpu-index', '0'])
                    try:
                        _run_child(stage, args_child, limit, True)
                    except RuntimeError:
                        if stage != 'P5.3':
                            raise
                        partial = RUN / 'P5.3_status.json'
                        if not partial.exists() or json.loads(partial.read_text()).get('validity_gate') != 'PASS':
                            raise
                        _inference_gate()  # A valid bounded partial intersection can still be analyzed.
                if stage in ('P5.1', 'P5.2'):
                    s, charged, updates = _train_gate(stage, limit, 20 if stage == 'P5.1' else 2000)
                    if stage == 'P5.1':
                        _feasibility(s, charged, updates)
                else:
                    result = _inference_gate()
                    if result['status'] == 'PARTIAL':
                        _status('P5.3', 'PARTIAL', reason='Inference ended within original budget; analyze paired intersection')
            _status('CPU_ANALYSIS', 'RUNNING')
            if not (RUN / 'P5.3_cpu_analysis/metrics.json').exists():
                _run_child('CPU_ANALYSIS', [SCRIPT / 'p5_evaluate.py', '--config', CONFIG,
                    '--manifest-index', INDEX, '--run-dir', RUN], 10800, False)
            metrics = json.loads((RUN / 'P5.3_cpu_analysis/metrics.json').read_text())
            terminal = 'COMPLETE' if metrics['status'] == 'PASS' else 'PARTIAL'
            _status(terminal, terminal, metrics_path=str(RUN / 'P5.3_cpu_analysis/metrics.json'))
            _report(terminal)
        except BaseException as exc:
            message = type(exc).__name__ + ': ' + str(exc)
            stage = json.loads((RUN / 'status.json').read_text()).get('stage', 'PREFLIGHT')
            state = 'HOLD' if stage in ('P5.1', 'P5.2', 'P5.3') else 'BLOCKED'
            _status(stage, state, reason=message)
            _report(state, message)
            raise


if __name__ == '__main__':
    main()
