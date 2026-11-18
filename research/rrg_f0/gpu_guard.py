"""Run one local GPU command under a cumulative, crash-conservative budget (default 1h)."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    parser.add_argument('--seconds', type=int, required=True)
    parser.add_argument('--ledger', type=Path, default=BASE / 'gpu_budget.json')
    parser.add_argument('--limit-seconds', type=int, default=3600)
    parser.add_argument('--stage-total-seconds', type=int)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or not 10 < args.seconds <= args.limit_seconds - 30:
        parser.error('A command and an allowance below the cumulative cap are required')
    reading = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    gpu_name, used_mib = reading.rsplit(',', 1)
    if 'RTX 4000' not in gpu_name or int(used_mib.strip()) > 50:
        raise RuntimeError('Authorized GPU unavailable or occupied')
    ledger_path = args.ledger.resolve()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with (ledger_path.parent / '.gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'limit_seconds': args.limit_seconds, 'attempts': []}
        if ledger['limit_seconds'] != args.limit_seconds:
            raise ValueError('Existing budget cap cannot change')
        used = sum(a['charged_seconds'] for a in ledger['attempts'])
        allowance = min(args.seconds, int(args.limit_seconds - used - 15))
        if args.stage_total_seconds is not None:
            stage_used = sum(a['charged_seconds'] for a in ledger['attempts'] if a['stage'] == args.stage)
            allowance = min(allowance, int(args.stage_total_seconds - stage_used - 15))
        if allowance <= 10:
            raise RuntimeError('STOP_GPU_BUDGET_EXHAUSTED')
        attempt = {'stage': args.stage, 'status': 'RESERVED', 'charged_seconds': allowance + 10, 'allowance_seconds': allowance}
        ledger['attempts'].append(attempt)
        ledger['charged_seconds_total'] = sum(a['charged_seconds'] for a in ledger['attempts'])
        temporary = ledger_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(ledger, indent=2) + '\n')
        temporary.replace(ledger_path)
        start = time.monotonic()
        child = None
        try:
            child = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=5s', str(allowance - 5), *args.command], start_new_session=True)
            attempt['returncode'] = child.wait(timeout=allowance)
            attempt['status'] = 'PASS' if child.returncode == 0 else 'STOP_CHILD_FAILED'
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if child is not None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            attempt['status'] = 'STOP_TIMEOUT_OR_INTERRUPT'
            raise
        finally:
            attempt['charged_seconds'] = time.monotonic() - start
            ledger['charged_seconds_total'] = sum(a['charged_seconds'] for a in ledger['attempts'])
            temporary = ledger_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(ledger, indent=2) + '\n')
            temporary.replace(ledger_path)
        if attempt['returncode']:
            raise SystemExit(attempt['returncode'])


if __name__ == '__main__':
    main()
