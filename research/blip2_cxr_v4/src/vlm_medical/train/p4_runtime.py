"""Local execution lease and cumulative wall-time accounting for bounded runs."""

import fcntl
import hashlib
import json
import math
import os
import socket
import tempfile
import time
import uuid
from pathlib import Path


class LeaseBusy(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def atomic_json(path, data):
    """Replace a JSON file without exposing a partially written document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _process_start_ticks(pid):
    stat = Path('/proc') / str(pid) / 'stat'
    try:
        fields = stat.read_text().rsplit(')', 1)[1].split()
        return int(fields[19])  # Linux /proc stat field 22.
    except (FileNotFoundError, IndexError, ValueError, PermissionError):
        return None


def _boot_id():
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip() or None
    except (FileNotFoundError, PermissionError):
        return None


def process_identity():
    start_ticks = _process_start_ticks(os.getpid())
    boot_id = _boot_id()
    if start_ticks is None or boot_id is None:
        raise RuntimeError('Cannot verify current process and boot identity')
    return {'hostname': socket.gethostname(), 'pid': os.getpid(),
            'process_start_ticks': start_ticks, 'boot_id': boot_id}


def identity_matches(identity):
    boot_id = _boot_id()
    if (not isinstance(identity, dict) or identity.get('hostname') != socket.gethostname()
            or boot_id is None or identity.get('boot_id') != boot_id):
        return False
    pid = identity.get('pid')
    ticks = identity.get('process_start_ticks')
    return isinstance(pid, int) and isinstance(ticks, int) and _process_start_ticks(pid) == ticks


class ExecutionLease:
    """Exclusive flock lease; retain the lock file to avoid unlink/recreate races."""

    def __init__(self, path):
        self.path = Path(path)
        self.identity = None
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open('a+', encoding='utf-8')
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.identity = process_identity()
            handle.seek(0)
            handle.truncate()
            json.dump(self.identity, handle, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
            self._handle = handle
            return self
        except BaseException as exc:
            self.identity = None
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
            if isinstance(exc, BlockingIOError):
                raise LeaseBusy('Execution lease is held: ' + str(self.path)) from exc
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None

    def require_owner(self):
        if self._handle is None or self.identity != process_identity():
            raise RuntimeError('Current process does not own the execution lease')


class BudgetLedger:
    """One cumulative budget, with interrupted attempts charged through recovery."""

    def __init__(self, path, budget_seconds, lease):
        if (not isinstance(budget_seconds, (int, float)) or isinstance(budget_seconds, bool)
                or not math.isfinite(budget_seconds) or budget_seconds <= 0):
            raise ValueError('A finite positive budget_seconds is required')
        lease.require_owner()
        self.path = Path(path)
        self.lease = lease
        self.budget_seconds = float(budget_seconds)
        self._active_id = None
        self._started_monotonic = None
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get('schema_version') != 1 or data.get('budget_seconds') != self.budget_seconds:
                raise ValueError('Existing budget ledger identity or limit differs')
            self.data = data
        else:
            self.data = {'schema_version': 1, 'budget_seconds': self.budget_seconds, 'attempts': []}
            atomic_json(self.path, self.data)

    def _require_owner(self):
        self.lease.require_owner()

    def _running(self):
        attempts = self.data['attempts']
        return attempts[-1] if attempts and attempts[-1]['status'] == 'RUNNING' else None

    def _elapsed(self, attempt):
        if attempt['attempt_id'] == self._active_id:
            return max(0.0, time.monotonic() - self._started_monotonic)
        now_ns = time.time_ns()
        start_ns = attempt['started_utc_ns']
        if now_ns < start_ns:
            return self.budget_seconds  # Clock rollback cannot grant more time.
        return (now_ns - start_ns) / 1e9

    def remaining_seconds(self):
        self._require_owner()
        spent = sum(attempt.get('charged_seconds', 0.0) for attempt in self.data['attempts'])
        running = self._running()
        if running is not None:
            spent += self._elapsed(running)
        return max(0.0, self.budget_seconds - spent)

    def check_remaining(self, required_seconds=0):
        if (not isinstance(required_seconds, (int, float)) or isinstance(required_seconds, bool)
                or not math.isfinite(required_seconds) or required_seconds < 0):
            raise ValueError('required_seconds must be nonnegative')
        remaining = self.remaining_seconds()
        if remaining <= required_seconds:
            raise BudgetExceeded(f'Cumulative wall-time budget exhausted; remaining={remaining:.3f}s')
        return remaining

    def start_attempt(self, run_id):
        self._require_owner()
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('A nonempty run_id is required')
        running = self._running()
        if running is not None:
            if identity_matches(running['identity']):
                raise RuntimeError('Recorded attempt is still owned by a live process')
            running['charged_seconds'] = self._elapsed(running)
            running['ended_utc_ns'] = time.time_ns()
            running['status'] = 'INTERRUPTED'
            atomic_json(self.path, self.data)
        self.check_remaining()
        attempt = {'attempt_id': uuid.uuid4().hex, 'run_id': run_id,
                   'identity': self.lease.identity, 'started_utc_ns': time.time_ns(),
                   'status': 'RUNNING'}
        self.data['attempts'].append(attempt)
        self._active_id = attempt['attempt_id']
        self._started_monotonic = time.monotonic()
        atomic_json(self.path, self.data)
        return attempt['attempt_id']

    def finish_attempt(self):
        self._require_owner()
        running = self._running()
        if running is None or running['attempt_id'] != self._active_id:
            raise RuntimeError('No locally active attempt to finish')
        running['charged_seconds'] = self._elapsed(running)
        running['ended_utc_ns'] = time.time_ns()
        running['status'] = 'FINISHED'
        self._active_id = None
        self._started_monotonic = None
        atomic_json(self.path, self.data)
        return self.remaining_seconds()

    def apply_amendment(self, new_budget_seconds, amendment_sha256, prior_ledger_sha256):
        """Increase a stopped ledger's cap, retaining all earlier attempts."""
        self._require_owner()
        if self._running() is not None or self._active_id is not None:
            raise RuntimeError('Cannot amend a running budget attempt')
        if (not isinstance(new_budget_seconds, (int, float)) or isinstance(new_budget_seconds, bool)
                or not math.isfinite(new_budget_seconds) or new_budget_seconds <= self.budget_seconds):
            raise ValueError('Amended budget must be finite and larger than current budget')
        if (not isinstance(amendment_sha256, str) or len(amendment_sha256) != 64
                or not isinstance(prior_ledger_sha256, str) or len(prior_ledger_sha256) != 64):
            raise ValueError('Amendment and prior ledger SHA256 are required')
        actual_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if actual_sha256 != prior_ledger_sha256:
            raise ValueError('Prior ledger SHA256 mismatch')
        self.data.setdefault('amendments', []).append({
            'amendment_sha256': amendment_sha256,
            'prior_ledger_sha256': prior_ledger_sha256,
            'old_budget_seconds': self.budget_seconds,
            'new_budget_seconds': float(new_budget_seconds),
        })
        self.budget_seconds = float(new_budget_seconds)
        self.data['budget_seconds'] = self.budget_seconds
        atomic_json(self.path, self.data)
        return self.remaining_seconds()
