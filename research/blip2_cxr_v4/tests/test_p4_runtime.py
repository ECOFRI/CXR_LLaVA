"""CPU-only tests for P4 execution bookkeeping; no model or GPU is used."""

import json
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

VR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VR / 'src'))

from vlm_medical.train.p4_runtime import (BudgetExceeded, BudgetLedger, ExecutionLease,
                                           LeaseBusy, atomic_json, identity_matches,
                                           process_identity)


class P4RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=VR / '.cache')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_atomic_status_remains_complete_during_replacement(self):
        path = self.root / 'status.json'
        atomic_json(path, {'status': 'PLANNED', 'sequence': 0})
        stop = threading.Event()
        errors = []

        def read_during_writes():
            while not stop.is_set():
                try:
                    record = json.loads(path.read_text())
                    if 'sequence' not in record or 'status' not in record:
                        errors.append('Incomplete status')
                except (OSError, ValueError) as exc:
                    errors.append(str(exc))

        reader = threading.Thread(target=read_during_writes)
        reader.start()
        for sequence in range(1, 50):
            atomic_json(path, {'status': 'RUNNING', 'sequence': sequence})
            self.assertEqual(json.loads(path.read_text())['sequence'], sequence)
        stop.set()
        reader.join(timeout=5)
        self.assertFalse(reader.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(list(self.root.glob('.status.json.*')), [])

    def test_cross_process_lease_exclusion(self):
        lock = self.root / 'gpu.lock'
        with ExecutionLease(lock):
            program = """import sys
from vlm_medical.train.p4_runtime import ExecutionLease, LeaseBusy
try:
    with ExecutionLease(sys.argv[1]):
        sys.exit(3)
except LeaseBusy:
    sys.exit(0)
"""
            env = {**os.environ, 'PYTHONPATH': str(VR / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
            child = subprocess.run([sys.executable, '-c', program, str(lock)], env=env,
                                   capture_output=True, text=True, timeout=10)
            self.assertEqual(child.returncode, 0, child.stderr)
        with ExecutionLease(lock) as lease:
            self.assertEqual(json.loads(lock.read_text()), lease.identity)

    def test_budget_persists_across_attempts_and_reopen(self):
        lock = self.root / 'gpu.lock'
        path = self.root / 'budget.json'
        with ExecutionLease(lock) as lease:
            ledger = BudgetLedger(path, 2.0, lease)
            ledger.start_attempt('one')
            time.sleep(0.02)
            after_first = ledger.finish_attempt()
            self.assertLess(after_first, 2.0)
            reopened = BudgetLedger(path, 2.0, lease)
            self.assertLess(reopened.remaining_seconds(), 2.0)
            reopened.start_attempt('two')
            time.sleep(0.02)
            after_second = reopened.finish_attempt()
            self.assertLess(after_second, after_first)
            with self.assertRaises(ValueError):
                BudgetLedger(path, 3.0, lease)
        self.assertEqual(len(json.loads(path.read_text())['attempts']), 2)

    def test_budget_expiry_uses_live_monotonic_time(self):
        with ExecutionLease(self.root / 'gpu.lock') as lease:
            ledger = BudgetLedger(self.root / 'budget.json', 0.01, lease)
            ledger.start_attempt('expiry')
            time.sleep(0.03)
            with self.assertRaises(BudgetExceeded):
                ledger.check_remaining()
            self.assertEqual(ledger.finish_attempt(), 0.0)
            self.assertGreater(ledger.data['attempts'][0]['charged_seconds'], 0.01)

    def test_audited_budget_amendment_preserves_attempts_and_rejects_wrong_hash(self):
        path = self.root / 'budget.json'
        with ExecutionLease(self.root / 'gpu.lock') as lease:
            ledger = BudgetLedger(path, 1.0, lease)
            ledger.start_attempt('first')
            time.sleep(0.01)
            ledger.finish_attempt()
            original_charge = ledger.data['attempts'][0]['charged_seconds']
            prior_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'Prior ledger SHA256 mismatch'):
                ledger.apply_amendment(2.0, 'a' * 64, 'b' * 64)
            self.assertEqual(json.loads(path.read_text())['budget_seconds'], 1.0)
            ledger.apply_amendment(2.0, 'a' * 64, prior_hash)
            reopened = BudgetLedger(path, 2.0, lease)
            self.assertEqual(reopened.data['attempts'][0]['charged_seconds'], original_charge)
            self.assertEqual(reopened.data['amendments'][0]['prior_ledger_sha256'], prior_hash)
            self.assertEqual(reopened.data['amendments'][0]['amendment_sha256'], 'a' * 64)
            with self.assertRaises(ValueError):
                BudgetLedger(path, 1.0, lease)

    def test_stale_process_is_charged_through_recovery(self):
        lock = self.root / 'gpu.lock'
        path = self.root / 'budget.json'
        program = """import sys
from vlm_medical.train.p4_runtime import BudgetLedger, ExecutionLease
with ExecutionLease(sys.argv[1]) as lease:
    BudgetLedger(sys.argv[2], 1.0, lease).start_attempt('crash')
"""
        env = {**os.environ, 'PYTHONPATH': str(VR / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
        child = subprocess.run([sys.executable, '-c', program, str(lock), str(path)], env=env,
                               capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr)
        time.sleep(0.03)
        with ExecutionLease(lock) as lease:
            ledger = BudgetLedger(path, 1.0, lease)
            ledger.start_attempt('retry')
            previous = ledger.data['attempts'][0]
            self.assertEqual(previous['status'], 'INTERRUPTED')
            self.assertGreater(previous['charged_seconds'], 0.03)
            self.assertLess(ledger.remaining_seconds(), 0.97)
            ledger.finish_attempt()

    def test_process_start_mismatch_and_lease_release(self):
        identity = process_identity()
        self.assertTrue(identity_matches(identity))
        self.assertFalse(identity_matches({**identity, 'process_start_ticks':
                                            identity['process_start_ticks'] + 1}))
        self.assertFalse(identity_matches({**identity, 'boot_id': 'different-boot'}))
        with ExecutionLease(self.root / 'gpu.lock') as lease:
            lease.identity = {**identity, 'process_start_ticks': identity['process_start_ticks'] + 1}
            with self.assertRaises(RuntimeError):
                lease.require_owner()
        with self.assertRaises(RuntimeError):
            lease.require_owner()

    def test_failed_lease_entry_releases_lock(self):
        lock = self.root / 'gpu.lock'
        with patch('vlm_medical.train.p4_runtime.process_identity', side_effect=RuntimeError('probe failed')):
            with self.assertRaisesRegex(RuntimeError, 'probe failed'):
                with ExecutionLease(lock):
                    pass
        with patch('vlm_medical.train.p4_runtime.os.fsync', side_effect=OSError('sync failed')):
            with self.assertRaisesRegex(OSError, 'sync failed'):
                with ExecutionLease(lock):
                    pass
        with ExecutionLease(lock) as lease:
            self.assertEqual(json.loads(lock.read_text()), lease.identity)


if __name__ == '__main__':
    unittest.main()
