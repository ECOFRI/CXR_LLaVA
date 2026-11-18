"""CPU-only guard refusal tests: cumulative and per-arm limits survive resume."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from research.rrg_f0 import gpu_guard


class BudgetTests(unittest.TestCase):
    def refuse(self, charged, stage):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ledger.json'
            ledger = {'limit_seconds': 50400, 'attempts': [{'stage': stage, 'charged_seconds': charged}]}
            path.write_text(json.dumps(ledger))
            argv = ['guard', '--stage', 'A2', '--ledger', str(path), '--limit-seconds', '50400',
                    '--seconds', '60', '--stage-total-seconds', '14400', 'unused_command']
            with patch('sys.argv', argv), patch.object(gpu_guard.subprocess, 'check_output', return_value='Quadro RTX 4000, 1'):
                with self.assertRaisesRegex(RuntimeError, 'BUDGET_EXHAUSTED'):
                    gpu_guard.main()
            self.assertEqual(json.loads(path.read_text()), ledger)

    def test_global_budget_refuses_after_prior_stage(self):
        self.refuse(50390, 'features')

    def test_arm_budget_refuses_cumulative_resumed_time(self):
        self.refuse(14390, 'A2')


if __name__ == '__main__':
    unittest.main()
