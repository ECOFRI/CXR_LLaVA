"""Read P5 state or safely request bounded termination."""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

VR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VR / 'src'))
from vlm_medical.train.p4_runtime import atomic_json

RUN = VR / 'runs/P5_qformer_20261025'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    path = RUN / 'status.json'
    if not path.exists():
        print(json.dumps({'status': 'PENDING', 'run_id': RUN.name}))
        return
    state = json.loads(path.read_text())
    if args.stop and state.get('status') == 'RUNNING':
        child = state.get('child_pid')
        if isinstance(child, int):
            try:
                os.killpg(child, signal.SIGTERM)
            except ProcessLookupError:
                pass
        supervisor = state.get('supervisor_pid')
        if isinstance(supervisor, int):
            try:
                os.kill(supervisor, signal.SIGTERM)
            except ProcessLookupError:
                pass
        state.update({'status': 'HOLD', 'reason': 'operator safe termination',
                      'heartbeat_utc_ns': time.time_ns()})
        atomic_json(path, state)
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
