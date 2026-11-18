"""Read-only sanitized progress; never emit rows, reports, paths or case labels."""
import json
from research.rrg_f1.cohort import BASE, PRIVATE


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    state = read(PRIVATE / 'supervisor_status.json')
    heartbeat = read(PRIVATE / 'supervisor_heartbeat.json')
    ledger = read(PRIVATE / 'gpu_budget.json')
    attempts = ledger.get('attempts', [])
    progress = {'status': state.get('status', 'NOT_STARTED'), 'reason': state.get('reason'),
                'stage': heartbeat.get('stage'), 'heartbeat_unix': heartbeat.get('unix_time'),
                'gpu_completed_seconds': sum(a['charged_seconds'] for a in attempts if a['status'] != 'RESERVED'),
                'gpu_reserved_seconds': sum(a['charged_seconds'] for a in attempts if a['status'] == 'RESERVED'),
                'gpu_limit_seconds': 50400, 'arms': {}}
    for arm in ('A2', 'A3', 'A4'):
        summary = read(BASE / (arm + '_summary.json'))
        live = read(PRIVATE / arm / 'heartbeat.json')
        progress['arms'][arm] = {k: summary[k] for k in ('status', 'examples_seen', 'optimizer_updates') if k in summary}
        if not summary:
            progress['arms'][arm].update({k: live[k] for k in ('status', 'position', 'updates', 'lm_loss', 'bce_loss') if k in live})
    progress['feature_progress'] = {k: v for k, v in read(PRIVATE / 'feature_heartbeat.json').items() if k in ('family', 'new_features')}
    chunks = PRIVATE / 'train_label_chunks'
    progress['label_chunks_complete'] = sum(1 for _ in chunks.glob('chunk_*/done.json')) if chunks.exists() else 0
    print(json.dumps(progress), flush=True)


if __name__ == '__main__':
    main()
