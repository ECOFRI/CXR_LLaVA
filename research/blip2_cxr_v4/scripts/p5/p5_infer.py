"""Generate only M3 reports on the frozen P4 validation/Indiana cohort."""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from p5_common import ROOT, VR, configure_model, contract_and_index, digest, expected_names
from vlm_medical.train.p4_runtime import BudgetLedger, ExecutionLease, atomic_json

sys.path.insert(0, str(VR / 'scripts/experiments'))
from p4_paired_infer import (continuation_tokens, generate_single_sequence, paired_cohort,
                             reference_for, stable_key)


def _timeout(signum, frame):
    raise TimeoutError('P5.3 cumulative inference watchdog reached')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--manifest-index', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--gpu-index', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    cfg, index = contract_and_index(a.config, a.manifest_index)
    if not a.run_dir.resolve().is_relative_to((VR / 'runs').resolve()):
        raise ValueError('P5 output outside run root')
    rows = paired_cohort(index)
    if len(rows) != 1000 or len({stable_key(x) for x in rows}) != 1000:
        raise ValueError('Frozen paired cohort invalid')
    p4run = VR / 'runs/P4_projection_pilot_20261023'
    p4pairs = [json.loads(x) for x in (p4run / 'P4.3_pairs.jsonl').read_text().splitlines()]
    if len(p4pairs) != len(rows) or any(tuple(p4pairs[i]['stable_key']) != stable_key(rows[i]) for i in range(len(rows))):
        raise ValueError('Saved P4 M0/M1 baseline does not match P5 ordered cohort')
    trained = a.run_dir / 'P5.2_last.pt'
    stage2 = a.run_dir / 'P5.2_status.json'
    if a.dry_run:
        print(json.dumps({'status': 'DRY_RUN_PASS', 'studies': len(rows), 'new_generations': 1000,
                          'reused_p4_pairs': len(p4pairs), 'config_sha256': digest(a.config)}))
        return
    prior = json.loads(stage2.read_text())
    if (prior['status'] != 'PASS' or prior['optimizer_updates'] != 2000
            or not prior['checkpoint_reload_verified'] or prior['checkpoint_sha256'] != digest(trained)):
        raise ValueError('P5.2 final checkpoint gate failed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(a.gpu_index):
        raise ValueError('Exactly selected GPU must be visible')
    reading = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu_index), '--query-gpu=uuid,memory.used',
                                       '--format=csv,noheader,nounits'], text=True).strip()
    gpu_uuid, used = [part.strip() for part in reading.split(',')]
    if int(used) != 0:
        raise RuntimeError('GPU occupied before P5.3')
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    from vlm_medical.data.preprocess import load_image
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'K80' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one visible K80 required')
    run = a.run_dir
    run.mkdir(parents=True, exist_ok=True)
    pairs_path = run / 'P5.3_predictions.jsonl'
    stage_path = run / 'P5.3_status.json'
    with ExecutionLease(VR / 'runs/.p4_gpu_job.lock') as lease:
        ledger = BudgetLedger(run / 'P5.3_budget.json', cfg['budgets']['p5_3_wall_seconds_cumulative'], lease)
        remaining = ledger.check_remaining(required_seconds=60)
        attempt = ledger.start_attempt(cfg['run_id'] + '_P5.3')
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, max(1, remaining - 30))
        start = time.monotonic()
        saved = [json.loads(x) for x in pairs_path.read_text().splitlines()] if pairs_path.exists() else []
        if len(saved) > len(rows):
            raise ValueError('Too many P5 predictions')
        for i, item in enumerate(saved):
            if (tuple(item['stable_key']) != stable_key(rows[i]) or item['model_id'] != 'M3'
                    or item['checkpoint_sha256'] != prior['checkpoint_sha256']
                    or item['config_sha256'] != digest(a.config) or not item['generated_report'].strip()):
                raise ValueError('Existing P5 output identity/validity mismatch')
        status = {'status': 'RUNNING', 'stage': 'P5.3', 'run_id': cfg['run_id'], 'pid': os.getpid(),
                  'attempt_id': attempt, 'gpu_uuid': gpu_uuid, 'config_sha256': digest(a.config),
                  'checkpoint_sha256': prior['checkpoint_sha256'], 'completed_generations': len(saved),
                  'validity_gate': 'PASS' if len(saved) >= 10 else 'PENDING'}
        atomic_json(stage_path, status)
        try:
            checkpoint = ROOT / cfg['model']['checkpoint_dir']
            proc = Blip2Processor.from_pretrained(checkpoint, local_files_only=True, use_fast=False)
            model, loading = Blip2ForConditionalGeneration.from_pretrained(checkpoint,
                local_files_only=True, torch_dtype=torch.float16, output_loading_info=True)
            if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('error_msgs'):
                raise RuntimeError('Pinned base-model load mismatch')
            configure_model(model)
            payload = torch.load(trained, map_location='cpu')
            names = expected_names(model)
            if (payload['stage'] != 'P5.2' or payload['config_sha256'] != digest(a.config)
                    or set(payload['adapter']) != names):
                raise ValueError('P5.2 adapter identity mismatch')
            named = dict(model.named_parameters())
            for name in names:
                named[name].data.copy_(payload['adapter'][name].to(named[name].dtype))
            del payload
            model.to('cuda:0').eval()
            with Path(rows[5]['reference_source']).open(newline='', encoding='utf-8-sig') as stream:
                x00 = list(csv.DictReader(stream))
            with pairs_path.open('a', encoding='utf-8') as output:
                for i in range(len(saved), len(rows)):
                    if ledger.remaining_seconds() < 45:
                        status['stop_reason'] = 'inference budget nearly exhausted'
                        break
                    row = rows[i]
                    reference_for(row, x00)
                    gen = cfg['generation']
                    batch = proc(images=load_image(row['image_path']), text=gen['prompt'], return_tensors='pt')
                    inputs = {k: v.to('cuda:0', dtype=torch.float16) if k == 'pixel_values' else v.to('cuda:0')
                              for k, v in batch.items()}
                    torch.cuda.synchronize(); tick = time.monotonic()
                    with torch.no_grad():
                        tokens = generate_single_sequence(model, inputs, gen)
                    torch.cuda.synchronize()
                    continuation = continuation_tokens(tokens, gen['bos_token_id'])
                    report = proc.tokenizer.batch_decode(continuation, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)[0].strip()
                    if not report:
                        raise RuntimeError('Empty M3 generation at frozen position ' + str(i))
                    entry = {'run_id': cfg['run_id'], 'model_id': 'M3', 'stable_key': list(stable_key(row)),
                             'study_id': row['study_id'], 'image_id': row['image_id'],
                             'generated_report': report, 'valid_nonempty': True,
                             'generated_tokens': int(continuation.shape[1]),
                             'hit_max_length': bool(continuation.shape[1] >= gen['max_new_tokens']
                                 and continuation[0, -1].item() != gen['eos_token_id']),
                             'latency_seconds': time.monotonic() - tick,
                             'config_sha256': digest(a.config), 'checkpoint_sha256': prior['checkpoint_sha256']}
                    output.write(json.dumps(entry, ensure_ascii=False) + '\n'); output.flush(); os.fsync(output.fileno())
                    saved.append(entry)
                    peak = torch.cuda.max_memory_reserved() / 2**30
                    if peak > cfg['budgets']['peak_reserved_gib_limit']:
                        raise MemoryError('P5.3 peak reserved exceeded 10.0 GiB')
                    status.update({'completed_generations': len(saved), 'peak_reserved_gib': peak,
                                   'heartbeat_utc_ns': time.time_ns()})
                    if len(saved) == 10:
                        status['validity_gate'] = 'PASS'
                    if len(saved) % 10 == 0:
                        atomic_json(stage_path, status)
            status.update({'status': 'PASS' if len(saved) == 1000 else 'PARTIAL',
                           'completed_generations': len(saved),
                           'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                           'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                           'predictions_sha256': digest(pairs_path),
                           'wall_seconds_this_attempt': time.monotonic() - start})
        except BaseException as exc:
            status.update({'status': 'PARTIAL' if saved else 'FAILED',
                           'reason': type(exc).__name__ + ': ' + str(exc),
                           'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                           'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                           'wall_seconds_this_attempt': time.monotonic() - start})
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            ledger.finish_attempt()
            status['cumulative_charged_seconds'] = sum(x.get('charged_seconds', 0) for x in ledger.data['attempts'])
            atomic_json(stage_path, status)


if __name__ == '__main__':
    main()
