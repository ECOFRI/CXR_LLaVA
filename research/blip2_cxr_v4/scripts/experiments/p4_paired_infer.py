"""P4.3 paired M0/M1 report inference on frozen validation and Indiana cohorts."""
import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from transformers import StoppingCriteria, StoppingCriteriaList

VR = Path(__file__).resolve().parents[2]
ROOT = VR.parents[1]
sys.path.insert(0, str(VR / 'src'))
sys.path.insert(0, str(VR / 'scripts/train'))
from p4_projection_gate import digest, read_contract, read_manifest, target_for
from p4_projection_train import _pretrained_projection_check
from vlm_medical.models.bridges import FP32Projection
from vlm_medical.train.p4_checkpoint import load_adapter
from vlm_medical.train.p4_runtime import BudgetLedger, ExecutionLease, atomic_json


def paired_cohort(index):
    cohorts = []
    for name, expected, split in [('mimic_validate', 500, 'validate'), ('indiana_x00', 500, 'external_x00')]:
        item = index['manifests'][name]
        rows = read_manifest(ROOT / item['path'], item['sha256'], expected, split)
        cohorts.append((name, rows))
    first = [cohorts[0][1][i] for i in range(5)] + [cohorts[1][1][i] for i in range(5)]
    rest = [cohorts[0][1][i] for i in range(5, 500)] + [cohorts[1][1][i] for i in range(5, 500)]
    return first + rest


def stable_key(row):
    return (row['dataset'], row['split'], row['subject_id'], row['study_id'], row['image_id'])


class SingleSequenceEOS(StoppingCriteria):
    """Stop one greedy sequence at the approved EOS without CUDA EOS reduction."""

    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[0] != 1:
            raise ValueError('P4.3 EOS compatibility requires batch size one')
        return bool(input_ids[0, -1].item() == self.eos_token_id)


def generate_single_sequence(model, inputs, gen):
    batch = inputs.get('pixel_values', inputs.get('input_ids', inputs.get('inputs_embeds')))
    if batch is None or batch.shape[0] != 1 or gen['batch_size'] != 1 or gen['do_sample'] or gen['num_beams'] != 1:
        raise ValueError('P4.3 EOS compatibility requires single-sequence greedy generation')
    language_model = getattr(model, 'language_model', model)
    original_eos = language_model.generation_config.eos_token_id
    language_model.generation_config.eos_token_id = None
    try:
        return model.generate(**inputs, max_new_tokens=gen['max_new_tokens'],
            do_sample=False, num_beams=1, use_cache=gen['use_cache'],
            repetition_penalty=gen['repetition_penalty'], no_repeat_ngram_size=gen['no_repeat_ngram_size'],
            bos_token_id=gen['bos_token_id'], eos_token_id=None, pad_token_id=gen['pad_token_id'],
            stopping_criteria=StoppingCriteriaList([SingleSequenceEOS(gen['eos_token_id'])]))
    finally:
        language_model.generation_config.eos_token_id = original_eos


def continuation_tokens(tokens, bos_token_id):
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 2 or tokens[0, 0].item() != bos_token_id:
        raise ValueError('Expected one synthetic BOS followed by generated tokens')
    return tokens[:, 1:]


def paired_record(row, run_id, config_hash, adapter_hash, outputs):
    if set(outputs) != {'M0', 'M1'}:
        raise ValueError('Both paired model outputs are required')
    return {'run_id': run_id, 'study_id': row['study_id'], 'image_id': row['image_id'],
            'stable_key': list(stable_key(row)), 'config_sha256': config_hash,
            'adapter_sha256': adapter_hash, 'outputs': outputs}


def reference_for(row, x00_cohort):
    if row['dataset'] == 'mimic':
        return target_for(row)
    source = x00_cohort[row['source_row']]
    if (source['report_id'] != row['study_id'] or source['image_id'] != row['image_id']
            or hashlib.sha256(source['ground_truth_report'].encode()).hexdigest() != row['target_text_sha256']):
        raise ValueError('Indiana reference stable-ID mismatch')
    if digest(row['image_path']) != row['expected_image_sha256']:
        raise ValueError('Indiana selected image hash mismatch')
    return source['ground_truth_report']


def previous_pairs(path, rows, run_id, config_hash, adapter_hash):
    if not path.exists():
        return []
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    if len(saved) > len(rows):
        raise ValueError('More predictions than frozen cohort')
    for i, row in enumerate(saved):
        if (tuple(row['stable_key']) != stable_key(rows[i]) or row['config_sha256'] != config_hash
                or row['adapter_sha256'] != adapter_hash or row['run_id'] != run_id
                or row['study_id'] != rows[i]['study_id'] or row['image_id'] != rows[i]['image_id']
                or set(row['outputs']) != {'M0', 'M1'}
                or any(row['outputs'][arm]['model_id'] != arm for arm in ('M0', 'M1'))):
            raise ValueError('Prediction prefix identity mismatch')
    if len(saved) >= 10 and not all(o['valid_nonempty'] for pair in saved[:10] for o in pair['outputs'].values()):
        raise ValueError('Saved validity gate has empty outputs')
    return saved


def budget_amendment(path, run_id, config_hash, index_hash, original_budget):
    if not path.exists():
        return None
    amendment = json.loads(path.read_text())
    expected = {'run_id': run_id, 'config_sha256': config_hash, 'manifest_index_sha256': index_hash,
                'original_budget_seconds': float(original_budget),
                'additional_budget_seconds': 10800.0, 'new_cumulative_budget_seconds': 18000.0,
                'completed_pairs_at_amendment': 474,
                'remaining_studies_by_dataset': {'mimic': 31, 'indiana': 495},
                'authorization': 'user 2026-10-25: hoàn tất 4.3 đi'}
    if any(amendment.get(key) != value for key, value in expected.items()):
        raise ValueError('P4.3 budget amendment identity or authorization mismatch')
    if (not isinstance(amendment.get('prior_ledger_sha256'), str)
            or not isinstance(amendment.get('prior_pairs_sha256'), str)):
        raise ValueError('P4.3 budget amendment lacks prior artifact hashes')
    return amendment


def _timeout(signum, frame):
    raise TimeoutError('P4.3 cumulative inference budget reached')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--manifest-index', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu-index', type=int, required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    config = read_contract(args.config)
    if not args.output_dir.resolve().is_relative_to((VR / 'runs').resolve()):
        raise ValueError('Output must be under V4 runs')
    index = json.loads(args.manifest_index.read_text())
    if not index['approved'] or index['status'] != 'APPROVED_P4_20261023':
        raise ValueError('Evaluation manifest index is not approved')
    rows = paired_cohort(index)
    if len(rows) != 1000 or len({stable_key(r) for r in rows}) != 1000:
        raise ValueError('Evaluation stable-ID cohort invalid')
    config_hash = digest(args.config)
    run = args.output_dir
    amendment_path = run / 'P4.3_budget_amendment_20261025.json'
    amendment = budget_amendment(amendment_path, config['run_id'], config_hash,
                                 digest(args.manifest_index), config['budgets']['p4_3_wall_seconds_cumulative'])
    train_status_path = run / 'P4.2_status.json'
    if args.dry_run:
        if amendment is not None:
            ledger_path = run / 'P4.3_budget.json'
            pairs_path = run / 'P4.3_pairs.jsonl'
            current_ledger = json.loads(ledger_path.read_text())
            if current_ledger['budget_seconds'] == amendment['original_budget_seconds']:
                if (digest(ledger_path) != amendment['prior_ledger_sha256']
                        or digest(pairs_path) != amendment['prior_pairs_sha256']
                        or len(pairs_path.read_text().splitlines()) != amendment['completed_pairs_at_amendment']):
                    raise ValueError('P4.3 pre-amendment artifacts changed')
            elif (current_ledger['budget_seconds'] != amendment['new_cumulative_budget_seconds']
                  or not current_ledger.get('amendments')
                  or current_ledger['amendments'][-1]['amendment_sha256'] != digest(amendment_path)):
                raise ValueError('P4.3 amended ledger identity mismatch')
        print(json.dumps({'status': 'DRY_RUN_PASS', 'paired_studies': len(rows), 'validity_gate_studies': 10,
                          'mimic_split': 'validate', 'locked_test_used': False, 'config_sha256': config_hash,
                          'budget_amendment_sha256': digest(amendment_path) if amendment else None}))
        return
    train_status = json.loads(train_status_path.read_text())
    adapter_path = run / 'P4.2_last.pt'
    if (train_status.get('status') != 'PASS' or train_status.get('completed_studies') != 2000
            or train_status.get('epochs_complete') != 1 or not train_status.get('final_reload_verified')
            or not adapter_path.is_file() or train_status.get('final_checkpoint_sha256') != digest(adapter_path)
            or train_status.get('config_sha256') != config_hash):
        raise RuntimeError('P4.2 PASS and exact final checkpoint identity required before P4.3')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu_index):
        raise ValueError('CUDA_VISIBLE_DEVICES does not map the requested physical GPU')
    reading = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu_index), '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    gpu_uuid, used_mb = [part.strip() for part in reading.split(',')]
    if int(used_mb) > 0:
        raise RuntimeError('Selected physical GPU is occupied')
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    from vlm_medical.data.preprocess import load_image
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'K80' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one Tesla K80 required')
    with ExecutionLease(VR / 'runs/.p4_gpu_job.lock') as lease:
        ledger_path = run / 'P4.3_budget.json'
        ledger_cap = json.loads(ledger_path.read_text())['budget_seconds'] if amendment else config['budgets']['p4_3_wall_seconds_cumulative']
        if amendment and ledger_cap not in (amendment['original_budget_seconds'], amendment['new_cumulative_budget_seconds']):
            raise ValueError('P4.3 ledger cap differs from approved amendment')
        ledger = BudgetLedger(ledger_path, ledger_cap, lease)
        adapter_hash = digest(run / 'P4.2_last.pt')
        predictions_path = run / 'P4.3_pairs.jsonl'
        if amendment:
            saved_before = previous_pairs(predictions_path, rows, config['run_id'], config_hash, adapter_hash)
            amendment_sha = digest(amendment_path)
            if ledger_cap == amendment['original_budget_seconds']:
                if (len(saved_before) != amendment['completed_pairs_at_amendment']
                        or digest(predictions_path) != amendment['prior_pairs_sha256']):
                    raise ValueError('P4.3 prediction prefix changed before amendment')
                ledger.apply_amendment(amendment['new_cumulative_budget_seconds'], amendment_sha,
                                       amendment['prior_ledger_sha256'])
            elif (not ledger.data.get('amendments')
                  or ledger.data['amendments'][-1]['amendment_sha256'] != amendment_sha
                  or ledger.data['amendments'][-1]['prior_ledger_sha256'] != amendment['prior_ledger_sha256']):
                raise ValueError('P4.3 ledger amendment audit mismatch')
        remaining = ledger.check_remaining(required_seconds=30)
        attempt = ledger.start_attempt(config['run_id'] + '_P4.3')
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, max(1, remaining - 30))
        started = time.monotonic()
        if not predictions_path.exists():
            os.close(os.open(str(predictions_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        saved = previous_pairs(predictions_path, rows, config['run_id'], config_hash, adapter_hash)
        status = {'status': 'RUNNING', 'stage': 'P4.3', 'run_id': config['run_id'], 'attempt_id': attempt,
                  'process': lease.identity, 'gpu_uuid': gpu_uuid, 'physical_gpu_index': args.gpu_index,
                  'config_sha256': config_hash, 'adapter_sha256': adapter_hash,
                  'planned_studies': 1000, 'planned_generations': 2000,
                  'completed_studies': len(saved), 'completed_generations': len(saved) * 2,
                  'validity_gate': 'PASS' if len(saved) >= 10 else 'PENDING'}
        atomic_json(run / 'P4.3_status.json', status)
        try:
            checkpoint = ROOT / config['model']['checkpoint_dir']
            processor = Blip2Processor.from_pretrained(checkpoint, local_files_only=True, use_fast=False)
            tokenizer = processor.tokenizer
            if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (2, 2, 1):
                raise ValueError('Tokenizer IDs changed')
            model, loading = Blip2ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True,
                torch_dtype=torch.float16, output_loading_info=True)
            if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('error_msgs'):
                raise RuntimeError('Pretrained load identity failure')
            model.requires_grad_(False)
            model.language_projection = FP32Projection(model.language_projection)
            _pretrained_projection_check(model, checkpoint)
            model.to('cuda:0').eval()
            native = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
                      if name in config['model']['trainable_parameters']}
            adapter = torch.load(adapter_path, map_location='cpu')
            if adapter.get('stage') != 'P4.2':
                raise ValueError('M1 adapter is not P4.2 final checkpoint')
            load_adapter(model, adapter, config_hash, index['manifests']['mimic_train']['sha256'])
            trained = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
                       if name in config['model']['trainable_parameters']}
            if all(torch.equal(native[name], trained[name]) for name in native):
                raise RuntimeError('M1 projection equals native M0 after training')
            named = dict(model.named_parameters())
            def set_projection(values):
                for name, value in values.items():
                    named[name].data.copy_(value.to('cuda:0'))
            x00_source = Path(rows[5]['reference_source'])
            with x00_source.open(newline='', encoding='utf-8-sig') as stream:
                x00_cohort = list(csv.DictReader(stream))
            gen = config['generation']
            latencies = []
            with predictions_path.open('a', encoding='utf-8') as output:
                for i in range(len(saved), len(rows)):
                    if ledger.remaining_seconds() < 40:
                        status['stop_reason'] = 'insufficient remaining inference budget'
                        break
                    row = rows[i]
                    reference_for(row, x00_cohort)  # integrity only; never passed to the model
                    inputs = processor(images=load_image(row['image_path']), text=gen['prompt'], return_tensors='pt')
                    inputs = {key: value.to('cuda:0', dtype=torch.float16) if key == 'pixel_values'
                              else value.to('cuda:0') for key, value in inputs.items()}
                    outputs = {}
                    for arm, projection in [('M0', native), ('M1', trained)]:
                        set_projection(projection)
                        torch.cuda.synchronize()
                        tick = time.monotonic()
                        with torch.no_grad():
                            tokens = generate_single_sequence(model, inputs, gen)
                        torch.cuda.synchronize()
                        continuation = continuation_tokens(tokens, gen['bos_token_id'])
                        text = tokenizer.batch_decode(continuation, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)[0].strip()
                        outputs[arm] = {'model_id': arm, 'generated_report': text, 'valid_nonempty': bool(text),
                                        'generated_tokens': int(continuation.shape[1]), 'stopped_by_eos': bool(continuation[0, -1].item() == gen['eos_token_id']),
                                        'hit_max_length': bool(continuation[0, -1].item() != gen['eos_token_id'] and continuation.shape[1] >= gen['max_new_tokens']),
                                        'latency_seconds': time.monotonic() - tick}
                        latencies.append(outputs[arm]['latency_seconds'])
                    pair = paired_record(row, config['run_id'], config_hash, adapter_hash, outputs)
                    output.write(json.dumps(pair, ensure_ascii=False) + '\n')
                    output.flush()
                    saved.append(pair)
                    status['completed_studies'] = len(saved)
                    status['completed_generations'] = len(saved) * 2
                    if len(saved) == 10:
                        status['validity_gate'] = 'PASS' if all(o['valid_nonempty'] for pair in saved for o in pair['outputs'].values()) else 'FAIL'
                        status['validity_gate_mean_generation_seconds'] = sum(latencies) / len(latencies)
                        status['forecast_2000_generation_seconds_at_gate_mean'] = 2000 * status['validity_gate_mean_generation_seconds']
                        atomic_json(run / 'P4.3_status.json', status)
                        if status['validity_gate'] != 'PASS':
                            raise RuntimeError('Ten-study/two-arm output validity gate failed')
                    if len(saved) % 10 == 0:
                        atomic_json(run / 'P4.3_status.json', status)
            status.update({'status': 'PASS' if len(saved) == 1000 else 'PARTIAL',
                           'completed_studies': len(saved), 'completed_generations': len(saved) * 2,
                           'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                           'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                           'wall_seconds_this_attempt': time.monotonic() - started})
        except BaseException as exc:
            status.update({'status': 'FAIL' if status.get('validity_gate') == 'FAIL' else 'PARTIAL',
                           'reason': type(exc).__name__ + ': ' + str(exc),
                           'wall_seconds_this_attempt': time.monotonic() - started})
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            ledger.finish_attempt()
            status['cumulative_charged_seconds'] = sum(a.get('charged_seconds', 0) for a in ledger.data['attempts'])
            status['remaining_seconds'] = ledger.remaining_seconds()
            atomic_json(run / 'P4.3_status.json', status)


if __name__ == '__main__':
    main()
