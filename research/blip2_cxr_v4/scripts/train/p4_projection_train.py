"""One approved P4.2 projection epoch over the frozen 2,000-study cohort."""
import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

VR = Path(__file__).resolve().parents[2]
ROOT = VR.parents[1]
sys.path.insert(0, str(VR / 'src'))
from p4_projection_gate import digest, read_contract, read_manifest, target_for, tokenize_training
from vlm_medical.models.bridges import FP32Projection
from vlm_medical.train.p4_checkpoint import save_state, resume_state
from vlm_medical.train.p4_runtime import BudgetLedger, ExecutionLease, atomic_json


def _timeout(signum, frame):
    raise TimeoutError('P4.2 cumulative wall-time watchdog reached')


def _pretrained_projection_check(model, checkpoint):
    """Compare fresh wrapped tensors with pinned source after native FP16 load."""
    from safetensors import safe_open
    import torch
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    parameters = dict(model.named_parameters())
    hashes = {}
    for short in ('weight', 'bias'):
        source_name = 'language_projection.' + short
        wrapped_name = 'language_projection.layer.' + short
        with safe_open(checkpoint / index[source_name], framework='pt', device='cpu') as stream:
            source = stream.get_tensor(source_name).to(torch.float16).to(torch.float32)
        current = parameters[wrapped_name].detach().cpu()
        if not torch.equal(current, source):
            raise RuntimeError('Projection differs from pinned pretrained initialization: ' + short)
        hashes[wrapped_name] = hashlib.sha256(current.numpy().tobytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest-index', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-index', type=int, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = read_contract(args.config)
    if not args.output_dir.resolve().is_relative_to((VR / 'runs').resolve()):
        raise ValueError('Output must be under V4 runs')
    index = json.loads(args.manifest_index.read_text())
    if not index['approved'] or index['status'] != 'APPROVED_P4_20261023':
        raise ValueError('Manifest index is not approved')
    item = index['manifests']['mimic_train']
    rows = read_manifest(ROOT / item['path'], item['sha256'], 2000, 'train')
    order = list(range(len(rows)))
    random.Random(config['optimization']['seed']).shuffle(order)
    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN_PASS', 'train_studies': len(rows), 'unique_studies': len({(r['subject_id'], r['study_id']) for r in rows}), 'epochs': 1, 'config_sha256': digest(args.config), 'manifest_sha256': item['sha256']}))
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu_index):
        raise ValueError('CUDA_VISIBLE_DEVICES must equal physical GPU index')
    reading = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu_index), '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    gpu_uuid, used_mb = [part.strip() for part in reading.split(',')]
    if int(used_mb) > 0:
        raise RuntimeError('Selected physical GPU occupied before launch')
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    from vlm_medical.data.preprocess import load_image
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'K80' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one Tesla K80 required')
    run = args.output_dir
    run.mkdir(parents=True, exist_ok=True)
    with ExecutionLease(VR / 'runs/.p4_gpu_job.lock') as lease:
        ledger = BudgetLedger(run / 'P4.2_budget.json', config['budgets']['p4_2_wall_seconds_cumulative'], lease)
        remaining = ledger.check_remaining(required_seconds=60)
        attempt = ledger.start_attempt(config['run_id'] + '_P4.2')
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, max(1, remaining - 60))
        started = time.monotonic()
        config_hash = digest(args.config)
        status = {'status': 'RUNNING', 'stage': 'P4.2', 'run_id': config['run_id'], 'attempt_id': attempt,
                  'process': lease.identity, 'physical_gpu_index': args.gpu_index, 'gpu_uuid': gpu_uuid,
                  'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'config_sha256': config_hash,
                  'manifest_sha256': item['sha256'], 'planned_studies': len(rows), 'completed_studies': 0,
                  'optimizer_updates': 0, 'epochs_complete': 0}
        atomic_json(run / 'P4.2_status.json', status)
        model = optimizer = None
        position = 0
        try:
            torch.manual_seed(config['optimization']['seed'])
            random.seed(config['optimization']['seed'])
            checkpoint = ROOT / config['model']['checkpoint_dir']
            processor = Blip2Processor.from_pretrained(checkpoint, local_files_only=True, use_fast=False)
            tokenizer = processor.tokenizer
            if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (2, 2, 1):
                raise ValueError('Tokenizer special IDs changed')
            model, loading = Blip2ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True,
                torch_dtype=torch.float16, output_loading_info=True)
            if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('error_msgs'):
                raise RuntimeError('Pretrained load contains missing/unexpected/error keys')
            model.requires_grad_(False)
            model.language_projection = FP32Projection(model.language_projection)
            pretrained_hashes = _pretrained_projection_check(model, checkpoint)
            model.to('cuda:0').eval()
            model.language_projection.train()
            names = {name for name, p in model.named_parameters() if p.requires_grad}
            if names != set(config['model']['trainable_parameters']):
                raise RuntimeError('Trainable parameter allowlist mismatch')
            frozen_versions = {name: p._version for name, p in model.named_parameters() if name not in names}
            opt = config['optimization']
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=opt['learning_rate'],
                betas=tuple(opt['betas']), eps=opt['eps'], weight_decay=opt['weight_decay'], amsgrad=opt['amsgrad'])
            state_path = run / 'P4.2_last.pt'
            prior_elapsed = 0.0
            if state_path.exists():
                position, prior_elapsed = resume_state(state_path, model, optimizer, config_hash, item['sha256'], order)
                status['resumed_from_position'] = position
            else:
                status['pretrained_projection_sha256'] = pretrained_hashes
                status['resumed_from_position'] = 0
            status['completed_studies'] = position
            status['optimizer_updates'] = position
            atomic_json(run / 'P4.2_status.json', status)
            torch.cuda.reset_peak_memory_stats()
            updates_file = run / 'P4.2_updates.jsonl'
            if updates_file.exists():
                saved_lines = updates_file.read_text().splitlines()
                kept = [line for line in saved_lines if json.loads(line)['position'] <= position]
                if len(kept) != position or [json.loads(line)['position'] for line in kept] != list(range(1, position + 1)):
                    raise ValueError('Update log/checkpoint position mismatch')
                if len(kept) != len(saved_lines):
                    updates_file.write_text('\n'.join(kept) + ('\n' if kept else ''))
            elif position:
                raise ValueError('Resume checkpoint has no matching update log')
            with updates_file.open('a', encoding='utf-8') as updates_stream:
                while position < len(rows):
                    if ledger.remaining_seconds() < 90:
                        raise TimeoutError('Insufficient budget for next update and checkpoint')
                    row = rows[order[position]]
                    target = target_for(row)
                    input_ids, labels, target_before, target_used = tokenize_training(tokenizer, config, target)
                    pixels = processor(images=load_image(row['image_path']), return_tensors='pt')['pixel_values'].to('cuda:0', dtype=torch.float16)
                    input_ids, labels = input_ids.to('cuda:0'), labels.to('cuda:0')
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(pixel_values=pixels, input_ids=input_ids, labels=labels).loss
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('Nonfinite training loss at position ' + str(position))
                    loss.backward()
                    grads = [p.grad for p in model.parameters() if p.requires_grad]
                    if any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
                        raise RuntimeError('Invalid projection gradients')
                    norm = math.sqrt(sum(float(g.float().square().sum().item()) for g in grads))
                    if norm <= 0:
                        raise RuntimeError('Zero projection gradient')
                    optimizer.step()
                    torch.cuda.synchronize()
                    position += 1
                    update = {'position': position, 'loss': float(loss.item()), 'grad_norm': norm,
                              'text_tokens': int(input_ids.shape[1]), 'target_tokens_before_truncation': target_before,
                              'target_tokens_used': target_used, 'update_seconds': time.monotonic() - tick}
                    updates_stream.write(json.dumps(update) + '\n')
                    updates_stream.flush()
                    if position % opt['checkpoint_every_optimizer_updates'] == 0 or position == len(rows):
                        save_state(state_path, model, optimizer, config_hash, item['sha256'], position, position,
                                   order, prior_elapsed + time.monotonic() - started)
                        status.update({'completed_studies': position, 'optimizer_updates': position,
                                       'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                                       'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                                       'checkpoint_sha256': digest(state_path)})
                        atomic_json(run / 'P4.2_status.json', status)
            for name, p in model.named_parameters():
                if name in frozen_versions and (p._version != frozen_versions[name] or p.grad is not None):
                    raise RuntimeError('Frozen parameter version or gradient changed: ' + name)
            check_model, check_loading = Blip2ForConditionalGeneration.from_pretrained(checkpoint,
                local_files_only=True, torch_dtype=torch.float16, output_loading_info=True)
            if check_loading.get('missing_keys') or check_loading.get('unexpected_keys') or check_loading.get('error_msgs'):
                raise RuntimeError('Reload base model identity failed')
            check_model.requires_grad_(False)
            check_model.language_projection = FP32Projection(check_model.language_projection)
            check_optimizer = torch.optim.AdamW((p for p in check_model.parameters() if p.requires_grad), lr=opt['learning_rate'],
                betas=tuple(opt['betas']), eps=opt['eps'], weight_decay=opt['weight_decay'], amsgrad=opt['amsgrad'])
            restored_position, _ = resume_state(state_path, check_model, check_optimizer, config_hash, item['sha256'], order)
            if restored_position != 2000:
                raise RuntimeError('Final checkpoint reload position mismatch')
            status.update({'status': 'PASS', 'completed_studies': 2000, 'optimizer_updates': 2000,
                           'epochs_complete': 1, 'final_checkpoint_sha256': digest(state_path),
                           'final_reload_verified': True, 'all_frozen_parameter_versions_unchanged': True,
                           'wall_seconds_this_attempt': time.monotonic() - started})
        except BaseException as exc:
            status.update({'status': 'PARTIAL' if position > 0 else 'FAIL',
                           'reason': type(exc).__name__ + ': ' + str(exc),
                           'completed_studies': position, 'optimizer_updates': position,
                           'wall_seconds_this_attempt': time.monotonic() - started})
            if model is not None and optimizer is not None and position > 0:
                save_state(run / 'P4.2_last.pt', model, optimizer, config_hash, item['sha256'], position, position,
                           order, time.monotonic() - started)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            ledger.finish_attempt()
            status['cumulative_charged_seconds'] = sum(a.get('charged_seconds', 0) for a in ledger.data['attempts'])
            status['remaining_seconds'] = ledger.remaining_seconds()
            atomic_json(run / 'P4.2_status.json', status)


if __name__ == '__main__':
    main()
