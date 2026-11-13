"""One bounded P5 Q-Former training stage; invoked by detached supervisor."""
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

from p5_common import (ROOT, VR, assert_modes, configure_model, contract_and_index, digest,
                       expected_names, read_manifest, restore_checkpoint, save_checkpoint,
                       target_for, tokenize_training, train_order)
from vlm_medical.data.preprocess import load_image
from vlm_medical.train.p4_runtime import BudgetLedger, ExecutionLease, atomic_json


def _timeout(signum, frame):
    raise TimeoutError('P5 stage cumulative watchdog reached')


def _pretrained_check(model, checkpoint):
    import torch
    from safetensors import safe_open
    weight_map = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    named = dict(model.named_parameters())
    for source, target in [('query_tokens', 'query_tokens'),
                           ('language_projection.weight', 'language_projection.layer.weight'),
                           ('language_projection.bias', 'language_projection.layer.bias'),
                           ('qformer.encoder.layer.0.attention.attention.query.weight',
                            'qformer.inner.encoder.layer.0.attention.attention.query.weight')]:
        with safe_open(checkpoint / weight_map[source], framework='pt', device='cpu') as handle:
            original = handle.get_tensor(source).to(torch.float16).float()
        if not torch.equal(named[target].detach().cpu(), original):
            raise RuntimeError('Pinned pretrained initialization mismatch: ' + target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('P5.1', 'P5.2'), required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--manifest-index', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--gpu-index', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    config, index = contract_and_index(args.config, args.manifest_index)
    if not args.run_dir.resolve().is_relative_to((VR / 'runs').resolve()):
        raise ValueError('P5 output outside run root')
    item = index['manifests']['mimic_train']
    rows = read_manifest(ROOT / item['path'], item['sha256'], 2000, 'train')
    opt = config['optimization']
    if (opt['microbatch_size'], opt['gradient_accumulation_steps'], opt['scheduler'], opt['seed']) != (1, 1, 'constant_no_scheduler', 42):
        raise ValueError('Unapproved optimization setting')
    order = train_order(len(rows), opt['seed'])
    count = config['budgets']['p5_1_optimizer_updates'] if args.stage == 'P5.1' else config['budgets']['p5_2_optimizer_updates']
    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN_PASS', 'stage': args.stage, 'planned_updates': count,
                          'config_sha256': digest(args.config), 'manifest_sha256': item['sha256']}))
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu_index):
        raise ValueError('CUDA_VISIBLE_DEVICES must expose only selected GPU')
    reading = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu_index), '--query-gpu=uuid,memory.used',
                                       '--format=csv,noheader,nounits'], text=True).strip()
    gpu_uuid, used = [part.strip() for part in reading.split(',')]
    if int(used) != 0:
        raise RuntimeError('GPU occupied before launch')
    import numpy as np
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'K80' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one visible Tesla K80 required')
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    limit = config['budgets']['p5_1_wall_seconds_cumulative' if args.stage == 'P5.1' else 'p5_2_wall_seconds_cumulative']
    stage_path = run / (args.stage + '_status.json')
    checkpoint_path = run / (args.stage + '_last.pt')
    with ExecutionLease(VR / 'runs/.p4_gpu_job.lock') as lease:
        ledger = BudgetLedger(run / (args.stage + '_budget.json'), limit, lease)
        remaining = ledger.check_remaining(required_seconds=60)
        attempt = ledger.start_attempt(config['run_id'] + '_' + args.stage)
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, max(1, remaining - 30))
        start = time.monotonic()
        status = {'status': 'RUNNING', 'stage': args.stage, 'run_id': config['run_id'],
                  'attempt_id': attempt, 'pid': os.getpid(), 'gpu_uuid': gpu_uuid,
                  'config_sha256': digest(args.config), 'manifest_sha256': item['sha256'],
                  'planned_updates': count, 'optimizer_updates': 0}
        atomic_json(stage_path, status)
        model = optimizer = None
        pos = 0
        elapsed_before = 0.0
        try:
            random.seed(opt['seed']); np.random.seed(opt['seed']); torch.manual_seed(opt['seed']); torch.cuda.manual_seed_all(opt['seed'])
            source = ROOT / config['model']['checkpoint_dir']
            processor = Blip2Processor.from_pretrained(source, local_files_only=True, use_fast=False)
            tokenizer = processor.tokenizer
            if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (2, 2, 1):
                raise ValueError('Tokenizer identity mismatch')
            model, info = Blip2ForConditionalGeneration.from_pretrained(source, local_files_only=True,
                torch_dtype=torch.float16, output_loading_info=True)
            if info.get('missing_keys') or info.get('unexpected_keys') or info.get('error_msgs'):
                raise RuntimeError('BLIP-2 checkpoint load mismatch')
            names = configure_model(model)
            if sum(p.numel() for p in model.parameters() if p.requires_grad) != config['model']['trainable_parameter_count']:
                raise RuntimeError('Trainable parameter count mismatch')
            if not checkpoint_path.exists():
                _pretrained_check(model, source)
                status['initialized_from'] = 'original_pretrained_blip2'
            model.to('cuda:0')
            assert_modes(model)
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                lr=opt['learning_rate'], betas=tuple(opt['betas']), eps=opt['eps'],
                weight_decay=opt['weight_decay'], amsgrad=opt['amsgrad'])
            if checkpoint_path.exists():
                pos, elapsed_before = restore_checkpoint(checkpoint_path, model, optimizer, args.stage,
                    digest(args.config), item['sha256'], order)
                status['resumed_from_position'] = pos
            elif args.stage == 'P5.2':
                status['p5_1_checkpoint_used'] = False
            start_pos = pos
            representatives = ('qformer.inner.encoder.layer.0.attention.attention.query.weight',
                               'query_tokens', 'language_projection.layer.weight')
            named = dict(model.named_parameters())
            initial_representatives = {name: named[name].detach().cpu().clone() for name in representatives}
            updates_path = run / (args.stage + '_updates.jsonl')
            if updates_path.exists():
                saved = [json.loads(x) for x in updates_path.read_text().splitlines()]
                if len(saved) < pos or [x['position'] for x in saved] != list(range(1, len(saved) + 1)):
                    raise ValueError('Update log/checkpoint cursor mismatch')
                if len(saved) > pos:
                    orphaned = run / (args.stage + '_uncommitted_attempt_' + str(time.time_ns()) + '.jsonl')
                    orphaned.write_text(''.join(json.dumps(x) + '\n' for x in saved[pos:]))
                    updates_path.write_text(''.join(json.dumps(x) + '\n' for x in saved[:pos]))
            elif pos:
                raise ValueError('Missing update log on resume')
            status['optimizer_updates'] = pos
            atomic_json(stage_path, status)
            frozen_versions = {n: p._version for n, p in model.named_parameters() if n not in names}
            with updates_path.open('a', encoding='utf-8') as log:
                while pos < count:
                    if ledger.remaining_seconds() < 90:
                        raise TimeoutError('Insufficient remaining budget for update/checkpoint')
                    row = rows[order[pos]]
                    target = target_for(row)
                    ids, labels, before, used_tokens = tokenize_training(tokenizer, config, target)
                    pixels = processor(images=load_image(row['image_path']), return_tensors='pt')['pixel_values'].to('cuda:0', dtype=torch.float16)
                    ids, labels = ids.to('cuda:0'), labels.to('cuda:0')
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(); tick = time.monotonic()
                    assert_modes(model)
                    loss = model(pixel_values=pixels, input_ids=ids, labels=labels).loss
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('Nonfinite loss')
                    loss.backward()
                    groups = {'qformer': [], 'query': [], 'projection': []}
                    for name, parameter in model.named_parameters():
                        if not parameter.requires_grad:
                            if parameter.grad is not None:
                                raise RuntimeError('Frozen gradient: ' + name)
                            continue
                        if parameter.grad is None:
                            continue  # Q-Former text-only embedding weights are not used by query-only forward.
                        if not bool(torch.isfinite(parameter.grad).all()):
                            raise RuntimeError('Nonfinite gradient: ' + name)
                        group = 'qformer' if name.startswith('qformer.') else 'query' if name == 'query_tokens' else 'projection'
                        groups[group].append(parameter.grad.float().square().sum())
                    norms = {k: math.sqrt(sum(float(x.item()) for x in values)) for k, values in groups.items()}
                    if any(value <= 0 or not math.isfinite(value) for value in norms.values()):
                        raise RuntimeError('Missing/zero component gradient: ' + str(norms))
                    optimizer.step()
                    torch.cuda.synchronize()
                    pos += 1
                    peak = torch.cuda.max_memory_reserved() / 2**30
                    if peak > config['budgets']['peak_reserved_gib_limit']:
                        raise MemoryError('Peak reserved GPU memory exceeded 10.0 GiB')
                    event = {'position': pos, 'loss': float(loss.item()), 'gradient_norms': norms,
                             'step_seconds': time.monotonic() - tick, 'target_tokens_before_truncation': before,
                             'target_tokens_used': used_tokens, 'text_tokens': int(ids.shape[1]), 'peak_reserved_gib': peak}
                    log.write(json.dumps(event) + '\n'); log.flush(); os.fsync(log.fileno())
                    if args.stage == 'P5.1' and pos == count or args.stage == 'P5.2' and (pos % opt['checkpoint_every_optimizer_updates'] == 0 or pos == count):
                        save_checkpoint(checkpoint_path, model, optimizer, args.stage, digest(args.config),
                                        item['sha256'], order, pos, elapsed_before + time.monotonic() - start)
                        status.update({'optimizer_updates': pos, 'checkpoint_sha256': digest(checkpoint_path),
                                       'peak_reserved_gib': peak, 'heartbeat_utc_ns': time.time_ns()})
                        atomic_json(stage_path, status)
            for name, parameter in model.named_parameters():
                if name in frozen_versions and (parameter._version != frozen_versions[name] or parameter.grad is not None):
                    raise RuntimeError('Frozen parameter changed: ' + name)
            changed = {name: not torch.equal(named[name].detach().cpu(), initial_representatives[name])
                       for name in representatives}
            if pos > start_pos and not all(changed.values()):
                raise RuntimeError('Trainable component representative did not update: ' + str(changed))
            if pos == start_pos == count:
                changed = {name: None for name in representatives}  # Previous attempt's updates are in the verified log.
            restored, _ = restore_checkpoint(checkpoint_path, model, optimizer, args.stage,
                digest(args.config), item['sha256'], order, restore_rng=False)
            if restored != count:
                raise RuntimeError('Final checkpoint reload cursor mismatch')
            status.update({'status': 'PASS', 'optimizer_updates': pos, 'completed_studies': pos,
                           'epochs_complete': 1 if args.stage == 'P5.2' else 0,
                           'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                           'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                           'checkpoint_sha256': digest(checkpoint_path), 'checkpoint_reload_verified': True,
                           'frozen_versions_unchanged': True, 'trainable_components_updated': changed,
                           'wall_seconds_this_attempt': time.monotonic() - start})
        except BaseException as exc:
            status.update({'status': 'PARTIAL' if pos else 'FAILED', 'optimizer_updates': pos,
                           'reason': type(exc).__name__ + ': ' + str(exc),
                           'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                           'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                           'wall_seconds_this_attempt': time.monotonic() - start})
            if model is not None and optimizer is not None and pos > 0:
                save_checkpoint(checkpoint_path, model, optimizer, args.stage, digest(args.config),
                                item['sha256'], order, pos, elapsed_before + time.monotonic() - start)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            ledger.finish_attempt()
            status['cumulative_charged_seconds'] = sum(x.get('charged_seconds', 0) for x in ledger.data['attempts'])
            atomic_json(stage_path, status)


if __name__ == '__main__':
    main()
