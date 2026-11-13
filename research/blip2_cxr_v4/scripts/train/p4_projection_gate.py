"""Approved P4 projection-only memory gate on a single K80."""
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

from vlm_medical.data.manifests import report_sections
from vlm_medical.models.bridges import FP32Projection
from vlm_medical.train.p4_runtime import BudgetLedger, ExecutionLease, atomic_json
from vlm_medical.train.p4_checkpoint import load_adapter


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_contract(path):
    data = json.loads(Path(path).read_text())
    if data['approval_status'] != 'APPROVED_P4_20261023' or data['model']['revision'] != '59a1ef6c1e5117b3f65523d1c6066825bcf315e3':
        raise ValueError('P4 scientific contract is not approved or pinned')
    if data['model']['trainable_parameters'] != ['language_projection.layer.weight', 'language_projection.layer.bias']:
        raise ValueError('Projection parameter allowlist changed')
    return data


def read_manifest(path, expected_hash, count, split):
    if digest(path) != expected_hash:
        raise ValueError('Frozen manifest hash mismatch')
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    keys = [(r['dataset'], r['subject_id'], r['study_id']) for r in rows]
    if len(rows) != count or len(set(keys)) != count or any(r['split'] != split for r in rows):
        raise ValueError('Frozen cohort count, split, or uniqueness mismatch')
    return rows


def target_for(row):
    report = Path(row['reference_path'])
    if digest(report) != row['reference_file_sha256']:
        raise ValueError('Reference changed for study ' + row['study_id'])
    sections = report_sections(report.read_text(errors='replace'))
    target = '\n'.join(sections[k] for k in ('findings', 'impression') if k in sections)
    if not target or hashlib.sha256(target.encode()).hexdigest() != row['target_text_sha256']:
        raise ValueError('Target linkage mismatch for study ' + row['study_id'])
    return target


def tokenize_training(tokenizer, contract, target):
    import torch
    t = contract['training_text']
    prompt = tokenizer.encode(t['prompt'] + t['separator_before_target'], add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    max_text = t['max_text_sequence_tokens']
    max_target = t['max_target_tokens_including_eos']
    if 1 + len(prompt) + 1 > max_text:
        raise ValueError('Prompt exceeds approved text context')
    allowed = min(max_target - 1, max_text - 2 - len(prompt))
    used = target_ids[:allowed]
    input_ids = [tkn for tkn in [contract['tokenizer']['bos_token_id']] + prompt + used + [contract['tokenizer']['eos_token_id']]]
    labels = [-100] * (1 + len(prompt)) + used + [contract['tokenizer']['eos_token_id']]
    if len(input_ids) != len(labels) or not used or len(input_ids) > max_text:
        raise ValueError('Invalid tokenized training target')
    return (torch.tensor([input_ids], dtype=torch.long), torch.tensor([labels], dtype=torch.long),
            len(target_ids), len(used))


def _timeout(signum, frame):
    raise TimeoutError('P4.1 cumulative wall-time watchdog reached')


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
        raise ValueError('Manifest index is not approved')
    item = index['manifests']['mimic_train']
    rows = read_manifest(ROOT / item['path'], item['sha256'], 2000, 'train')
    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN_PASS', 'train_studies': len(rows), 'config_sha256': digest(args.config), 'manifest_sha256': item['sha256']}))
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu_index):
        raise ValueError('CUDA_VISIBLE_DEVICES must equal the physical GPU index')
    reading = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu_index), '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    gpu_uuid, used_mb = [part.strip() for part in reading.split(',')]
    if int(used_mb) > 0:
        raise RuntimeError('Selected physical GPU is occupied before launch: ' + str(used_mb) + ' MiB')
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    from vlm_medical.data.preprocess import load_image
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA device required')
    gpu_name = torch.cuda.get_device_name(0)
    if 'K80' not in gpu_name:
        raise RuntimeError('P4.1 requires a Tesla K80, got ' + gpu_name)
    device = 'cuda:0'
    run = args.output_dir
    run.mkdir(parents=True, exist_ok=True)
    with ExecutionLease(VR / 'runs/.p4_gpu_job.lock') as lease:
        ledger = BudgetLedger(run / 'P4.1_budget.json', config['budgets']['p4_1_wall_seconds_cumulative'], lease)
        remaining = ledger.check_remaining()
        attempt = ledger.start_attempt(config['run_id'] + '_P4.1')
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, max(1, remaining - 3))
        started = time.monotonic()
        status = {'status': 'RUNNING', 'stage': 'P4.1', 'run_id': config['run_id'],
                  'attempt_id': attempt, 'process': lease.identity, 'physical_gpu_index': args.gpu_index,
                  'gpu_name': gpu_name, 'gpu_uuid': gpu_uuid, 'initial_device_used_mib': int(used_mb), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                  'config_sha256': digest(args.config), 'manifest_sha256': item['sha256']}
        atomic_json(run / 'P4.1_status.json', status)
        try:
            occupancy = torch.cuda.memory_reserved(0)
            if occupancy:
                raise RuntimeError('GPU has existing PyTorch reservation')
            torch.manual_seed(config['optimization']['seed'])
            random.seed(config['optimization']['seed'])
            processor = Blip2Processor.from_pretrained(ROOT / config['model']['checkpoint_dir'], local_files_only=True, use_fast=False)
            tokenizer = processor.tokenizer
            if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (2, 2, 1):
                raise ValueError('Tokenizer special IDs differ from frozen contract')
            checkpoint = ROOT / config['model']['checkpoint_dir']
            model, loading = Blip2ForConditionalGeneration.from_pretrained(
                checkpoint, local_files_only=True, torch_dtype=torch.float16, output_loading_info=True)
            if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('error_msgs'):
                raise RuntimeError('Pretrained load has missing/unexpected/error keys: ' + str(loading))
            model.requires_grad_(False)
            model.language_projection = FP32Projection(model.language_projection)
            model.to(device).eval()
            model.language_projection.train()
            names = {name for name, param in model.named_parameters() if param.requires_grad}
            if names != set(config['model']['trainable_parameters']):
                raise RuntimeError('Trainable parameter allowlist mismatch: ' + str(names))
            if sum(p.numel() for p in model.parameters() if p.requires_grad) != config['model']['trainable_parameter_count']:
                raise RuntimeError('Trainable parameter count mismatch')
            opt = config['optimization']
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=opt['learning_rate'],
                                          betas=tuple(opt['betas']), eps=opt['eps'], weight_decay=opt['weight_decay'],
                                          amsgrad=opt['amsgrad'])
            initial_projection = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name in names}
            frozen_versions = {name: p._version for name, p in model.named_parameters() if name not in names}
            if any(p.grad is not None for name, p in model.named_parameters() if name not in names):
                raise RuntimeError('Frozen parameter has a gradient before training')
            torch.cuda.reset_peak_memory_stats()
            order = list(range(len(rows)))
            random.Random(opt['seed']).shuffle(order)
            updates = []
            for step in range(config['budgets']['p4_1_optimizer_updates']):
                row = rows[order[step]]
                target = target_for(row)
                input_ids, labels, original_length, used_length = tokenize_training(tokenizer, config, target)
                image = load_image(row['image_path'])
                pixels = processor(images=image, return_tensors='pt')['pixel_values'].to(device, dtype=torch.float16)
                input_ids, labels = input_ids.to(device), labels.to(device)
                torch.cuda.synchronize()
                tick = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                loss = model(pixel_values=pixels, input_ids=input_ids, labels=labels).loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite loss at step ' + str(step))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad]
                if any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
                    raise RuntimeError('Missing or nonfinite projection gradient')
                norm = math.sqrt(sum(float(g.float().square().sum().item()) for g in grads))
                if norm <= 0:
                    raise RuntimeError('Zero projection gradient')
                optimizer.step()
                torch.cuda.synchronize()
                updates.append({'step': step + 1, 'loss': float(loss.item()), 'grad_norm': norm,
                                'target_tokens_before_truncation': original_length, 'target_tokens_used': used_length,
                                'text_tokens': int(input_ids.shape[1]), 'update_seconds': time.monotonic() - tick})
                atomic_json(run / 'P4.1_updates.json', {'updates': updates})
            changed = {name: not torch.equal(p.detach().cpu(), initial_projection[name]) for name, p in model.named_parameters() if name in names}
            if not all(changed.values()):
                raise RuntimeError('Projection parameter did not update')
            for name, p in model.named_parameters():
                if name in frozen_versions and (p._version != frozen_versions[name] or p.grad is not None):
                    raise RuntimeError('Frozen parameter version or gradient changed: ' + name)
            adapter = {name: p.detach().cpu() for name, p in model.named_parameters() if name in names}
            path = run / 'P4.1_adapter_smoke.pt'
            torch.save({'stage': 'P4.1', 'adapter': adapter, 'config_sha256': digest(args.config), 'manifest_sha256': item['sha256']}, path)
            loaded = torch.load(path, map_location='cpu')
            load_adapter(model, loaded, digest(args.config), item['sha256'], allow_smoke=True)
            if set(loaded['adapter']) != names or any(not torch.equal(loaded['adapter'][key], adapter[key]) for key in names):
                raise RuntimeError('Adapter save/reload mismatch')
            status.update({'status': 'PASS', 'optimizer_updates': len(updates), 'adapter_sha256': digest(path),
                           'all_projection_parameters_changed': changed, 'all_frozen_parameter_versions_unchanged': True,
                           'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                           'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                           'wall_seconds': time.monotonic() - started, 'update_seconds_total': sum(u['update_seconds'] for u in updates)})
            if status['peak_reserved_gib'] > config['budgets']['p4_1_peak_gib_limit']:
                raise RuntimeError('GPU peak reserved exceeded 10 GiB')
        except BaseException as exc:
            status.update({'status': 'FAIL', 'reason': type(exc).__name__ + ': ' + str(exc),
                           'wall_seconds': time.monotonic() - started,
                           'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                           'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30})
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            ledger.finish_attempt()
            status['cumulative_charged_seconds'] = sum(a.get('charged_seconds', 0) for a in ledger.data['attempts'])
            status['remaining_seconds'] = ledger.remaining_seconds()
            atomic_json(run / 'P4.1_status.json', status)


if __name__ == '__main__':
    main()
