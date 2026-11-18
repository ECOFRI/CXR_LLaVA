"""Reimplemented research bridge arms; never a reproduction of paper models."""
import argparse
import json
import sys
from pathlib import Path

VR = Path(__file__).resolve().parents[2]
ROOT = VR.parents[1]
sys.path.insert(0, str(VR / 'src'))

from vlm_medical.data.manifests import check_cohorts, eligible_records
from vlm_medical.models.bridges import FP32Projection, FP32QFormer, inject_opt_lora, make_bridge
from vlm_medical.models.registry import MissingCheckpointError, resolve_checkpoint, resolve_encoder
from vlm_medical.train.common import log_status, preflight_storage, preflight_vram, require_approval, resume_checkpoint, save_checkpoint, seed_all


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config['classification'] != 'REIMPLEMENTED':
        raise ValueError('This entrypoint accepts REIMPLEMENTED arms only')
    return config


def dry_run(config, out):
    import importlib.util
    import torch
    for dependency in ('transformers', 'PIL'):
        if importlib.util.find_spec(dependency) is None:
            raise ImportError('Missing dependency: ' + dependency)
    checkpoint = resolve_checkpoint(ROOT / config['checkpoint_dir'])
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    missing = [name for name in shards if not (checkpoint / name).is_file()]
    if missing:
        raise MissingCheckpointError('Local BLIP-2 shards missing: ' + ', '.join(missing))
    data = json.loads((ROOT / config['mimic_config']).read_text())
    cohorts = check_cohorts(data, ROOT / config['indiana_x00_csv'])
    bridge = make_bridge(config['synthetic_bridge'], config['feature_dim'], config['language_dim'], config['n_query'])
    with torch.no_grad():
        result = bridge(torch.zeros(1, 4, config['feature_dim']))
    if tuple(result.shape) != (1, config['n_query'], config['language_dim']):
        raise RuntimeError('Bridge shape mismatch')
    if config['arm'] == 'projection_opt_lora':
        from vlm_medical.models.bridges import LoRALinear
        adapter = LoRALinear(torch.nn.Linear(8, 8))
        with torch.no_grad():
            if tuple(adapter(torch.zeros(1, 8)).shape) != (1, 8):
                raise RuntimeError('LoRA shape mismatch')
    if config['encoder'] != 'blip2':
        resolve_encoder(config['encoder'], ROOT / config['encoder_dir'])
        cache = config.get('feature_cache')
        if not cache or not (ROOT / cache).is_dir():
            log_status(out / 'status.json', status='BLOCKED', experiment_id=config['experiment_id'],
                       reason='Frozen encoder feature cache and adapter not verified: ' + config['encoder'],
                       synthetic_bridge_shape=list(result.shape), cuda_used=False, model_weights_loaded=False)
            raise MissingCheckpointError('Frozen encoder feature cache and adapter not verified: ' + config['encoder'])
    status = {'status': 'DRY_RUN_PASS', 'experiment_id': config['experiment_id'],
              'classification': 'REIMPLEMENTED_SCRIPT_READY', 'checkpoint_files': len(shards),
              'cohort_sample_check': cohorts, 'synthetic_bridge_shape': list(result.shape),
              'cuda_used': False, 'model_weights_loaded': False, 'trainable_bridge_parameters': sum(p.numel() for p in bridge.parameters())}
    log_status(out / 'status.json', **status)
    return status


def train(config, out, seed, resume):
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    from vlm_medical.data.preprocess import prepare
    seed_all(seed)
    preflight_storage(out)
    preflight_vram(config['required_free_gib'])
    checkpoint = ROOT / config['checkpoint_dir']
    processor = Blip2Processor.from_pretrained(checkpoint, local_files_only=True, use_fast=False)
    model = Blip2ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True, torch_dtype=torch.float16)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.language_projection = FP32Projection(model.language_projection)
    if config['arm'] == 'qformer_projection':
        model.qformer = FP32QFormer(model.qformer)
        model.query_tokens = torch.nn.Parameter(model.query_tokens.float())
    elif config['arm'] == 'projection_opt_lora':
        inject_opt_lora(model.language_model)
    elif config['arm'] != 'projection_only':
        raise ValueError('Arm requires a verified feature adapter: ' + config['arm'])
    model.to('cuda:0')
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config['learning_rate'])
    start = resume_checkpoint(resume, model, optimizer) if resume else 0
    data = json.loads((ROOT / config['mimic_config']).read_text())
    step = start
    model.train()
    for record in eligible_records(data, 'train'):
        if step >= config['max_steps']:
            break
        batch = prepare(processor, record['image'], record['text'], 'cuda:0')
        batch['pixel_values'] = batch['pixel_values'].to(dtype=torch.float16)
        outputs = model(**batch, labels=batch['input_ids'])
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % config['save_every'] == 0:
            save_checkpoint(out / ('step_' + str(step) + '.pt'), model, optimizer, step, config)
    save_checkpoint(out / 'last.pt', model, optimizer, step, config)
    log_status(out / 'status.json', status='TRAIN_COMPLETE', step=step, classification='REIMPLEMENTED_NOT_REPRODUCED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--dry-run', action='store_true')
    choice.add_argument('--train', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if not args.output_dir.resolve().is_relative_to(VR.resolve()):
        raise ValueError('Output directory must be under research/blip2_cxr_v4')
    if args.train:
        require_approval(VR, config['experiment_id'])
        dry_run(config, args.output_dir)
        train(config, args.output_dir, args.seed, args.resume)
        print(json.dumps({'status': 'TRAIN_COMPLETE', 'experiment_id': config['experiment_id']}))
        return
    try:
        result = dry_run(config, args.output_dir)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as error:
        status_path = args.output_dir / 'status.json'
        existing = json.loads(status_path.read_text()) if status_path.is_file() else {}
        log_status(status_path, **{**existing, 'status': 'BLOCKED', 'experiment_id': config['experiment_id'],
                                   'reason': str(error), 'cuda_used': False, 'model_weights_loaded': False})
        raise SystemExit(2) from error
    print(json.dumps(result, separators=(',', ':')))


if __name__ == '__main__':
    main()
