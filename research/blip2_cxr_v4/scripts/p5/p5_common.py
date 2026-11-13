"""P5 identity, checkpoint, and module controls; no GPU work on import."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

VR = Path(__file__).resolve().parents[2]
ROOT = VR.parents[1]
sys.path.insert(0, str(VR / 'src'))
sys.path.insert(0, str(VR / 'scripts/train'))

from p4_projection_gate import digest, read_manifest, target_for, tokenize_training
from vlm_medical.models.bridges import FP32Projection, FP32QFormer


def contract_and_index(config_path, index_path):
    config = json.loads(Path(config_path).read_text())
    index = json.loads(Path(index_path).read_text())
    if config['approval_status'] != 'APPROVED_P5_MASTER_20261025':
        raise ValueError('P5 authorization record absent')
    if config['model']['revision'] != '59a1ef6c1e5117b3f65523d1c6066825bcf315e3':
        raise ValueError('BLIP-2 revision changed')
    p4_path = ROOT / 'research/blip2_cxr_v4/configs/experiments/P4_projection_contract.approved.json'
    if digest(p4_path) != config['source_p4_contract_sha256']:
        raise ValueError('P4 protocol source changed')
    p4 = json.loads(p4_path.read_text())
    for key in ('tokenizer', 'image_preprocessing', 'training_text', 'optimization', 'generation'):
        if config[key] != p4[key]:
            raise ValueError('P5/P4 matched scientific protocol mismatch: ' + key)
    if (config['model']['checkpoint_dir'] != p4['model']['checkpoint_dir']
            or config['model']['trainable_parameters'] != 'qformer.inner.*;query_tokens;language_projection.layer.weight;language_projection.layer.bias'):
        raise ValueError('P5 model identity/allowlist changed')
    if index['status'] != 'APPROVED_P4_20261023' or not index['approved']:
        raise ValueError('Frozen cohort approval missing')
    if config['cohorts']['manifest_index'] != str(Path(index_path).resolve().relative_to(ROOT)):
        raise ValueError('Wrong P4 manifest index')
    b = config['budgets']
    if (b['physical_gpus'], b['p5_1_optimizer_updates'], b['p5_2_optimizer_updates'],
            b['p5_3_max_new_generations'], b['peak_reserved_gib_limit']) != (1, 20, 2000, 1000, 10.0):
        raise ValueError('P5 resource contract changed')
    return config, index


def trainable_names(model):
    return {name for name, p in model.named_parameters() if p.requires_grad}


def expected_names(model):
    return {name for name, _ in model.named_parameters()
            if name.startswith('qformer.inner.') or name == 'query_tokens'
            or name in ('language_projection.layer.weight', 'language_projection.layer.bias')}


def configure_model(model):
    import torch
    model.requires_grad_(False)
    model.qformer = FP32QFormer(model.qformer)
    model.query_tokens = torch.nn.Parameter(model.query_tokens.float())
    model.language_projection = FP32Projection(model.language_projection)
    model.eval()
    model.qformer.train()
    model.language_projection.train()
    names = trainable_names(model)
    if names != expected_names(model) or len(names) < 100:
        raise RuntimeError('Exact P5 parameter allowlist mismatch')
    if any(p.requires_grad for p in model.vision_model.parameters()) or any(p.requires_grad for p in model.language_model.parameters()):
        raise RuntimeError('Frozen backbone requires gradients')
    return names


def assert_modes(model):
    if model.vision_model.training or model.language_model.training or not model.qformer.training or not model.language_projection.training:
        raise RuntimeError('Frozen/trainable module modes changed')


def _order_hash(order):
    return hashlib.sha256(','.join(map(str, order)).encode()).hexdigest()


def save_checkpoint(path, model, optimizer, stage, config_hash, manifest_hash, order, position, elapsed):
    import numpy as np
    import torch
    if not 0 <= position <= len(order):
        raise ValueError('Invalid cursor')
    names = expected_names(model)
    named = dict(model.named_parameters())
    payload = {'schema_version': 1, 'stage': stage, 'config_sha256': config_hash,
               'manifest_sha256': manifest_hash, 'order_sha256': _order_hash(order),
               'next_position': position, 'optimizer_updates': position, 'epoch': int(position == len(order)),
               'sampler_state': {'permutation_sha256': _order_hash(order), 'cursor': position},
               'adapter': {name: named[name].detach().cpu().clone() for name in names},
               'optimizer': optimizer.state_dict(), 'scheduler': {'kind': 'constant_no_scheduler', 'step': position},
               'scaler': None, 'gradient_accumulation': {'steps': 1, 'pending_microbatches': 0},
               'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
               'torch_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
               'elapsed_seconds': float(elapsed)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def restore_checkpoint(path, model, optimizer, stage, config_hash, manifest_hash, order, *, restore_rng=True):
    import numpy as np
    import torch
    payload = torch.load(path, map_location='cpu')
    if (payload['stage'] != stage or payload['config_sha256'] != config_hash
            or payload['manifest_sha256'] != manifest_hash or payload['order_sha256'] != _order_hash(order)
            or payload['schema_version'] != 1 or payload['sampler_state']['permutation_sha256'] != _order_hash(order)):
        raise ValueError('P5 checkpoint identity mismatch')
    pos = payload['next_position']
    if (not isinstance(pos, int) or not 0 <= pos <= len(order) or payload['optimizer_updates'] != pos
            or payload['sampler_state']['cursor'] != pos or payload['scheduler'] != {'kind': 'constant_no_scheduler', 'step': pos}
            or payload['gradient_accumulation'] != {'steps': 1, 'pending_microbatches': 0} or payload['scaler'] is not None):
        raise ValueError('P5 checkpoint cursor/optimization mismatch')
    named = dict(model.named_parameters())
    if set(payload['adapter']) != expected_names(model):
        raise ValueError('P5 adapter names mismatch')
    for name, value in payload['adapter'].items():
        if tuple(named[name].shape) != tuple(value.shape):
            raise ValueError('P5 adapter tensor shape mismatch: ' + name)
        named[name].data.copy_(value.to(named[name].device, dtype=named[name].dtype))
    optimizer.load_state_dict(payload['optimizer'])
    if restore_rng:
        random.setstate(payload['python_rng'])
        np.random.set_state(payload['numpy_rng'])
        torch.set_rng_state(payload['torch_rng'])
        if payload['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return pos, payload['elapsed_seconds']


def train_order(length, seed):
    order = list(range(length))
    random.Random(seed).shuffle(order)
    return order
