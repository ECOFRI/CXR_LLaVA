"""Approval-gated, local-only training helpers."""
import json
import random
import shutil
from pathlib import Path


def seed_all(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)


def require_approval(vr, experiment_id):
    path = Path(vr) / 'approvals' / (experiment_id + '.yaml')
    if not path.is_file():
        raise PermissionError('Training requires explicit approval file: ' + str(path))
    return path


def save_checkpoint(path, model, optimizer, step, config):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    weights = {name: value.detach().cpu() for name, value in model.state_dict().items() if name in trainable}
    torch.save({'model': weights, 'optimizer': optimizer.state_dict(), 'step': step, 'config': config}, path)


def resume_checkpoint(path, model, optimizer):
    import torch
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model'], strict=False)
    optimizer.load_state_dict(state['optimizer'])
    return state['step']


def log_status(path, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields, indent=2) + '\n')


def preflight_storage(path, minimum_free_gib=5):
    free = shutil.disk_usage(Path(path).parent).free / 2**30
    if free < minimum_free_gib:
        raise RuntimeError('Insufficient free output disk space')
    return round(free, 2)


def preflight_vram(required_free_gib):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA device required for this entrypoint')
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gib = free_bytes / 2**30
    if free_gib < required_free_gib:
        raise RuntimeError('Insufficient free GPU memory for arm preflight')
    return {'free_gib': round(free_gib, 2), 'total_gib': round(total_bytes / 2**30, 2)}
