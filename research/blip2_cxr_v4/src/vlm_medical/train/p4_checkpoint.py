"""P4 projection checkpoint identity and exact resume helpers."""
import hashlib
import os
import random
from pathlib import Path

ALLOWLIST = {'language_projection.layer.weight', 'language_projection.layer.bias'}


def load_adapter(model, payload, config_sha256, manifest_sha256, *, allow_smoke=False):
    adapter = payload['adapter']
    if set(adapter) != ALLOWLIST:
        raise ValueError('Adapter key mismatch')
    if payload.get('config_sha256') != config_sha256 or payload.get('manifest_sha256') != manifest_sha256:
        raise ValueError('Adapter config or manifest identity mismatch')
    if payload.get('stage') == 'P4.1' and not allow_smoke:
        raise ValueError('P4.1 adapter cannot initialize P4.2 or M1')
    named = dict(model.named_parameters())
    if not ALLOWLIST <= set(named):
        raise ValueError('Architecture projection keys missing')
    for key in ALLOWLIST:
        if tuple(adapter[key].shape) != tuple(named[key].shape):
            raise ValueError('Adapter tensor shape mismatch: ' + key)
    for key in ALLOWLIST:
        named[key].data.copy_(adapter[key].to(named[key].device, dtype=named[key].dtype))
    for key in ALLOWLIST:
        if not named[key].detach().cpu().equal(adapter[key].to(dtype=named[key].dtype)):
            raise RuntimeError('Adapter reload differs: ' + key)


def save_state(path, model, optimizer, config_sha256, manifest_sha256, next_position, updates, order, elapsed_seconds):
    import numpy as np
    import torch
    if not 0 <= next_position <= len(order) or updates != next_position:
        raise ValueError('Resume position/update accounting mismatch')
    named = dict(model.named_parameters())
    state = {
        'schema_version': 1, 'stage': 'P4.2', 'config_sha256': config_sha256,
        'manifest_sha256': manifest_sha256, 'next_position': next_position,
        'updates': updates, 'order_sha256': hashlib.sha256(','.join(map(str, order)).encode()).hexdigest(),
        'optimizer': optimizer.state_dict(),
        'adapter': {name: named[name].detach().cpu() for name in ALLOWLIST},
        'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'elapsed_seconds': float(elapsed_seconds),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def resume_state(path, model, optimizer, config_sha256, manifest_sha256, order):
    import numpy as np
    import torch
    state = torch.load(path, map_location='cpu')
    if state.get('schema_version') != 1 or state.get('stage') != 'P4.2':
        raise ValueError('Wrong resume checkpoint stage/schema')
    if state.get('order_sha256') != hashlib.sha256(','.join(map(str, order)).encode()).hexdigest():
        raise ValueError('Training order identity mismatch')
    position = state.get('next_position')
    if not isinstance(position, int) or not 0 <= position <= len(order) or state.get('updates') != position:
        raise ValueError('Resume position/update accounting mismatch')
    load_adapter(model, state, config_sha256, manifest_sha256)
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return position, float(state['elapsed_seconds'])
