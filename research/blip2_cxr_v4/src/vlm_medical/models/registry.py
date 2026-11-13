"""Local-only checkpoint and encoder resolution."""
from pathlib import Path


class MissingCheckpointError(FileNotFoundError):
    pass


def resolve_checkpoint(path, required=('config.json', 'model.safetensors.index.json')):
    root = Path(path)
    missing = [name for name in required if not (root / name).is_file()]
    if not root.is_dir() or missing:
        raise MissingCheckpointError('Local checkpoint incomplete: ' + str(root) + '; missing ' + ', '.join(missing))
    return root


def resolve_encoder(name, path):
    if name == 'blip2':
        return resolve_checkpoint(path)
    root = Path(path)
    if not root.is_dir():
        raise MissingCheckpointError('Local encoder unavailable: ' + name)
    return root
