"""F0-only cohort and token helpers. No model or data access at import."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / 'research/blip2_cxr_v4/runs/P4_projection_pilot_20261023/manifest_index.approved.json'
EXPECTED_CLEAR_SHA = 'da1617a7d47970667a7a158eabfa4c4f3798d5d03b7f98caec150d8ee6ab6a03'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def cohort():
    index = json.loads(INDEX.read_text())
    if not index['approved'] or index['status'] != 'APPROVED_P4_20261023':
        raise ValueError('Unapproved P4 index')
    result = {}
    for name, count, split in [('mimic_train', 2000, 'train'), ('mimic_validate', 500, 'validate')]:
        item = index['manifests'][name]
        path = ROOT / item['path']
        if sha256(path) != item['sha256']:
            raise ValueError('P4 manifest hash mismatch')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != count or any(row['split'] != split for row in rows):
            raise ValueError('P4 count or split mismatch')
        if len({(row['subject_id'], row['study_id']) for row in rows}) != count:
            raise ValueError('P4 duplicated studies')
        result[name] = rows
    patients = [{r['subject_id'] for r in rows} for rows in result.values()]
    if patients[0] & patients[1]:
        raise ValueError('Train/validation patient overlap')
    return result


def pool_grid(tokens, factor=2):
    """Spatial average, preserving row-major order and FP32 accumulation."""
    side = int(tokens.shape[-2] ** 0.5)
    if side * side != tokens.shape[-2] or side % factor:
        raise ValueError('Expected square patch grid divisible by pooling factor')
    batch_shape = tokens.shape[:-2]
    width = tokens.shape[-1]
    grid = tokens.float().reshape(*batch_shape, side // factor, factor, side // factor, factor, width)
    return grid.mean(dim=(-4, -2)).reshape(*batch_shape, (side // factor) ** 2, width).to(tokens.dtype)
