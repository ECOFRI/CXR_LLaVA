"""CPU completeness/shape audit without printing source IDs or clinical text."""
import json

from research.rrg_f0.core import ROOT, INDEX, EXPECTED_CLEAR_SHA, cohort, pool_grid, sha256


def main():
    import torch
    torch.set_num_threads(2)
    base = ROOT / 'research/rrg_f0'
    extraction = json.loads((base / 'extraction.json').read_text())
    if extraction['status'] != 'PASS' or extraction['count'] != 2500:
        raise ValueError('Extraction is incomplete')
    config = json.loads((base / 'config.json').read_text())
    total_bytes = 0
    seen = 0
    for split, rows in cohort().items():
        for i, row in enumerate(rows):
            path = base / 'features' / f'{split}_{i:04d}.pt'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload['index'] != i or payload['split'] != split:
                raise ValueError('Feature row linkage mismatch')
            patches, scores = payload['patch_tokens'], payload['scores']
            if patches.shape != (256, 768) or patches.dtype != torch.float16 or scores.shape != (14,):
                raise ValueError('Feature shape/dtype mismatch')
            if not torch.isfinite(patches).all() or not torch.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
                raise ValueError('Nonfinite/out-of-range features')
            if i in (0, len(rows) - 1):
                if pool_grid(patches).shape != (64, 768) or payload['image_sha256'] != sha256(row['image_path']):
                    raise ValueError('Representative source hash or load-pooling mismatch')
            total_bytes += path.stat().st_size
            seen += 1
    result = {'status': 'PASS', 'count': seen, 'bytes': total_bytes, 'manifest_index_sha256': sha256(INDEX),
              'clear_sha256': EXPECTED_CLEAR_SHA, 'labels': config['finding_labels'], 'disk_shape': [256, 768],
              'disk_dtype': 'float16', 'loaded_shape_primary': [64, 768], 'source_image_rehash': 'first and last per split; all hashed at extraction'}
    (base / 'feature_contract.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
