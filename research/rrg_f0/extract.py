"""Frozen CLEAR features on the unchanged P4 manifests; local-only outputs."""
import argparse
import json
import time

from research.rrg_f0.core import ROOT, cohort, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--benchmark', action='store_true', help='Exactly first 20 train images, no feature output')
    args = parser.parse_args()
    import torch
    from PIL import Image
    from research.rrg_f0.encoder import FrozenClear
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    base = ROOT / 'research/rrg_f0'
    config = json.loads((base / 'config.json').read_text())
    encoder = FrozenClear(config['finding_labels'], args.device)
    preprocess = encoder.preprocess
    rows_by_split = cohort()
    selected = [('mimic_train', rows_by_split['mimic_train'][:20])] if args.benchmark else list(rows_by_split.items())
    if args.device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    timings = []
    feature_dir = base / 'features'
    if not args.benchmark:
        feature_dir.mkdir(exist_ok=True)
    start = time.perf_counter()
    with torch.inference_mode():
        for split, rows in selected:
            for i, row in enumerate(rows):
                t = time.perf_counter()
                with Image.open(row['image_path']) as image:
                    inputs = preprocess(image).unsqueeze(0).to(args.device)
                pooled, scores = encoder.encode(inputs)
                pooled, scores = pooled.squeeze(0), scores.squeeze(0)
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t
                timings.append(elapsed)
                if not args.benchmark:
                    path = feature_dir / f'{split}_{i:04d}.pt'
                    if path.exists():
                        raise FileExistsError('Existing F0 feature must be preserved')
                    torch.save({'patch_tokens': pooled, 'scores': scores, 'image_sha256': sha256(row['image_path']),
                                'split': split, 'index': i}, path)
                if (i + 1) % 20 == 0:
                    print(json.dumps({'split': split, 'images_completed': i + 1, 'mean_seconds_per_image': sum(timings) / len(timings)}), flush=True)
    result = {'status': 'PASS', 'device': args.device, 'count': len(timings), 'seconds_per_image_compute_and_preprocess': sum(timings) / len(timings),
              'wall_seconds_including_feature_writes': time.perf_counter() - start, 'disk_shape': [256, 768], 'disk_dtype': 'float16',
              'input_size': 448, 'zero_shot': 'lowercase label / no label, normalized paired cosine softmax; no logit scale',
              'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if args.device == 'cuda' else None,
              'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20 if args.device == 'cuda' else None}
    name = f'benchmark_{args.device}.json' if args.benchmark else 'extraction.json'
    (base / name).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
