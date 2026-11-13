"""Single-GPU CXR vision-only gates; native and projector features stay separate."""
import argparse
import json
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor

from CXR_LLAVA_HF.CXR_LLAVA_HF import CLIPVisionCfg
from CXR_LLAVA_HF.VisualTransformer import LayerNorm, VisionTransformer
from research.common import BASELINE, RESEARCH, ROOT, finish_run, new_run, now, pseudonym, sha, write_csv, write_json


def load_vision(model_dir):
    cfg = json.loads((model_dir / 'config.json').read_text())
    vc = CLIPVisionCfg(**cfg['clip_vision_cfg'])
    norm = partial(LayerNorm, **vc.norm_kwargs) if vc.norm_kwargs else LayerNorm
    activation = partial(torch.nn.GELU, **vc.act_kwargs) if vc.act_kwargs else torch.nn.GELU
    tower = VisionTransformer(in_channels=1, image_size=vc.image_size, patch_size=vc.patch_size,
                              width=vc.width, layers=vc.layers, heads=vc.width // vc.head_width,
                              mlp_ratio=vc.mlp_ratio, ls_init_value=vc.ls_init_value,
                              patch_dropout=vc.patch_dropout, attentional_pool=vc.attentional_pool,
                              attn_pooler_queries=vc.attn_pooler_queries, attn_pooler_heads=vc.attn_pooler_heads,
                              pos_embed_type=vc.pos_embed_type, no_ln_pre=vc.no_ln_pre,
                              final_ln_after_pool=vc.final_ln_after_pool, pool_type=vc.pool_type,
                              output_tokens=True, output_dim=cfg['clip_embed_dim'], act_layer=activation, norm_layer=norm)
    projector = torch.nn.Linear(cfg['mm_projector_dim'], cfg['llama']['hidden_size'])
    index = json.loads((model_dir / 'model.safetensors.index.json').read_text())['weight_map']
    selected = {k: v for k, v in index.items() if k.startswith(('vision_tower.', 'mm_projector.'))}
    state = {}
    for shard in sorted(set(selected.values())):
        with safe_open(model_dir / shard, framework='pt', device='cpu') as source:
            for key, filename in selected.items():
                if filename == shard:
                    state[key] = source.get_tensor(key).float()
    tower.load_state_dict({k.removeprefix('vision_tower.'): v for k, v in state.items() if k.startswith('vision_tower.')}, strict=True)
    projector.load_state_dict({k.removeprefix('mm_projector.'): v for k, v in state.items() if k.startswith('mm_projector.')}, strict=True)
    tensor_digests = {k: __import__('hashlib').sha256(v.numpy().tobytes()).hexdigest() for k, v in state.items()}
    processor = CLIPImageProcessor(do_resize=True, size={'shortest_edge': vc.image_size}, resample=True,
                                  do_center_crop=True, crop_size=vc.image_size, do_rescale=True,
                                  rescale_factor=1 / 255, do_normalize=True,
                                  image_mean=cfg['image_preprocess_cfg']['mean'], image_std=cfg['image_preprocess_cfg']['std'],
                                  do_convert_rgb=False)
    return tower, projector, processor, cfg, tensor_digests


def run_cxr(max_images):
    if max_images not in (1, 10, 50):
        raise ValueError('Only authorized technical gates 1, 10 or 50')
    run, manifest = new_run('E0A_CXR')
    manifest.update(command=sys.argv, env=sys.prefix, precision='FP32', seed=42,
                    dataset='CheXpert', split='locked_test_technical_only',
                    checkpoint='b2224786bb90d54b1e1291171866706cfbb44e2b', estimated_gpu_hours=.5,
                    input_cohort_sha256=sha(BASELINE / 'cohort.json'), runner_sha256=sha(__file__),
                    next_action='CLEAR compatibility gate; no tuning on test', device_map={'vision_tower': 0, 'mm_projector': 0})
    wall_start = time.perf_counter()
    gpu_start = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available')
        free, _ = torch.cuda.mem_get_info(0)
        if free < 4 * 1024**3:
            raise RuntimeError('GPU0 has <4 GiB free; defer single-GPU gate')
        torch.manual_seed(42)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model_dir = ROOT / 'model_cache/cxr-llava-v2-original'
        tower, projector, processor, config, digests = load_vision(model_dir)
        assert tower.attn_pool is None and not tower.final_ln_after_pool and tower.pool_type == 'tok'
        manifest.update(vision_tensor_sha256=digests, model_config_sha256=sha(model_dir / 'config.json'),
                        source_sha256=sha(ROOT / 'CXR_LLAVA_HF/VisualTransformer.py'),
                        original_model_source_sha256=sha(model_dir / 'CXR_LLAVA_HF.py'),
                        preprocessing=processor.to_dict(), precision_note='Released weights upcast to FP32 for K80; original model uses BF16 casts')
        cohort = json.loads((BASELINE / 'cohort.json').read_text())[:max_images]
        manifest['inputs'] = [{k: row[k] for k in ['index', 'image_path', 'image_sha256']} for row in cohort]
        write_json(run / 'run_manifest.json', manifest)
        gpu_start = time.perf_counter()
        tower = tower.float().cuda().eval().requires_grad_(False)
        projector = projector.float().cuda().eval().requires_grad_(False)
        capture = {}
        handle = tower.ln_post.register_forward_hook(lambda module, args, out: capture.update(native=out))
        def features(x):
            global_projected, patches = tower(x)
            native = capture['native']
            assert native.ndim == 3 and patches.shape[1] == tower.grid_size[0] * tower.grid_size[1]
            assert torch.equal(native[:, 1:], patches)
            return {'native_cls': native[:, 0], 'native_patches': patches,
                    'native_patch_mean': patches.mean(dim=1), 'tower_global_projection': global_projected,
                    'llm_projected_patches': projector(patches)}
        cache_key = __import__('hashlib').sha256(json.dumps({'vision': digests, 'preprocess': processor.to_dict(),
                                                          'precision': 'FP32', 'cohort': manifest['input_cohort_sha256'],
                                                          'layer': 'final_ln_post_pre_mm_projector'}, sort_keys=True).encode()).hexdigest()[:20]
        cache = RESEARCH / 'cache' / ('cxr_' + cache_key) / manifest['run_id']
        cache.mkdir(parents=True, exist_ok=False)
        rows, gates = [], []
        for i, row in enumerate(cohort):
            assert sha(row['image_path']) == row['image_sha256']
            with Image.open(row['image_path']) as image:
                # Model helper native preprocessing keeps grayscale and specifies input_data_format=LAST.
                image = image.convert('L')
                import numpy as np
                pixel = np.expand_dims(np.asarray(image), -1)
                x = processor(pixel, return_tensors='pt', input_data_format='channels_last')['pixel_values'].cuda().float()
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = features(x)
            torch.cuda.synchronize()
            latency = time.perf_counter() - start
            assert all(torch.isfinite(v).all() for v in output.values())
            repeat_diff = None
            if i in (0, 9, 49):
                with torch.inference_mode():
                    repeated = features(x)
                torch.cuda.synchronize()
                repeat_diff = max(float((output[k] - repeated[k]).abs().max()) for k in output)
                assert repeat_diff <= 1e-6, repeat_diff
            saved = {'image_id': pseudonym(row['relative_path']), 'study_id': pseudonym(row['study']),
                     'features': {k: v.cpu() for k, v in output.items()}}
            path = cache / f'{i:05d}.pt'
            torch.save(saved, path)
            rows.append({'image_id': saved['image_id'], 'study_id': saved['study_id'], 'latency_sec': latency,
                         'allocated_peak_gib': torch.cuda.max_memory_allocated(0) / 1024**3,
                         'reserved_peak_gib': torch.cuda.max_memory_reserved(0) / 1024**3,
                         'determinism_max_abs_diff': repeat_diff, 'feature_shapes': json.dumps({k: list(v.shape) for k, v in output.items()}),
                         'cache_path': str(path), 'cache_sha256': sha(path), 'cache_bytes': path.stat().st_size})
            if i + 1 in (1, 10, 50):
                gates.append({'n': i + 1, 'status': 'PASS', 'repeat_abs_diff': repeat_diff,
                              'cumulative_wall_sec': time.perf_counter() - gpu_start})
                write_json(run / 'gates.json', gates)
                print('Gate', i + 1, 'PASS', 'shapes', rows[-1]['feature_shapes'], flush=True)
            del output, saved, x
        handle.remove()
        write_csv(run / 'feature_manifest.csv', rows)
        mean = sum(r['latency_sec'] for r in rows) / len(rows)
        metrics = {'n': len(rows), 'gates': gates, 'mean_latency_sec': mean,
                   'max_allocated_gib': max(r['allocated_peak_gib'] for r in rows),
                   'max_reserved_gib': max(r['reserved_peak_gib'] for r in rows),
                   'cache_bytes': sum(r['cache_bytes'] for r in rows),
                   'estimated_3000_image_cache_gib': sum(r['cache_bytes'] for r in rows) / len(rows) * 3000 / 1024**3,
                   'feature_shapes': json.loads(rows[0]['feature_shapes']), 'precision': 'FP32',
                   'note': 'Tower global projection is its 128D head, distinct from 4096D LLM patch projector. No LLM loaded.'}
        write_json(run / 'metrics.json', metrics)
        manifest.update(actual_gpu_hours=(time.perf_counter() - gpu_start) / 3600,
                        wall_sec=time.perf_counter() - wall_start, cache_key=cache_key,
                        outputs=[str(cache)], next_action='CLEAR E0A pending compatibility; data gate pending')
        finish_run(run, manifest)
        print(json.dumps({'run': str(run), **metrics}, indent=2), flush=True)
    except (RuntimeError, ValueError, AssertionError, OSError) as exc:
        (run / 'stderr.log').write_text(traceback.format_exc())
        manifest.update(blocker=str(exc), wall_sec=time.perf_counter() - wall_start,
                        actual_gpu_hours=(time.perf_counter() - gpu_start) / 3600 if gpu_start else 0)
        finish_run(run, manifest, 'FAILED')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--max-images', type=int, choices=[1, 10, 50], default=50)
    run_cxr(p.parse_args().max_images)
