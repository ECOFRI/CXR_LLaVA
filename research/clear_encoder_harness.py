"""Prepared official CLEAR native-feature gates, CPU by default; no installs/downloads."""
import argparse
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

from research.common import BASELINE, RESEARCH, ROOT, finish_run, new_run, pseudonym, sha, write_csv, write_json


def run_clear(max_images):
    wall_start = time.perf_counter()
    if max_images != 1:
        raise ValueError('Current approval is exactly one image; gate 10/50 requires review')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only approval requires CUDA_VISIBLE_DEVICES empty')
    import torch
    if torch.version.cuda is not None or torch.cuda.is_initialized():
        raise RuntimeError('CPU-only PyTorch build is required')
    if sys.version_info < (3, 10) or not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        raise RuntimeError('CLEAR requires Python>=3.10 and verified PyTorch SDPA; run compatibility preflight first')
    run, manifest = new_run('E0A_CLEAR_CPU')
    manifest.update(env=sys.prefix, command=sys.argv, dataset='CheXpert', split='locked_test_technical_only',
                    precision='FP32', device_map={'CLEAR': 'cpu'}, actual_gpu_hours=0,
                    input_cohort_sha256=sha(BASELINE / 'cohort.json'), runner_sha256=sha(__file__))
    manifest.update(cpu_threads=4, approval='User approved isolated CPU environment and one unique image only',
                    executable=sys.executable, estimated_gpu_hours=0,
                    max_images=1, next_action='STOP_FOR_ONE_IMAGE_REVIEW; no gate 10/50 approved')
    try:
        clear_repo = ROOT.parent / 'CLEAR'
        dino = RESEARCH / 'upstream/dinov2'
        sys.path.insert(0, str(clear_repo / 'src'))
        hub_cache = RESEARCH / 'cache/torch_hub'
        hub_cache.mkdir(parents=True, exist_ok=True)
        architecture_link = hub_cache / 'facebookresearch_dinov2_main'
        if not architecture_link.exists():
            architecture_link.symlink_to(dino, target_is_directory=True)
        assert architecture_link.resolve() == dino.resolve()
        torch.hub.set_dir(str(hub_cache))
        from clear import load_pretrained
        from PIL import Image
        checkpoints = list(Path('/home/student4/.cache/huggingface/hub/models--peterhan91--CLEAR/snapshots').glob('*/best_model.pt'))
        assert len(checkpoints) == 1
        checkpoint = checkpoints[0]
        expected_checkpoint_sha = 'da1617a7d47970667a7a158eabfa4c4f3798d5d03b7f98caec150d8ee6ab6a03'
        before_checkpoint_sha = sha(checkpoint)
        assert before_checkpoint_sha == expected_checkpoint_sha, 'Official cached checkpoint changed'
        torch.manual_seed(42)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        load_start = time.perf_counter()
        model, preprocess = load_pretrained(checkpoint_path=checkpoint, device='cpu', local_files_only=True)
        model = model.float().eval().requires_grad_(False)
        load_wall_sec = time.perf_counter() - load_start
        assert all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters())
        manifest.update(checkpoint=checkpoint.parent.name, checkpoint_sha256=sha(checkpoint),
                        clear_source_sha256=sha(clear_repo / 'src/clear/hub.py'),
                        dinov2_attention_sha256=sha(dino / 'dinov2/layers/attention.py'),
                        preprocess=repr(preprocess), torch=torch.__version__,
                        note='Official strict full-checkpoint load; native DINO features before CLEAR visual projection; K80 not tested')
        key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
        cache = RESEARCH / 'cache' / ('clear_' + key)
        cache.mkdir(parents=True, exist_ok=False)
        rows, gates = [], []
        start = time.perf_counter()
        def extract(x):
            f = model.visual.backbone.forward_features(x)
            result = {k: f[k] for k in ('x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens')}
            width = model.visual.backbone.embed_dim
            assert result['x_norm_clstoken'].shape == (x.shape[0], width)
            assert result['x_norm_regtokens'].ndim == 3 and result['x_norm_regtokens'].shape[-1] == width
            assert result['x_norm_patchtokens'].ndim == 3 and result['x_norm_patchtokens'].shape[-1] == width
            result['native_patch_mean'] = f['x_norm_patchtokens'].mean(dim=1)
            result['clear_image_text_projection'] = model.visual.projection(f['x_norm_clstoken'])
            patch_size = model.visual.backbone.patch_embed.patch_size
            count = (x.shape[-2] // patch_size[0]) * (x.shape[-1] // patch_size[1])
            assert result['x_norm_patchtokens'].shape[1] == count
            assert result['x_norm_regtokens'].shape[1] == model.visual.backbone.num_register_tokens
            return result
        for i, row in enumerate(json.loads((BASELINE / 'cohort.json').read_text())[:max_images]):
            assert sha(row['image_path']) == row['image_sha256']
            with Image.open(row['image_path']) as image:
                x = preprocess(image).unsqueeze(0).float()
            t = time.perf_counter()
            with torch.inference_mode():
                output = extract(x)
            latency = time.perf_counter() - t
            assert all(torch.isfinite(v).all() for v in output.values())
            delta = None
            if i in (0, 9, 49):
                with torch.inference_mode():
                    repeat = extract(x)
                    delta = max(float((output[k] - repeat[k]).abs().max()) for k in output)
                assert delta <= 1e-6
            official_start = time.perf_counter()
            with torch.inference_mode():
                official_features = model.encode_image(x)
            official_latency = time.perf_counter() - official_start
            official_delta = float((official_features - output['clear_image_text_projection']).abs().max())
            assert official_delta <= 1e-6
            assert torch.isfinite(official_features).all()
            f = cache / f'{i:05d}.pt'
            torch.save({'image_id': pseudonym(row['relative_path']), 'study_id': pseudonym(row['study']), 'features': output}, f)
            rows.append({'image_id': pseudonym(row['relative_path']), 'study_id': pseudonym(row['study']),
                         'latency_sec': latency, 'determinism_max_abs_diff': delta,
                         'official_encode_image_latency_sec': official_latency,
                         'official_encode_image_max_abs_diff': official_delta,
                         'feature_shapes': json.dumps({k: list(v.shape) for k, v in output.items()}),
                         'cache_path': str(f), 'cache_sha256': sha(f), 'cache_bytes': f.stat().st_size})
            if i + 1 in (1, 10, 50):
                gates.append({'n': i + 1, 'status': 'PASS', 'device': 'cpu', 'repeat_abs_diff': delta})
                write_json(run / 'gates.json', gates)
                print('CLEAR CPU gate', i + 1, 'PASS', flush=True)
        write_csv(run / 'feature_manifest.csv', rows)
        after_checkpoint_sha = sha(checkpoint)
        assert after_checkpoint_sha == before_checkpoint_sha
        assert not torch.cuda.is_initialized()
        peak_ram_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        assert peak_ram_bytes < 8 * 1024**3, 'Exceeded approved peak RAM budget'
        write_json(run / 'metrics.json', {'gates': gates, 'wall_sec': time.perf_counter() - wall_start,
                   'feature_gate_wall_sec': time.perf_counter() - start,
                   'load_wall_sec': load_wall_sec, 'peak_ram_bytes': peak_ram_bytes,
                   'cpu_user_sec': resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                   'cpu_system_sec': resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                   'unique_images': len(rows), 'cpu_threads': 4, 'torch_cuda_build': torch.version.cuda,
                   'preprocessed_input_shape': list(x.shape),
                   'register_tokens': model.visual.backbone.num_register_tokens,
                   'patch_size': list(model.visual.backbone.patch_embed.patch_size),
                   'numeric_summary': {k: {'finite': bool(torch.isfinite(v).all()),
                                          'min': float(v.min()), 'max': float(v.max()),
                                          'mean': float(v.mean()), 'std': float(v.std())}
                                       for k, v in output.items()},
                   'cuda_initialized': torch.cuda.is_initialized(), 'checkpoint_integrity': 'PASS_BEFORE_AND_AFTER',
                   'checkpoint_sha256': after_checkpoint_sha,
                   'mean_latency_sec': sum(r['latency_sec'] for r in rows) / len(rows),
                   'shapes': json.loads(rows[0]['feature_shapes']), 'K80_measurements': 'NOT_EXECUTED'})
        manifest.update(outputs=[str(cache)], next_action='STOP_FOR_ONE_IMAGE_REVIEW; gate 10/50 not approved')
        finish_run(run, manifest)
    except (RuntimeError, AttributeError, ValueError, AssertionError, OSError, ImportError) as exc:
        (run / 'stderr.log').write_text(traceback.format_exc())
        manifest['blocker'] = str(exc)
        finish_run(run, manifest, 'FAILED')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--max-images', type=int, choices=[1], default=1)
    run_clear(p.parse_args().max_images)
