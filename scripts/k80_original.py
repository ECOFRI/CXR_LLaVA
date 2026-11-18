"""Unquantized official v2 checkpoint, with explicit K80 compatibility changes."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from scripts.exp02_provenance import MODEL_REVISION
from scripts.manual_experiment import inputs, write_csv, sha256

PROJECT = Path(__file__).resolve().parents[1]


def prepare(original, adapted):
    """Preserve original files; adapt only BF16 casts and streaming execution."""
    adapted.mkdir(parents=True, exist_ok=True)
    for path in original.iterdir():
        if path.is_file():
            target = adapted / path.name
            if path.suffix == '.safetensors':
                if not target.exists():
                    target.symlink_to(path.resolve())
            else:
                shutil.copy2(path, target)
    source = (original / 'CXR_LLAVA_HF.py').read_text()
    patched = source.replace('torch.bfloat16', 'torch.float32')
    # KV tensors stay on the GPU assigned to their layer during cached decoding.
    patched = patched.replace(
        '    config_class = CXRLLAVAConfig',
        '    config_class = CXRLLAVAConfig\n'
        '    _skip_keys_device_placement = ["past_key_values", "past_key_value"]',
    )
    # Synchronous generation preserves arguments and surfaces errors directly.
    start = patched.index('            thread = Thread(target=self.generate, kwargs=dict(')
    end = patched.index('\n        return generated_text', start)
    patched = patched[:start] + '''            output_ids = self.generate(
                inputs=input_ids,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                stopping_criteria=[stopping_criteria],
                use_cache=True,
                generation_config=self.generation_config,
                **image_args
            )
            generated_text = self.tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            )
''' + patched[end:]
    (adapted / 'CXR_LLAVA_HF.py').write_text(patched)
    return dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                adapted_source_sha256=hashlib.sha256(patched.encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--model-dir', type=Path,
                        default=PROJECT / 'model_cache/cxr-llava-v2-original')
    parser.add_argument('--image', type=Path, default=PROJECT / 'IMG/img.jpg')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cohort', type=Path, help='Fixed CSV: report_id,image_id,image_path,ground_truth_report')
    parser.add_argument('--image-root', type=Path, default=PROJECT)
    parser.add_argument('--max-cases', type=int, default=10)
    parser.add_argument('--dataset', choices=['indiana', 'mimic'], default='indiana')
    args = parser.parse_args()
    if args.execute and (not args.output or args.output.exists()):
        parser.error('--execute requires a new --output directory')
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download('ECOFRI/CXR-LLAVA-v2', revision=MODEL_REVISION,
                          local_dir=str(args.model_dir), max_workers=3,
                          allow_patterns=['*.json', '*.py', '*.model', '*.safetensors', 'README.md'])
    if not args.execute:
        if args.cohort:
            rows = inputs(args)
            print(f'Preflight passed: {len(rows)} image/report pairs; no inference.')
            return
        print('No inference requested. Use --download or --execute --output NEW_DIR.')
        return

    # Reuse path/hash/ID validation; this runner accepts a user-fixed cohort,
    # and does not assert that any clinical GT/view review has passed.
    validation = argparse.Namespace(**vars(args))
    validation.execute = False
    rows = inputs(validation)

    import torch
    import transformers
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError('Use the K80 environment with torch==1.12.1+cu102')
    if torch.cuda.device_count() < 4:
        raise RuntimeError('Expose at least 4 GPUs via CUDA_VISIBLE_DEVICES for FP32 weights')
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    for row in rows:
        with Image.open(row['image_path']) as image:
            if image.mode not in ('L', 'RGB', 'RGBA'):
                raise ValueError('Use 8-bit L/RGB/RGBA images; DICOM/16-bit conversion is not automatic')
            image.verify()
    index = json.loads((args.model_dir / 'model.safetensors.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        if not (args.model_dir / shard).is_file():
            raise FileNotFoundError(f'Missing {shard}; run --download first')
    adapted = args.model_dir.parent / 'cxr-llava-v2-k80-fp32'
    audit = prepare(args.model_dir, adapted)
    device_map = {'vision_tower': 0, 'mm_projector': 0, 'lm_head': 0,
                  'llama.embed_tokens': 0, 'llama.norm': 3}
    device_map.update({f'llama.layers.{i}': i // 8 for i in range(32)})
    print('Loading original unquantized weights as FP32 across 4 GPUs', flush=True)
    model = transformers.AutoModel.from_pretrained(
        str(adapted), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
        device_map=device_map, use_safetensors=True,
    ).eval()
    if any(parameter.dtype == torch.bfloat16 for parameter in model.parameters()):
        raise RuntimeError('Unexpected BF16 parameter on K80')
    args.output.mkdir(parents=True, exist_ok=False)
    audit.update(model_id='ECOFRI/CXR-LLAVA-v2', revision=MODEL_REVISION,
                 torch=torch.__version__, transformers=transformers.__version__,
                 dtype='float32', quantization=None, device_map=device_map,
                 temperature=0.2, top_p=0.8, seed=42,
                 classification='subset evaluation pending labeling' if args.cohort else 'research-only smoke test; K80 adaptation',
                 dataset=args.dataset if args.cohort else None, cases=len(rows),
                 cohort_sha256=sha256(args.cohort) if args.cohort else None,
                 runner_sha256=sha256(Path(__file__)),
                 adaptations=['BF16 -> FP32', '4 GPU model dispatch',
                              'preserve per-layer KV cache device placement',
                              'synchronous generation instead of streaming thread'],
                 generation_config=model.generation_config.to_dict())
    from scripts.exp02_provenance import report_prompt_audit
    report_prompt_audit(model, args.output / 'prompt_audit')
    (args.output / 'run_config.json').write_text(json.dumps(audit, indent=2))
    if args.cohort:
        execute_cohort(model, torch, Image, args, rows)
        return
    print('Generating one report with official helper defaults (0.2, 0.8)', flush=True)
    start = time.perf_counter()
    with Image.open(args.image) as image, torch.inference_mode():
        report = model.write_radiologic_report(image)
    for i in range(4):
        torch.cuda.synchronize(i)
    if not report.strip():
        raise RuntimeError('Empty generated report')
    (args.output / 'generated_report.txt').write_text(report)
    (args.output / 'runtime.json').write_text(json.dumps(dict(
        latency_sec=time.perf_counter()-start,
        peak_vram_gb={str(i): torch.cuda.max_memory_allocated(i)/1024**3 for i in range(4)}
    ), indent=2))
    print(report, flush=True)
    print(f'Saved research-only output to {args.output}', flush=True)


def execute_cohort(model, torch, Image, args, rows):
    prefix = f'{args.dataset}_f1_subset_{len(rows)}'
    for name in ('subsets', 'predictions', 'chexpert_inputs', 'chexpert_labels'):
        (args.output / name).mkdir()
    write_csv(args.output / 'subsets' / f'{prefix}_seed42.csv', rows,
              ['report_id', 'image_id', 'image_path', 'ground_truth_report', 'image_sha256'])
    columns = ['report_id', 'image_id', 'image_path', 'ground_truth_report',
               'generated_report', 'status', 'latency_sec', 'peak_vram_gb', 'error_message']
    prediction_path = args.output / 'predictions' / f'{prefix}_predictions.csv'
    completed = []
    for index, row in enumerate(rows, 1):
        record = dict(row, generated_report='', status='failed', error_message='',
                      latency_sec=None, peak_vram_gb=None)
        for device in range(4):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        try:
            with Image.open(row['image_path']) as image, torch.inference_mode():
                report = model.write_radiologic_report(image)
            for device in range(4):
                torch.cuda.synchronize(device)
            if not report.strip():
                raise ValueError('Empty generated report')
            record.update(generated_report=report.strip(), status='success',
                          latency_sec=time.perf_counter()-start,
                          peak_vram_gb=max(torch.cuda.max_memory_allocated(i) for i in range(4))/1024**3)
        except (OSError, RuntimeError, ValueError) as error:
            record['error_message'] = repr(error)
            completed.append(record)
            write_csv(prediction_path, completed, columns)
            raise
        completed.append(record)
        write_csv(prediction_path, completed, columns)
        print(f'{index}/{len(rows)} saved; research-only output', flush=True)
    export_reports(args.output, prefix, completed)
    print(f'Inference complete: {args.output}. Run official CheXpert labeling next.', flush=True)


def export_reports(output, prefix, completed):
    folder = output / 'chexpert_inputs'
    for kind, column in [('gt', 'ground_truth_report'), ('generated', 'generated_report')]:
        write_csv(folder / f'{prefix}_{kind}_reports.csv', completed, [column], header=False)
    row_map = [dict(row, chexpert_row=i) for i, row in enumerate(completed)]
    write_csv(folder / f'{prefix}_row_map.csv', row_map, ['chexpert_row', 'report_id', 'image_id'])


if __name__ == '__main__':
    main()
