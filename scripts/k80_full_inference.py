"""Run frozen Indiana cohort inference on four K80 GPUs with atomic resume."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from scripts.exp02_provenance import MODEL_REVISION, report_prompt_audit
from scripts.k80_original import prepare


PROJECT = Path(__file__).resolve().parents[1]
PREDICTION_COLUMNS = [
    'cohort_index', 'report_id', 'image_id', 'image_path', 'ground_truth_report',
    'generated_report', 'status', 'latency_sec', 'peak_vram_gb', 'error_message',
    'per_case_seed', 'image_sha256',
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, obj):
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_csv(path, rows, columns, header=True):
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore', quoting=csv.QUOTE_ALL)
        if header:
            writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def case_seed(base, report_id, image_id):
    token = f'{base}|{report_id}|{image_id}'.encode('utf-8')
    return int(hashlib.sha256(token).hexdigest()[:8], 16)


def read_cohort(path, config_path, limit):
    config = json.loads(config_path.read_text())
    if sha256(path) != config['cohort_sha256']:
        raise ValueError('Frozen cohort SHA-256 mismatch')
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) != config['n_cohort']:
        raise ValueError('Frozen cohort row count mismatch')
    if limit is not None:
        if limit < 1 or limit > len(rows):
            raise ValueError('--max-cases must be in [1, cohort size]')
        rows = rows[:limit]
    for key in ('report_id', 'image_id'):
        values = [row[key] for row in rows]
        if len(values) != len(set(values)) or any(not value for value in values):
            raise ValueError(f'Cohort {key} must be nonempty and unique')
    for row in rows:
        if row.get('view_label') != 'frontal_unspecified' or row.get('pairing_status') != 'xml_linked':
            raise ValueError(f'Cohort lacks official frontal/XML evidence: {row["image_id"]}')
        if (not row['image_id'].startswith(row['report_id'] + '_')
                or not row.get('view_evidence', '').startswith('NLM manually classified frontal;')):
            raise ValueError(f'Frozen image view or pairing evidence is invalid: {row["image_id"]}')
        if not row['ground_truth_report'].strip():
            raise ValueError(f'Empty reference: {row["report_id"]}')
        xml_path = Path(row['report_xml_path'])
        if not xml_path.is_file() or sha256(xml_path) != row['report_xml_sha256']:
            raise ValueError(f'XML pairing source missing or changed: {xml_path}')
        image_path = Path(row['image_path'])
        if not image_path.is_file() or sha256(image_path) != row['image_sha256']:
            raise ValueError(f'Image missing or changed since cohort freeze: {image_path}')
        with Image.open(image_path) as image:
            if image.mode not in ('L', 'RGB', 'RGBA'):
                raise ValueError(f'Unsupported image mode {image.mode}: {image_path}')
            image.verify()
    return rows, config


def model_shards(model_dir):
    index = json.loads((model_dir / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    for shard in shards:
        if not (model_dir / shard).is_file():
            raise FileNotFoundError(model_dir / shard)
    return shards


def read_records(output, rows):
    result = []
    for index, row in enumerate(rows):
        path = output / 'cases' / f'{index:05d}.json'
        if path.is_file():
            record = json.loads(path.read_text())
            if (record['cohort_index'] != index or record['report_id'] != row['report_id']
                    or record['image_id'] != row['image_id']
                    or record['image_sha256'] != row['image_sha256']
                    or record['ground_truth_report'] != row['ground_truth_report']):
                raise ValueError(f'Case checkpoint disagrees with frozen cohort: {path}')
            result.append(record)
    return result


def export(output, rows):
    records = read_records(output, rows)
    atomic_csv(output / 'predictions.csv', records, PREDICTION_COLUMNS)
    success = [r for r in records if r['status'] == 'success']
    inputs = output / 'chexpert_inputs'
    inputs.mkdir(exist_ok=True)
    atomic_csv(inputs / 'generated_reports.csv', success, ['generated_report'], header=False)
    atomic_csv(inputs / 'gt_reports.csv', success, ['ground_truth_report'], header=False)
    row_map = [{'chexpert_row': i, 'cohort_index': r['cohort_index'],
                'report_id': r['report_id'], 'image_id': r['image_id']}
               for i, r in enumerate(success)]
    atomic_csv(inputs / 'row_map.csv', row_map,
               ['chexpert_row', 'cohort_index', 'report_id', 'image_id'])
    counts = {'requested': len(rows), 'attempted': len(records), 'success': len(success),
              'failed': sum(r['status'] == 'failed' for r in records),
              'not_attempted': len(rows) - len(records)}
    atomic_json(output / 'progress.json', counts)
    return counts


def run(args, rows):
    import torch
    import transformers

    if torch.__version__ != '1.12.1+cu102' or transformers.__version__ != '4.36.2':
        raise RuntimeError('Use the verified cxr-llava-k80 environment')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError('Four CUDA-visible GPUs are required')
    for device in range(4):
        free, _ = torch.cuda.mem_get_info(device)
        threshold = 10 if device == 0 else 8
        if free / 1024**3 < threshold:
            raise RuntimeError(f'GPU {device} has only {free / 1024**3:.2f} GiB free')
    torch.cuda.set_device(0)

    existing = read_records(args.output, rows)
    if len(existing) == len(rows) and all(r['status'] == 'success' for r in existing):
        return export(args.output, rows)
    adapted = args.model_dir.parent / 'cxr-llava-v2-k80-fp32'
    audit = prepare(args.model_dir, adapted)
    device_map = {'vision_tower': 0, 'mm_projector': 0, 'lm_head': 0,
                  'llama.embed_tokens': 0, 'llama.norm': 3}
    device_map.update({f'llama.layers.{i}': i // 8 for i in range(32)})
    print('Loading original unquantized weights as FP32 across four K80 GPUs', flush=True)
    model = transformers.AutoModel.from_pretrained(
        str(adapted), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
        device_map=device_map, use_safetensors=True,
    ).eval()
    if any(parameter.dtype == torch.bfloat16 for parameter in model.parameters()):
        raise RuntimeError('Unexpected BF16 parameter on K80')
    if not (args.output / 'prompt_audit').exists():
        report_prompt_audit(model, args.output / 'prompt_audit')
    if not (args.output / 'model_adaptation.json').exists():
        atomic_json(args.output / 'model_adaptation.json', audit | {
            'device_map': device_map, 'torch': torch.__version__,
            'transformers': transformers.__version__, 'generation_config': model.generation_config.to_dict(),
        })

    success_count = sum(record['status'] == 'success' for record in existing)
    attempted_count = len(existing)
    new_cases = 0
    for index, row in enumerate(rows):
        case_path = args.output / 'cases' / f'{index:05d}.json'
        existing_case = case_path.is_file()
        if existing_case:
            prior = json.loads(case_path.read_text())
            if prior['status'] == 'success' or not args.retry_failed:
                continue
            archive = args.output / 'attempts' / f'{index:05d}.{time.time_ns()}.json'
            shutil.copy2(case_path, archive)
        seed = case_seed(args.seed, row['report_id'], row['image_id'])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        record = {
            'cohort_index': index, 'report_id': row['report_id'], 'image_id': row['image_id'],
            'image_path': row['image_path'], 'ground_truth_report': row['ground_truth_report'],
            'generated_report': '', 'status': 'failed', 'latency_sec': None,
            'peak_vram_gb': None, 'error_message': '', 'per_case_seed': seed,
            'image_sha256': row['image_sha256'],
            'started_utc': datetime.now(timezone.utc).isoformat(),
        }
        for device in range(4):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        try:
            with Image.open(row['image_path']) as image, torch.inference_mode():
                generated = model.write_radiologic_report(image)
            for device in range(4):
                torch.cuda.synchronize(device)
            if not str(generated).strip():
                raise ValueError('Empty generated report')
            record['generated_report'] = str(generated).strip()
            record['status'] = 'success'
            record['latency_sec'] = time.perf_counter() - start
            record['peak_vram_gb'] = max(torch.cuda.max_memory_allocated(i) for i in range(4)) / 1024**3
        except (OSError, RuntimeError, ValueError) as error:
            record['error_message'] = repr(error)
            record['latency_sec'] = time.perf_counter() - start
        record['completed_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(case_path, record)
        if not case_path.is_file():
            raise RuntimeError(f'Case checkpoint was not saved: {case_path}')
        if record['status'] == 'success':
            success_count += 1
        if not existing_case:
            attempted_count += 1
        new_cases += 1
        atomic_json(args.output / 'progress.json', {
            'requested': len(rows), 'attempted': min(attempted_count, len(rows)),
            'success': success_count, 'failed': max(0, attempted_count - success_count),
            'not_attempted': max(0, len(rows) - attempted_count),
        })
        if (index + 1) % 25 == 0 or index + 1 == len(rows) or record['status'] != 'success':
            export(args.output, rows)
        print(f'{index + 1}/{len(rows)} {record["status"]}; {success_count} successes saved', flush=True)
        if record['status'] != 'success':
            raise RuntimeError(f'Case {index} failed; inspect {case_path}; retry only with --retry-failed')
        if args.stop_after is not None and new_cases >= args.stop_after:
            print(f'Stopped after {new_cases} new case(s) for resume gate', flush=True)
            return export(args.output, rows)
    return export(args.output, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=PROJECT / 'model_cache/cxr-llava-v2-original')
    parser.add_argument('--gt-provenance', type=Path,
                        help='Completed pinned GT-labeling provenance for this frozen cohort')
    parser.add_argument('--max-cases', type=int)
    parser.add_argument('--stop-after', type=int, help='Stop after N newly generated cases to exercise resume')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    args.cohort = args.cohort.resolve()
    args.output = args.output.resolve()
    args.model_dir = args.model_dir.resolve()
    if args.gt_provenance is None:
        args.gt_provenance = args.cohort.parent.parent / 'gt_labeling' / 'provenance.json'
    args.gt_provenance = args.gt_provenance.resolve()
    if args.resume and not args.execute:
        parser.error('--resume requires --execute')
    if args.stop_after is not None and args.stop_after < 1:
        parser.error('--stop-after must be positive')
    if args.retry_failed and not args.resume:
        parser.error('--retry-failed requires --resume')
    if args.export_only and args.execute:
        parser.error('--export-only cannot be combined with --execute')
    rows, cohort_config = read_cohort(args.cohort, args.cohort.parent / 'cohort_config.json', args.max_cases)
    gt_provenance = json.loads(args.gt_provenance.read_text())
    if (gt_provenance.get('cohort_sha256') != sha256(args.cohort)
            or gt_provenance.get('sample_gate') != 'PASS'
            or gt_provenance.get('alignment') != 'PASS'):
        raise ValueError('GT labeling is missing or misaligned with frozen cohort')
    if args.export_only:
        if not args.output.is_dir():
            raise FileNotFoundError(args.output)
        print(json.dumps(export(args.output, rows), indent=2))
        return
    shards = model_shards(args.model_dir)
    base_config = {
        'model_id': 'ECOFRI/CXR-LLAVA-v2', 'model_revision': MODEL_REVISION,
        'cohort_sha256': sha256(args.cohort),
        'cohort_config_sha256': sha256(args.cohort.parent / 'cohort_config.json'),
        'cases_requested': len(rows), 'max_cases': args.max_cases,
        'seed': args.seed, 'seed_policy': 'independent SHA256(base_seed|report_id|image_id) per case; not the 50-case global RNG stream',
        'model_dir': str(args.model_dir),
        'model_source_sha256': sha256(args.model_dir / 'CXR_LLAVA_HF.py'),
        'vision_source_sha256': sha256(args.model_dir / 'VisualTransformer.py'),
        'model_config_sha256': sha256(args.model_dir / 'config.json'),
        'generation_config_sha256': sha256(args.model_dir / 'generation_config.json'),
        'tokenizer_model_sha256': sha256(args.model_dir / 'tokenizer.model'),
        'model_shard_sha256': {name: sha256(args.model_dir / name) for name in shards},
        'runner_sha256': sha256(Path(__file__)),
        'gt_provenance_sha256': sha256(args.gt_provenance),
    }
    if not args.execute:
        print(f'Preflight PASS: {len(rows)} frozen frontal cases; model revision {MODEL_REVISION}')
        print('Inference not requested; add --execute --output NEW_DIR')
        return
    lock_path = PROJECT / 'model_cache/.k80_full_inference.lock'
    with lock_path.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another K80 full-cohort inference holds the model lock') from error
        if args.resume:
            if not args.output.is_dir():
                raise FileNotFoundError(args.output)
            prior = json.loads((args.output / 'run_config.json').read_text())
            for key, value in base_config.items():
                if prior[key] != value:
                    raise ValueError(f'Resume config differs at {key}')
            read_records(args.output, rows)
            (args.output / 'status.txt').write_text('RUNNING\n')
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / 'cases').mkdir()
            (args.output / 'attempts').mkdir()
            commit = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip()
            dirty = subprocess.check_output(['git', '-C', str(PROJECT), 'status', '--porcelain'], text=True)
            atomic_json(args.output / 'run_config.json', base_config | {
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'git_commit': commit, 'git_status': dirty,
                'cohort_path': str(args.cohort), 'model_precision': 'FP32',
                'gpu_policy': '4 CUDA-visible Tesla K80 devices',
                'generation': 'official write_radiologic_report defaults temperature=0.2, top_p=0.8',
                'classification': 'reconstructed-cohort external evaluation; not exact paper reproduction',
            })
            shutil.copy2(args.cohort, args.output / 'cohort.csv')
            shutil.copy2(args.cohort.parent / 'cohort_config.json', args.output / 'cohort_config.json')
            shutil.copy2(args.gt_provenance, args.output / 'gt_provenance.json')
            snapshots = args.output / 'snapshots'
            snapshots.mkdir()
            for script in ('k80_full_inference.py', 'k80_original.py', 'select_indiana_nlm_views.py',
                           'label_indiana_gt.py', 'evaluate_indiana_full.py',
                           'write_indiana_full_report.py', 'exp02_provenance.py',
                           'run_indiana_full_k80.sh'):
                shutil.copy2(PROJECT / 'scripts' / script, snapshots / script)
            (snapshots / 'inference_pip_freeze.txt').write_text(
                subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
            (snapshots / 'tracked_changes.patch').write_bytes(
                subprocess.check_output(['git', '-C', str(PROJECT), 'diff', '--binary']))
            (args.output / 'status.txt').write_text('RUNNING\n')
        try:
            counts = run(args, rows)
            status = 'COMPLETED' if counts['success'] == len(rows) else 'PARTIAL'
            (args.output / 'status.txt').write_text(status + '\n')
            print(f'{status}: {counts}; output {args.output}', flush=True)
        except BaseException:
            (args.output / 'status.txt').write_text('FAILED\n')
            raise


if __name__ == '__main__':
    main()
