"""Evaluate the released CXR-LLaVA checkpoint on the 518 CheXpert frontal test images.

The paper used direct binary-label generation, but did not publish its exact
prompt/decoding code. This is a documented local approximation, not an exact
reproduction of Table 3.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from scripts.exp02_provenance import MODEL_REVISION
from scripts.k80_original import prepare


PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT.parent / 'datasets/CheXpert_test_labels'
LABELS = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
          'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
          'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
          'Fracture', 'Support Devices']
PAPER_F1 = {'Atelectasis': .69, 'Cardiomegaly': .62, 'Consolidation': .24,
            'Edema': .50, 'Pleural Effusion': .63, 'Lung Opacity': .84,
            'Support Devices': .78}
PRIMARY = list(PAPER_F1)
QUESTION = 'Is {finding} present in this chest X-ray? Answer only Yes or No.'
SYSTEM = ('You are a helpful radiologist. Try to interpret chest x ray image '
          'and answer to the question that user provides.')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cohort(dataset):
    image_csv = dataset / 'CheXpert/test_labels.csv'
    study_csv = dataset / 'groundtruth.csv'
    all_images = read_csv(image_csv)
    ground = {row['Study'].removeprefix('CheXpert-v1.0/'): row
              for row in read_csv(study_csv)}
    if len(all_images) != 668 or len(ground) != 500:
        raise ValueError('Expected 668 image labels and 500 unique study labels')
    rows = []
    for row in all_images:
        if not row['Path'].endswith('_frontal.jpg'):
            continue
        relative = Path(row['Path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe image path: {relative}')
        study = str(relative.parent)
        if study not in ground:
            raise ValueError(f'Study absent from official groundtruth: {study}')
        labels = {name: int(float(row[name])) for name in LABELS}
        if labels != {name: int(float(ground[study][name])) for name in LABELS}:
            raise ValueError(f'Image/study label mismatch: {relative}')
        candidates = [dataset / 'CheXpert' / relative,
                      dataset / 'CheXpert/test' / relative]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise ValueError(f'Expected one local image for {relative}: {existing}')
        path = existing[0].resolve()
        with Image.open(path) as image:
            if image.mode not in ('L', 'RGB', 'RGBA'):
                raise ValueError(f'Unsupported mode: {path} = {image.mode}')
            image.verify()
        rows.append({'index': len(rows), 'relative_path': str(relative),
                     'study': study, 'image_path': str(path),
                     'image_sha256': sha256(path), 'labels': labels})
    if len(rows) != 518 or len({r['relative_path'] for r in rows}) != 518:
        raise ValueError('Expected 518 unique frontal images')
    paper_support = {'No Finding': (450, 68), 'Cardiomegaly': (364, 154),
                     'Consolidation': (489, 29), 'Edema': (439, 79),
                     'Lung Opacity': (246, 272), 'Pleural Effusion': (413, 105),
                     'Atelectasis': (360, 158), 'Support Devices': (252, 266)}
    for name, (negative, positive) in paper_support.items():
        if (sum(r['labels'][name] == 0 for r in rows),
                sum(r['labels'][name] == 1 for r in rows)) != (negative, positive):
            raise ValueError(f'Paper supplementary GT support mismatch: {name}')
    return rows, {'image_csv_sha256': sha256(image_csv),
                  'study_csv_sha256': sha256(study_csv),
                  'cohort_n': 518, 'all_images': 668, 'all_studies': 500,
                  'source': 'CheXlocalize CheXpert/test; official frontal and labels',
                  'paper_support_match': True}


def parse_binary(text):
    """Accept an explicit first-word yes/no; leave all other answers unparsed."""
    match = re.match(r'^\s*(yes|no)\b', text, re.IGNORECASE)
    return int(match.group(1).lower() == 'yes') if match else None


def records(output, rows):
    result = []
    for row in rows:
        path = output / 'cases' / f'{row["index"]:05d}.json'
        if path.exists():
            item = json.loads(path.read_text())
            if (item['relative_path'] != row['relative_path']
                    or item['image_sha256'] != row['image_sha256']):
                raise ValueError(f'Checkpoint does not match frozen cohort: {path}')
            result.append(item)
    return result


def progress(output, rows):
    items = records(output, rows)
    counts = {'requested': len(rows), 'attempted': len(items),
              'generated': sum(item['status'] == 'success' for item in items),
              'failed': sum(item['status'] == 'failed' for item in items),
              'parsed_labels': sum(len(item.get('parsed_labels', {})) for item in items),
              'total_labels': len(rows) * len(LABELS)}
    save_json(output / 'progress.json', counts)
    return counts


def infer(args, rows, source):
    import torch
    import transformers

    if torch.__version__ != '1.12.1+cu102' or transformers.__version__ != '4.36.2':
        raise RuntimeError('Activate the verified cxr-llava-k80 environment')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError('Four CUDA-visible K80 GPUs are required')
    for device in range(4):
        free, _ = torch.cuda.mem_get_info(device)
        if free / 1024**3 < (10 if device == 0 else 8):
            raise RuntimeError(f'GPU {device} has insufficient free memory')
    model_dir = args.model_dir.resolve()
    index = json.loads((model_dir / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    config = {'cohort_source': source, 'model_revision': MODEL_REVISION,
              'model_dir': str(model_dir), 'model_config_sha256': sha256(model_dir / 'config.json'),
              'model_shard_sha256': {name: sha256(model_dir / name) for name in shards},
              'runner_sha256': sha256(__file__), 'question_template': QUESTION, 'system': SYSTEM,
              'seed': args.seed, 'seed_policy': 'per-image SHA256-derived',
              'temperature': .2, 'top_p': .8, 'precision': 'FP32',
              'cohort_image_paths_sha256': hashlib.sha256(
                  '\n'.join(r['relative_path'] + ':' + r['image_sha256'] for r in rows).encode()).hexdigest(),
              'classification': 'matched 518-image cohort; 14 locally chosen yes/no questions; not exact Table 3 reproduction'}
    output = args.output.resolve()
    if args.resume:
        prior = json.loads((output / 'run_config.json').read_text())
        if prior != config:
            raise ValueError('Resume configuration/hash mismatch')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'cases').mkdir()
        (output / 'snapshots').mkdir()
        save_json(output / 'run_config.json', config)
        save_json(output / 'cohort.json', rows)
        shutil.copy2(__file__, output / 'snapshots/chexpert_table3_k80.py')
        (output / 'snapshots/git_commit.txt').write_text(
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True))
        (output / 'snapshots/git_status.txt').write_text(
            subprocess.check_output(['git', 'status', '--porcelain'], text=True))
        (output / 'snapshots/pip_freeze.txt').write_text(
            subprocess.check_output([os.sys.executable, '-m', 'pip', 'freeze'], text=True))
    if all(item['status'] == 'success' for item in records(output, rows)) and len(records(output, rows)) == len(rows):
        return progress(output, rows)
    adapted = model_dir.parent / 'cxr-llava-v2-k80-fp32'
    adaptation = prepare(model_dir, adapted)
    save_json(output / 'model_adaptation.json', adaptation)
    device_map = {'vision_tower': 0, 'mm_projector': 0, 'lm_head': 0,
                  'llama.embed_tokens': 0, 'llama.norm': 3}
    device_map.update({f'llama.layers.{i}': i // 8 for i in range(32)})
    print('Loading CXR-LLaVA FP32 on four K80 GPUs', flush=True)
    model = transformers.AutoModel.from_pretrained(
        str(adapted), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
        device_map=device_map, use_safetensors=True).eval()
    (output / 'rendered_prompt.txt').write_text(model.apply_chat_template([
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': '<image>\n' + QUESTION.format(finding='Cardiomegaly')}]))
    for row in rows:
        path = output / 'cases' / f'{row["index"]:05d}.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record['status'] == 'success':
                continue
        else:
            record = {key: row[key] for key in ('index', 'relative_path', 'study', 'image_sha256')}
            record.update(status='partial', responses={}, parsed_labels={},
                          latency_sec=0.0, peak_vram_gb_max_device=0.0)
        with Image.open(row['image_path']) as image, torch.inference_mode():
            for name in LABELS:
                if name in record['responses']:
                    continue
                seed = int(hashlib.sha256(
                    f'{args.seed}|{row["relative_path"]}|{name}'.encode()).hexdigest()[:8], 16)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                for device in range(4):
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                question = QUESTION.format(finding=name.lower())
                try:
                    response = str(model.ask_question(question, image,
                                                      temperature=.2, top_p=.8)).strip()
                    for device in range(4):
                        torch.cuda.synchronize(device)
                    if not response:
                        raise ValueError('Empty model response')
                except (OSError, RuntimeError, ValueError) as error:
                    record['status'] = 'failed'
                    record['error'] = repr(error)
                    save_json(path, record)
                    raise
                value = parse_binary(response)
                record['responses'][name] = {'text': response, 'parsed': value, 'seed': seed}
                if value is not None:
                    record['parsed_labels'][name] = value
                record['latency_sec'] += time.perf_counter() - start
                record['peak_vram_gb_max_device'] = max(record['peak_vram_gb_max_device'],
                    max(torch.cuda.max_memory_allocated(i) for i in range(4)) / 1024**3)
                record['completed_utc'] = datetime.now(timezone.utc).isoformat()
                save_json(path, record)
        record['status'] = 'success'
        save_json(path, record)
        counts = progress(output, rows)
        print(f'{row["index"] + 1}/{len(rows)} {record["status"]}, '
              f'parsed={counts["parsed_labels"]}/{counts["total_labels"]}', flush=True)
        if args.stop_after and counts['generated'] >= args.stop_after:
            return counts
    return progress(output, rows)


def f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else None


def evaluate(args, rows):
    import numpy as np

    output = args.output.resolve()
    items = records(output, rows)
    if len(items) != len(rows):
        raise ValueError('Full 518-image inference has not completed')
    if any(item['status'] != 'success' for item in items):
        raise ValueError('Generation failures remain')
    predictions = {item['index']: item for item in items}
    metrics = []
    rng = np.random.default_rng(42)
    bootstrap = rng.integers(0, len(rows), size=(1000, len(rows)))
    errors = []
    for name in LABELS:
        gt = np.array([row['labels'][name] for row in rows])
        pred = np.array([predictions[i]['parsed_labels'].get(name, -1)
                         for i in range(len(rows))])
        valid = pred >= 0
        tp = int(np.sum((gt == 1) & (pred == 1)))
        fp = int(np.sum((gt == 0) & (pred == 1)))
        fn = int(np.sum((gt == 1) & (pred == 0)))
        tn = int(np.sum((gt == 0) & (pred == 0)))
        bs = []
        for sample in bootstrap:
            value = f1(int(np.sum((gt[sample] == 1) & (pred[sample] == 1))),
                       int(np.sum((gt[sample] == 0) & (pred[sample] == 1))),
                       int(np.sum((gt[sample] == 1) & (pred[sample] == 0))))
            if value is not None:
                bs.append(value)
        metrics.append({'pathology': name, 'n': len(rows), 'covered': int(valid.sum()),
                        'gt_positive': int((gt == 1).sum()), 'gt_negative': int((gt == 0).sum()),
                        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                        'precision': tp / (tp + fp) if tp + fp else None,
                        'recall': tp / (tp + fn) if tp + fn else None,
                        'specificity': tn / (tn + fp) if tn + fp else None,
                        'accuracy': (tp + tn) / int(valid.sum()) if valid.any() else None,
                        'f1': f1(tp, fp, fn),
                        'ci95_low': float(np.quantile(bs, .025)) if bs else None,
                        'ci95_high': float(np.quantile(bs, .975)) if bs else None,
                        'paper_f1': PAPER_F1.get(name)})
        if name in PRIMARY:
            for i in range(len(rows)):
                if pred[i] in (0, 1) and gt[i] != pred[i]:
                    errors.append({'index': i, 'pathology': name,
                                   'error_type': 'FP' if pred[i] == 1 else 'FN',
                                   'study': rows[i]['study'],
                                   'relative_path': rows[i]['relative_path'],
                                   'gt': int(gt[i]), 'prediction': int(pred[i]),
                                   'model_response': predictions[i]['responses'][name]['text']})
    write_csv(output / 'metrics.csv', metrics, list(metrics[0]))
    write_csv(output / 'review_queue.csv', errors,
              ['index', 'pathology', 'error_type', 'study', 'relative_path',
               'gt', 'prediction', 'model_response'])
    by_name = {row['pathology']: row for row in metrics}
    defined = [by_name[name]['f1'] for name in PRIMARY[:5] if by_name[name]['f1'] is not None]
    summary = {'n_images': len(rows), 'n_studies': len({r['study'] for r in rows}),
               'parsed': sum(len(item['parsed_labels']) for item in items),
               'total_label_questions': len(rows) * len(LABELS),
               'macro_f1_five_defined': sum(defined) / len(defined) if defined else None,
               'five_pathologies_defined': len(defined), 'paper_f1_five': .57,
               'prompt_and_decoding': '14 local per-finding yes/no questions; exact paper prompt unavailable',
               'mean_latency_sec': sum(item['latency_sec'] for item in items) / len(items),
               'max_peak_vram_gb_single_device': max(item['peak_vram_gb_max_device'] for item in items),
               'bootstrap': '1000 image-level resamples, seed 42',
               'classification': 'matched-cohort CheXpert Table 3 approximation'}
    save_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return metrics, errors, summary


def examples(args, rows, errors):
    run_name = args.output.resolve().name
    folder = PROJECT / 'reports' / f'chexpert_table3_example_images_{run_name}'
    folder.mkdir(parents=True, exist_ok=False)
    selected = []
    seen = set()
    for error in errors:
        key = (error['pathology'], error['error_type'])
        if key in seen:
            continue
        seen.add(key)
        selected.append(error)
        if len(selected) >= 14:
            break
    for error in selected:
        source = Path(rows[error['index']]['image_path'])
        name = f"{error['index']:03d}_{error['pathology'].replace(' ', '_')}_{error['error_type']}.jpg"
        shutil.copy2(source, folder / name)
        error['example_file'] = name
    write_csv(folder / 'index.csv', selected,
              ['index', 'pathology', 'error_type', 'study', 'relative_path',
               'gt', 'prediction', 'model_response', 'example_file'])
    (folder / 'README.md').write_text(
        'Research-only model error examples. Labels are official study-level '
        'CheXpert test labels. Images and model outputs require expert review; '
        'these examples are not clinical interpretations.\n')
    archive = folder.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in folder.rglob('*'):
            if path.is_file():
                zf.write(path, path.relative_to(folder.parent))
    return folder, archive


def report(args, rows, metrics, errors, summary, example_dir, archive):
    path = PROJECT / 'reports' / f'{datetime.now().date()}_k80_chexpert_table3_{args.output.resolve().name}.md'
    if path.exists():
        raise FileExistsError(path)
    primary = [row for row in metrics if row['pathology'] in PRIMARY]
    table = ['| Pathology | GT +/− | TP/FP/TN/FN | Precision | Recall | F1 (95% CI) | Paper F1 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    def number(x):
        return f'{x:.3f}' if x is not None else 'undefined'
    for row in primary:
        table.append(f"| {row['pathology']} | {row['gt_positive']}/{row['gt_negative']} | "
                     f"{row['tp']}/{row['fp']}/{row['tn']}/{row['fn']} | "
                     f"{number(row['precision'])} | {number(row['recall'])} | "
                     f"{number(row['f1'])} ({number(row['ci95_low'])}, {number(row['ci95_high'])}) | "
                     f"{number(row['paper_f1'])} |")
    text = f'''# CheXpert Table 3 K80 evaluation — matched 518-image cohort

## 1. Experiment Information

- Experiment ID: `{args.output.resolve().name}`; date: {datetime.now().date()}; researcher: student4 / project team.
- Repository: https://github.com/hanhpm/CXR_LLaVA_Improvement; commit: `{subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}`.
- Script: `scripts/chexpert_table3_k80.py`; status: completed local evaluation, research only.

## 2. Objective

Evaluate the released CXR-LLaVA v2 checkpoint on the official CheXpert frontal test images with 14 direct yes/no questions per image.

## 3. Survey / Technical Motivation

Paper Table 3: https://arxiv.org/pdf/2310.18341v3. CheXlocalize test images: https://github.com/rajpurkarlab/cheXlocalize. Public consensus labels: https://github.com/rajpurkarlab/cheXpert-test-set-labels. The paper says it directly generated binary labels, but does not expose the exact prompt or decoding code; this evaluation freezes a local prompt in `run_config.json`.

## 4. Environment

Four Tesla K80 GPUs, FP32, torch 1.12.1+cu102, transformers 4.36.2, model revision `{MODEL_REVISION}`. Exact package freeze and source snapshot are under `{args.output.resolve()}/snapshots/`.

## 5. Dependency Setup

Reused `cxr-llava-k80`; no model weights were changed. K80 adaptation changes BF16 casts to FP32 and dispatches the model across four devices.

## 6. Code / Notebook Changes

Added a direct binary-question runner with per-question checkpointing, strict image/study label alignment, transparent parse coverage, bootstrap metrics and example-image archive.

## 7. Experiment Configuration

668 downloaded CheXpert test images comprise 518 frontal and 150 lateral. Only 518 frontal images were used. Their 14 labels agree with `groundtruth.csv` per study, and their positive/negative support matches paper supplementary Table 3 distribution. Fourteen locally chosen yes/no questions per image, helper defaults temperature 0.2/top_p 0.8, per-image-and-label SHA256-derived seed, 1000 image-level bootstrap replicates.

## 8. Code Executed

`python -m scripts.chexpert_table3_k80 --run --output {args.output.resolve()}` then `python -m scripts.chexpert_table3_k80 --evaluate --output {args.output.resolve()}`.

## 9. Output

Raw responses/checkpoints: `{args.output.resolve()}/cases/`; metrics: `metrics.csv`; parse/error review: `review_queue.csv`. Example images: `{example_dir}`; ZIP: `{archive}`.

## 10. Output Analysis

Completed {summary['n_images']}/{summary['n_images']} images; {summary['parsed']}/{summary['total_label_questions']} explicit yes/no answers parsed. Unparsed answers are missing predictions, never converted to negatives. {len(errors)} parsed disagreements across seven shown pathologies; inspect review queue and images before clinical interpretation.

## 11. Quantitative Results

Macro F1 over {summary['five_pathologies_defined']}/5 defined paper-primary findings: {number(summary['macro_f1_five_defined'])}; paper reported 0.57. Mean latency: {number(summary['mean_latency_sec'])} seconds/image; peak allocated VRAM on a single device: {number(summary['max_peak_vram_gb_single_device'])} GiB.

{chr(10).join(table)}

All 14 findings, accuracy/specificity, support, coverage and bootstrap CI are in `metrics.csv`. The paper's average-F1 aggregation is not independently verified.

## 12. Qualitative Results

`{example_dir.name}/index.csv` pairs selected FP/FN cases with original images and model text. This is an error-review bundle, not radiologist adjudication.

## 13. Comparison with Baseline

The paper used 518 CheXpert images and reported F1 of 0.57 over five principal pathologies. Local cohort identity and GT support match. Prompt, decoding, dtype and exact metric aggregation are not verified as identical, so F1 differences are descriptive.

## 14. Problems Encountered

AzCopy created an extra `test/` directory in the local target; path resolution accounted for this without altering source files. Exact Table 3 prompt and decoding source were unavailable.

## 15. Conclusion

The released model was evaluated on the same-count, matching-label 518 frontal CheXpert test images, with direct binary-question responses and explicit parse coverage. This is a matched-cohort local Table 3 approximation, not an exact paper reproduction.

## 16. Next Experiment

- [ ] Review FP/FN examples with a qualified radiologist before making image-level claims.
- [ ] Obtain the authors' exact prompt/decoding and aggregation code if available.
- [ ] Separately review Indiana FP/FN/omissions and the eight NLM view ambiguities; do not alter its frozen cohort.

## 17. Reproducibility Checklist

- [x] 518 frontal image paths, study labels, source and model hashes, seed, prompt and raw outputs captured.
- [x] Counts and GT support match paper supplementary distribution; metrics and example ZIP saved.
- [ ] Exact Table 3 generation prompt/decoding and independent radiologist review verified.

## 18. Short Lab Update

> Matched the 518-image CheXpert frontal cohort, generated direct yes/no predictions for 14 findings on four K80s, measured {summary['parsed']}/{summary['total_label_questions']} parse coverage and five-finding macro F1 {number(summary['macro_f1_five_defined'])}; exact paper prompt remains unresolved.
'''
    path.write_text(text)
    print(f'Report: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--run', action='store_true')
    group.add_argument('--evaluate', action='store_true')
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--model-dir', type=Path, default=PROJECT / 'model_cache/cxr-llava-v2-original')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if (args.run or args.evaluate) and args.output is None:
        parser.error('--run/--evaluate require --output')
    if args.stop_after is not None and (not args.run or args.stop_after < 1):
        parser.error('--stop-after requires --run and positive count')
    rows, source = cohort(args.dataset.resolve())
    print(f"Cohort: {len(rows)} frontal images, 500 studies; GT/paper support PASS", flush=True)
    if args.check:
        return
    if args.run:
        counts = infer(args, rows, source)
        print(json.dumps(counts), flush=True)
        return
    metrics, errors, summary = evaluate(args, rows)
    example_dir, archive = examples(args, rows, errors)
    report(args, rows, metrics, errors, summary, example_dir, archive)


if __name__ == '__main__':
    main()
