"""Manual Linux smoke/subset runner. Default: inspect inputs, never load a model."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from scripts.exp02_provenance import MODEL_REVISION


PROJECT = Path(__file__).resolve().parents[1]
MODEL_ID = "ECOFRI/CXR-LLAVA-v2"
# Legacy Llama 2 formatting, with actual newlines in the rendered SYS block.
CHAT_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "{{ raise_exception('An explicit system message is required.') }}{% endif %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% for message in messages[1:] %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Roles must alternate user/assistant.') }}{% endif %}"
    "{% set content = message['content'] %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + content %}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, columns, header=True):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore',
                                quoting=csv.QUOTE_ALL)
        if header:
            writer.writeheader()
        writer.writerows(rows)


def inputs(args):
    if args.cohort:
        with args.cohort.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            required = {'report_id', 'image_id', 'image_path', 'ground_truth_report'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError('Cohort needs: ' + ', '.join(sorted(required)))
            rows = list(reader)
        if not rows or len(rows) > args.max_cases:
            raise ValueError('Use a fixed nonempty cohort within --max-cases; no automatic sampling.')
        for key in ('image_id', 'report_id'):
            values = [row[key] for row in rows]
            if any(not value.strip() for value in values) or len(set(values)) != len(values):
                raise ValueError(f'{key} must be nonempty and unique: one image per report.')
        if any(not row['ground_truth_report'].strip() for row in rows):
            raise ValueError('Ground-truth reports must be nonempty.')
        if args.execute:
            if not args.review_gate:
                raise ValueError('Cohort execution requires --review-gate with actual GT/view review evidence.')
            gate = json.loads(args.review_gate.read_text(encoding='utf-8'))
            if (gate.get('cohort_sha256') != sha256(args.cohort)
                    or gate.get('gt_review_pass') is not True
                    or gate.get('view_review_pass') is not True):
                raise ValueError('Cohort hash or GT/view review gate failed.')
    else:
        rows = [dict(report_id='sample', image_id='sample', image_path=str(args.image),
                     ground_truth_report='')]
    for row in rows:
        if not row['image_path'].strip():
            raise ValueError('Empty image path')
        path = Path(row['image_path']).expanduser()
        if not path.is_absolute():
            path = args.image_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        row['image_path'] = str(path)
        row['image_sha256'] = sha256(path)
    if args.output and args.output.exists():
        raise FileExistsError('Choose a new output directory; existing runs are preserved.')
    return rows


def execute(args, rows):
    # Heavy imports are deliberately limited to the explicitly requested execution path.
    import torch
    from PIL import Image
    from scripts.exp02_provenance import load_pinned_model, report_prompt_audit

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for inference.')
    for row in rows:
        with Image.open(row['image_path']) as image:
            if image.mode not in ('L', 'RGB', 'RGBA'):
                raise ValueError('Use reviewed 8-bit L/RGB/RGBA images; no automatic bit-depth conversion.')
            image.verify()
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('subsets', 'predictions', 'chexpert_inputs', 'chexpert_labels', 'logs'):
        (args.output / name).mkdir()
    prefix = f'indiana_f1_subset_{len(rows)}'
    write_csv(args.output / 'subsets' / f'{prefix}_seed42.csv', rows,
              ['report_id', 'image_id', 'image_path', 'ground_truth_report', 'image_sha256'])
    commit = subprocess.run(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True).stdout.strip()
    checkout_status = subprocess.run(['git', '-C', str(PROJECT), 'status', '--porcelain'],
                                    capture_output=True, text=True, check=True).stdout
    config = dict(model_id=MODEL_ID, revision=MODEL_REVISION, seed=42,
                  temperature=0, top_p=1, quantization='4-bit NF4, double quantization, FP16 compute',
                  classification='subset evaluation pending labeling' if args.cohort else 'smoke test',
                  cases=len(rows), git_commit=commit, git_status=checkout_status,
                  runner_sha256=sha256(Path(__file__)),
                  loader_sha256=sha256(PROJECT / 'scripts' / 'exp02_provenance.py'),
                  python=platform.python_version(),
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  cohort_sha256=sha256(args.cohort) if args.cohort else None,
                  packages={name: version(name) for name in
                            ('transformers', 'tokenizers', 'huggingface-hub', 'accelerate',
                             'bitsandbytes', 'sentencepiece', 'protobuf', 'pillow', 'numpy', 'pandas')})
    (args.output / 'logs' / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    model = load_pinned_model(MODEL_ID)
    if not model.tokenizer.chat_template:
        model.tokenizer.chat_template = CHAT_TEMPLATE
    report_prompt_audit(model, args.output / 'logs' / 'prompt_audit')
    columns = ['report_id', 'image_id', 'image_path', 'ground_truth_report', 'generated_report',
               'latency_sec', 'peak_vram_gb', 'status', 'error_message']
    prediction_path = args.output / 'predictions' / f'{prefix}_predictions.csv'
    completed = []
    for index, row in enumerate(rows, 1):
        record = dict(row, generated_report='', status='failed', error_message='',
                      latency_sec=None, peak_vram_gb=None)
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with Image.open(row['image_path']) as image, torch.inference_mode():
                report = model.write_radiologic_report(image, temperature=0, top_p=1)
            torch.cuda.synchronize()
            record.update(generated_report=str(report).strip(), latency_sec=time.perf_counter()-start,
                          peak_vram_gb=torch.cuda.max_memory_allocated()/1024**3)
            if not record['generated_report']:
                raise ValueError('Empty model-generated report')
            record['status'] = 'success'
        except (OSError, RuntimeError, ValueError) as error:
            record['error_message'] = repr(error)
            completed.append(record)
            write_csv(prediction_path, completed, columns)
            raise
        completed.append(record)
        write_csv(prediction_path, completed, columns)
        print(f'{index}/{len(rows)} saved (research-only model output)', flush=True)
    if args.cohort:
        folder = args.output / 'chexpert_inputs'
        for kind, column in [('gt', 'ground_truth_report'), ('generated', 'generated_report')]:
            write_csv(folder / f'{prefix}_{kind}_reports.csv', completed, [column], header=False)
        row_map = [dict(row, chexpert_row=i) for i, row in enumerate(completed)]
        write_csv(folder / f'{prefix}_row_map.csv', row_map, ['chexpert_row', 'report_id', 'image_id'])
    print(f'Saved {args.output}. No labeler or metric evaluation launched.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=PROJECT / 'IMG' / 'img.jpg')
    parser.add_argument('--cohort', type=Path, help='Fixed reviewed Indiana CSV; overrides --image')
    parser.add_argument('--review-gate', type=Path, help='Actual GT/view review gate for this exact cohort hash')
    parser.add_argument('--image-root', type=Path, default=PROJECT)
    parser.add_argument('--max-cases', type=int, default=10, help='Explicit cohort size ceiling')
    parser.add_argument('--output', type=Path, help='New run directory, required for execution')
    parser.add_argument('--execute', action='store_true', help='Download/load model and run inference')
    args = parser.parse_args()
    if args.execute and not args.output:
        parser.error('--execute requires --output')
    rows = inputs(args)
    print(f'Input preflight passed: {len(rows)} file(s), model revision {MODEL_REVISION}.')
    if args.execute:
        execute(args, rows)
    else:
        print('Preflight only: no model imports/downloads, inference, or output writes. Image decoding is checked on execution.')


if __name__ == '__main__':
    main()
