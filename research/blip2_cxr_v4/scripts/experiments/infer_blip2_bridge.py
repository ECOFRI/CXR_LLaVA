"""Offline inference entrypoint for locally trained REIMPLEMENTED BLIP-2 arms."""
import argparse
import csv
import json
import sys
from pathlib import Path

VR = Path(__file__).resolve().parents[2]
ROOT = VR.parents[1]
sys.path.insert(0, str(VR / 'src'))
sys.path.insert(0, str(VR / 'scripts' / 'train'))

from train_blip2_bridges import dry_run, load_config
from vlm_medical.data.manifests import eligible_records
from vlm_medical.data.preprocess import load_image
from vlm_medical.models.bridges import FP32Projection, FP32QFormer, inject_opt_lora
from vlm_medical.train.common import log_status


def test_cases(config, cohort, limit):
    if cohort == 'mimic_test':
        data = json.loads((ROOT / config['mimic_config']).read_text())
        for row in eligible_records(data, 'test', limit=limit):
            yield row['image']
    elif cohort == 'indiana_x00':
        with (ROOT / config['indiana_x00_csv']).open(newline='', encoding='utf-8-sig') as handle:
            selected = 0
            for row in csv.DictReader(handle):
                if row['view_label'].startswith('frontal'):
                    yield Path(row['image_path'])
                    selected += 1
                    if selected >= limit:
                        break
    else:
        raise ValueError('Unknown cohort')


def run(config, args):
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    checkpoint = ROOT / config['checkpoint_dir']
    processor = Blip2Processor.from_pretrained(checkpoint, local_files_only=True, use_fast=False)
    model = Blip2ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True, torch_dtype=torch.float16)
    model.language_projection = FP32Projection(model.language_projection)
    if config['arm'] == 'qformer_projection':
        model.qformer = FP32QFormer(model.qformer)
        model.query_tokens = torch.nn.Parameter(model.query_tokens.float())
    elif config['arm'] == 'projection_opt_lora':
        inject_opt_lora(model.language_model)
    elif config['arm'] != 'projection_only':
        raise ValueError('Feature adapter unavailable')
    state = torch.load(args.weights, map_location='cpu')
    if state['config']['experiment_id'] != config['experiment_id']:
        raise ValueError('Trained arm checkpoint/config mismatch')
    model.load_state_dict(state['model'], strict=False)
    model.to('cuda:0').eval()
    prompt = 'Question: What abnormalities are visible in this chest x-ray? Answer:'
    output = args.output_dir / 'predictions.jsonl'
    with output.open('w') as handle, torch.no_grad():
        count = 0
        for image_path in test_cases(config, args.cohort, args.max_cases):
            inputs = processor(images=load_image(image_path), text=prompt, return_tensors='pt')
            inputs = {key: value.to('cuda:0', dtype=torch.float16) if key == 'pixel_values' else value.to('cuda:0') for key, value in inputs.items()}
            generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            if not text:
                raise RuntimeError('Empty generated report')
            handle.write(json.dumps({'case_index': count, 'generated_report': text, 'classification': 'REIMPLEMENTED_NOT_REPRODUCED'}) + '\n')
            count += 1
    log_status(args.output_dir / 'status.json', status='PASS', cases=count, cohort=args.cohort, classification='REIMPLEMENTED_NOT_REPRODUCED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cohort', choices=('mimic_test', 'indiana_x00'), default='mimic_test')
    parser.add_argument('--max-cases', type=int, default=10)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--infer', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if not args.output_dir.resolve().is_relative_to(VR.resolve()):
        raise ValueError('Output must be under research/blip2_cxr_v4')
    if args.dry_run:
        result = dry_run(config, args.output_dir)
        result['inference_status'] = 'NOT_RUN'
        print(json.dumps(result, separators=(',', ':')))
        return
    if config['encoder'] != 'blip2' or not args.weights or not args.weights.is_file():
        raise FileNotFoundError('Trained local bridge checkpoint required; external encoder adapter unsupported')
    run(config, args)


if __name__ == '__main__':
    main()
