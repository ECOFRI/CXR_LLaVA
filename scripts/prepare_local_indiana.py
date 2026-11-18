"""Create a reproducible small image/report cohort from local OpenI XML/PNG files."""

import argparse
import csv
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from scripts.manual_experiment import sha256, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--all', action='store_true', help='All eligible reports, not the official paper split')
    args = parser.parse_args()
    if args.size < 1:
        parser.error('--size must be positive')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Choose a new cohort output; existing cohorts are preserved')
    root = args.dataset_root.resolve()
    images = root / 'NLMCXR_png'
    reports = root / 'NLMCXR_reports/ecgen-radiology'
    xmls = sorted(reports.glob('*.xml'))
    if not xmls or not images.is_dir():
        raise FileNotFoundError('Expected NLMCXR_png and NLMCXR_reports/ecgen-radiology')
    eligible = []
    for xml in xmls:
        report = ET.parse(xml).getroot()
        uid = report.find('uId')
        if uid is None or not uid.get('id'):
            continue
        sections = {e.get('Label', '').upper(): (e.text or '').strip()
                    for e in report.findall('.//AbstractText')}
        text = ' '.join(sections.get(key, '') for key in ('FINDINGS', 'IMPRESSION')).strip()
        if not text:
            continue
        # First available image by XML order; this is not a verified view label.
        for element in report.findall('.//parentImage'):
            image_id = element.get('id', '')
            path = images / f'{image_id}.png'
            if image_id and path.is_file():
                eligible.append(dict(report_id=uid.get('id'), image_id=image_id,
                    image_path=str(path), ground_truth_report=text,
                    report_xml_path=str(xml), caption=element.findtext('caption', ''),
                    report_text_source='findings+impression',
                    view_selection='first_existing_image_in_xml; view_not_verified'))
                break
    if len(eligible) < args.size:
        raise ValueError(f'Only {len(eligible)} eligible report/image pairs')
    if len({r['report_id'] for r in eligible}) != len(eligible):
        raise ValueError('Duplicate report IDs in XML collection')
    selected = eligible if args.all else random.Random(args.seed).sample(eligible, args.size)
    if len({r['image_id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate selected image IDs')
    for row in selected:
        path = Path(row['image_path'])
        with Image.open(path) as image:
            if image.mode not in ('L', 'RGB', 'RGBA'):
                raise ValueError(f'Unsupported 8-bit image mode: {image.mode}')
            row['image_mode'] = image.mode
            row['image_size'] = f'{image.width}x{image.height}'
            image.verify()
        row['image_sha256'] = sha256(path)
        row['report_xml_sha256'] = sha256(Path(row['report_xml_path']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, selected, list(selected[0]))
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        dataset='indiana', dataset_root=str(root), xml_reports=len(xmls),
        eligible_report_image_pairs=len(eligible), size=len(selected), seed=args.seed,
        cohort_sha256=sha256(args.output), reference='original FINDINGS + IMPRESSION',
        selection=('all eligible reports' if args.all else 'random reports') + '; first existing image in XML; not stratified',
        gt_review_pass=False, view_review_pass=False,
        limitation='Views are not verified; subset metrics are not paper benchmark reproduction.'
    ), indent=2))
    print(f'Saved {len(selected)} decoded image/report pairs: {args.output.resolve()}')
    print(f'Eligible pairs: {len(eligible)}. Image views require review before benchmark interpretation.')


if __name__ == '__main__':
    main()
