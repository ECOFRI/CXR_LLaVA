"""Aggregate, read-only MIMIC JPG/report linkage audit."""

import argparse
import csv
import gzip
import json
import random
import re
from collections import Counter
from pathlib import Path

VR = Path(__file__).resolve().parents[2]
ROOT = VR.parents[1]


def read_csv(path):
    if path.suffix == '.gz':
        return gzip.open(path, 'rt', newline='', encoding='utf-8-sig')
    return path.open(newline='', encoding='utf-8-sig')


def schema_count(path):
    with read_csv(path) as stream:
        rows = csv.reader(stream)
        columns = next(rows)
        return {'rows': sum(1 for _ in rows), 'columns': columns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=VR / 'configs/eval/mimic_p1.yaml')
    parser.add_argument('--output-dir', type=Path, default=VR / 'manifests')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    jpg = Path(config['jpg_root'])
    jpg_alt = Path(config['jpg_alt_files'])
    reports = Path(config['report_files'])
    testset = Path(config['repo_testset'])
    if not testset.is_absolute():
        testset = ROOT / testset
    metadata = jpg / 'mimic-cxr-2.0.0-metadata.csv.gz'
    split = jpg / 'mimic-cxr-2.0.0-split.csv.gz'
    label_files = [jpg / 'mimic-cxr-2.0.0-chexpert.csv.gz',
                   jpg / 'mimic-cxr-2.0.0-negbio.csv.gz',
                   jpg / 'mimic-cxr-2.1.0-test-set-labeled.csv']
    for path in [metadata, split, *label_files, testset]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.dry_run:
        print(json.dumps({'status': 'DRY_RUN_PASS', 'sample_per_prefix': config['sample_per_prefix']}))
        return
    summary_path = args.output_dir / 'mimic_p1_summary.json'
    sample_path = args.output_dir / 'mimic_sample_linkage.csv'
    if summary_path.exists() or sample_path.exists():
        if args.resume and summary_path.exists() and sample_path.exists():
            print(json.dumps({'status': 'COMPLETE', 'resumed': True}))
            return
        raise FileExistsError('MIMIC audit outputs already exist')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rngs = {f'p{i}': random.Random(config['seed'] + i) for i in range(10, 20)}
    samples = {prefix: [] for prefix in rngs}
    prefix_studies = Counter()
    subjects, studies, seen_studies = set(), set(), set()
    metadata_rows = 0
    with read_csv(metadata) as stream:
        reader = csv.DictReader(stream)
        metadata_columns = reader.fieldnames
        for row in reader:
            metadata_rows += 1
            subject, study, dicom = row['subject_id'], row['study_id'], row['dicom_id']
            subjects.add(subject)
            studies.add(study)
            key = (subject, study)
            if key in seen_studies:
                continue
            seen_studies.add(key)
            prefix = 'p' + subject[:2]
            if prefix not in samples:
                continue
            prefix_studies[prefix] += 1
            chosen = samples[prefix]
            n = prefix_studies[prefix]
            entry = (subject, study, dicom)
            if len(chosen) < config['sample_per_prefix']:
                chosen.append(entry)
            else:
                index = rngs[prefix].randrange(n)
                if index < len(chosen):
                    chosen[index] = entry
    repo_dicom = set()
    with testset.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.reader(stream):
            if row:
                repo_dicom.add(Path(row[0].replace('\\', '/')).stem)
    split_counts = Counter()
    repo_split_counts = Counter()
    with read_csv(split) as stream:
        split_reader = csv.DictReader(stream)
        split_columns = split_reader.fieldnames
        for row in split_reader:
            split_counts[row['split']] += 1
            if row['dicom_id'] in repo_dicom:
                repo_split_counts[row['split']] += 1
    by_prefix = []
    section_counts = Counter()
    total_sample = total_report = total_jpg = total_jpg_primary = total_jpg_alt = 0
    for prefix in sorted(samples):
        found_report = found_jpg = found_primary = found_alt = 0
        for subject, study, dicom in samples[prefix]:
            report_path = reports / prefix / f'p{subject}' / f's{study}.txt'
            jpg_path = jpg / 'files' / prefix / f'p{subject}' / f's{study}' / f'{dicom}.jpg'
            alt_path = jpg_alt / prefix / f'p{subject}' / f's{study}' / f'{dicom}.jpg'
            if report_path.is_file():
                found_report += 1
                content = report_path.read_text(errors='replace')
                has_findings = bool(re.search(r'(?im)^\s*FINDINGS\s*:', content))
                has_impression = bool(re.search(r'(?im)^\s*IMPRESSION\s*:', content))
                section_counts['findings' if has_findings else 'no_findings'] += 1
                section_counts['impression' if has_impression else 'no_impression'] += 1
                if has_findings and has_impression:
                    section_counts['both'] += 1
            primary_exists = jpg_path.is_file()
            alt_exists = alt_path.is_file()
            found_primary += primary_exists
            found_alt += alt_exists
            if primary_exists or alt_exists:
                found_jpg += 1
        n = len(samples[prefix])
        total_sample += n
        total_report += found_report
        total_jpg += found_jpg
        total_jpg_primary += found_primary
        total_jpg_alt += found_alt
        by_prefix.append({'prefix': prefix, 'population_studies': prefix_studies[prefix],
                          'sample_studies': n, 'report_found': found_report, 'jpg_found': found_jpg,
                          'jpg_found_primary': found_primary, 'jpg_found_alt': found_alt,
                          'report_relative_template': 'pXX/p{subject_id}/s{study_id}.txt',
                          'jpg_relative_template': 'files/pXX/p{subject_id}/s{study_id}/{dicom_id}.jpg'})
    weighted = sum(row['population_studies'] * row['report_found'] / row['sample_studies']
                   for row in by_prefix if row['sample_studies']) / sum(prefix_studies.values())
    file_schemas = {path.name: schema_count(path) for path in label_files}
    summary = {'status': 'COMPLETE', 'method': f'deterministic reservoir sample of unique studies, {config["sample_per_prefix"]} per subject prefix; exists checks only',
               'metadata': {'path': str(metadata), 'rows': metadata_rows, 'columns': metadata_columns,
                            'unique_subjects': len(subjects), 'unique_studies': len(studies)},
               'split': {'path': str(split), 'columns': split_columns, 'counts': dict(split_counts)},
               'labels': file_schemas, 'repo_testset': {'path': str(testset), 'unique_dicom_stems': len(repo_dicom),
                                                        'split_overlap': dict(repo_split_counts),
                                                        'unmatched': len(repo_dicom) - sum(repo_split_counts.values())},
               'sample': {'n': total_sample, 'report_found': total_report, 'jpg_found': total_jpg,
                          'jpg_found_primary': total_jpg_primary, 'jpg_found_alt': total_jpg_alt,
                          'report_coverage_unweighted': total_report / total_sample,
                          'report_coverage_prefix_weighted': weighted,
                          'section_counts': dict(section_counts)},
               'report_version': 'UNKNOWN; credentialed license file observed',
               'jpg_version': 'README says 2.0.0; directory 2.1.0; metadata filenames 2.0.0'}
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    with sample_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, by_prefix[0])
        writer.writeheader()
        writer.writerows(by_prefix)
    print(json.dumps({'status': 'COMPLETE', 'metadata_rows': metadata_rows, 'sample_n': total_sample,
                      'report_found': total_report, 'jpg_found': total_jpg,
                      'weighted_coverage': round(weighted, 4)}))


if __name__ == '__main__':
    main()
