"""Official MIMIC split reader; paths are checked without copying clinical text."""
import csv
import gzip
import re
from pathlib import Path


def rows(path):
    with gzip.open(path, 'rt', newline='') as handle:
        yield from csv.DictReader(handle)


def report_sections(text):
    hits = list(re.finditer(r'(?im)^\s*(FINDINGS|IMPRESSION)\s*:', text))
    sections = {}
    for ix, match in enumerate(hits):
        end = hits[ix + 1].start() if ix + 1 < len(hits) else len(text)
        value = text[match.end():end].strip()
        if value:
            sections[match.group(1).lower()] = value
    return sections


def image_path(cfg, subject, study, dicom):
    relative = Path('p' + subject[:2]) / ('p' + subject) / ('s' + study) / (dicom + '.jpg')
    for base in (Path(cfg['jpg_root']) / 'files', Path(cfg['jpg_alt_files'])):
        candidate = base / relative
        if candidate.is_file():
            return candidate
    return None


def eligible_records(cfg, split, limit=None):
    root = Path(cfg['jpg_root'])
    metadata = {r['dicom_id']: r for r in rows(root / 'mimic-cxr-2.0.0-metadata.csv.gz')}
    seen = set()
    emitted = 0
    for row in rows(root / 'mimic-cxr-2.0.0-split.csv.gz'):
        if row['split'] != split:
            continue
        subject, study, dicom = row['subject_id'], row['study_id'], row['dicom_id']
        meta = metadata.get(dicom)
        if not meta or meta['subject_id'] != subject or meta['study_id'] != study:
            continue
        if (meta.get('ViewPosition') or '').upper() not in ('AP', 'PA'):
            continue
        key = (subject, study)
        if key in seen:
            continue
        image = image_path(cfg, subject, study, dicom)
        report = Path(cfg['report_files']) / ('p' + subject[:2]) / ('p' + subject) / ('s' + study + '.txt')
        if image is None or not report.is_file():
            continue
        sections = report_sections(report.read_text(errors='replace'))
        if not sections:
            continue
        seen.add(key)
        yield {'subject_id': subject, 'study_id': study, 'image_id': dicom,
               'view_position': meta['ViewPosition'].upper(),
               'image': image, 'report': report,
               'text': '\n'.join(sections[k] for k in ('findings', 'impression') if k in sections)}
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def check_cohorts(cfg, x00_csv):
    result = {}
    for split in ('train', 'validate', 'test'):
        result[split] = sum(1 for _ in eligible_records(cfg, split, limit=1))
        if not result[split]:
            raise FileNotFoundError('No eligible local MIMIC record in ' + split)
    if not Path(x00_csv).is_file():
        raise FileNotFoundError('Indiana X00 cohort missing')
    result['indiana_external'] = True
    return result
