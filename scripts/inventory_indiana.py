"""Build an auditable all-image inventory and a human view-review package for OpenI."""

import argparse
import csv
import hashlib
import html
import os
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


IMAGE_FIELDS = [
    'report_id', 'xml_path', 'xml_sha256', 'figure_id', 'image_id', 'image_path',
    'image_exists', 'image_decodes', 'image_mode', 'image_width', 'image_height',
    'image_sha256', 'caption', 'findings', 'impression', 'ground_truth_report',
    'reference_policy', 'view_label', 'view_evidence', 'reviewer', 'reviewed_at',
    'selection_status', 'exclusion_reason',
]
REPORT_FIELDS = [
    'report_id', 'xml_path', 'xml_sha256', 'image_count_xml', 'image_count_exists',
    'findings', 'impression', 'ground_truth_report', 'reference_policy',
    'reference_valid', 'report_issue',
]
ANNOTATION_FIELDS = [
    'report_id', 'image_id', 'figure_id', 'image_path', 'caption',
    'view_label', 'view_evidence', 'reviewer', 'reviewed_at',
    'selection_status', 'pairing_status', 'review_notes',
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def report_sections(root):
    sections = defaultdict(list)
    for element in root.findall('.//AbstractText'):
        label = (element.get('Label') or '').strip().upper()
        text = ' '.join(''.join(element.itertext()).split())
        if label and text:
            sections[label].append(text)
    return {key: ' '.join(values) for key, values in sections.items()}


def relative_web_path(path, output_root):
    return Path(os.path.relpath(path, output_root)).as_posix()


def make_review_html(rows, output_root, dataset_root, page_size=80):
    image_rows = [row for row in rows if row['image_exists'] and row['image_decodes']]
    pages = math.ceil(len(image_rows) / page_size)
    payload = []
    for row in image_rows:
        payload.append({
            key: row.get(key, '') for key in
            ('report_id', 'image_id', 'figure_id', 'caption', 'findings', 'impression')
        } | {'image_src': relative_web_path(Path(row['image_path']), output_root / 'review')})
    data = json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')
    title = 'Indiana/OpenI image view and pairing review'
    page = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)}</title>
<style>
body{{font:16px system-ui,sans-serif;margin:1rem auto;max-width:1100px;padding:0 1rem;color:#18212b}}
nav{{position:sticky;top:0;background:#fff;padding:.8rem;border-bottom:1px solid #ccd;z-index:2}}
.card{{display:grid;grid-template-columns:minmax(300px,45%) 1fr;gap:1rem;padding:1rem 0;border-bottom:1px solid #ddd}}
img{{width:100%;height:auto;max-height:560px;object-fit:contain;background:#111}}
label{{display:block;margin:.45rem 0}} textarea{{width:100%;min-height:4rem}} .muted{{color:#59636e}}
@media(max-width:700px){{.card{{display:block}}}}
</style>
<nav><b>{html.escape(title)}</b> — <span id="count"></span><br>
Page <select id="page"></select> <button id="prev">Previous</button> <button id="next">Next</button>
<button id="export">Download annotations CSV</button> <button id="save">Save draft JSON</button>
<label>Load saved draft <input id="load" type="file" accept="application/json"></label>
<small>Classify each image independently. “Unknown” is a valid result. Select at most one eligible frontal per report only after reviewing all its images and the report text.</small></nav>
<main id="cards"></main>
<script>
const rows={data}; const pageSize={page_size}; const state=new Map();
const pageSelect=document.querySelector('#page');
for(let i=0;i<{pages};i++){{let o=document.createElement('option');o.value=i;o.textContent=`${{i+1}} / {pages}`;pageSelect.append(o)}}
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function key(r){{return r.report_id+'|'+r.image_id}}
function get(r){{return state.get(key(r))||{{view_label:'',view_evidence:'',reviewer:'',reviewed_at:'',selection_status:'',pairing_status:'',review_notes:''}}}}
function update(r,k,v){{const a={{...get(r),[k]:v}};state.set(key(r),a)}}
function options(values,current){{return '<option value="">-- choose --</option>'+values.map(v=>`<option ${{v===current?'selected':''}}>${{esc(v)}}</option>`).join('')}}
function render(){{const p=Number(pageSelect.value||0), start=p*pageSize, part=rows.slice(start,start+pageSize);document.querySelector('#count').textContent=`${{rows.length}} decodable images; showing ${{start+1}}–${{start+part.length}}`;
document.querySelector('#cards').innerHTML=part.map((r,j)=>{{const a=get(r), idx=start+j;return `<section class="card"><div><img loading="lazy" src="${{esc(r.image_src)}}"><p><b>Image:</b> ${{esc(r.image_id)}} · <b>report:</b> ${{esc(r.report_id)}} · ${{esc(r.figure_id)}}</p><p><b>Caption:</b> ${{esc(r.caption)}}</p></div><div><p><b>Reference text (findings + impression):</b></p><p>${{esc(r.findings)}} ${{esc(r.impression)}}</p>
<label>View <select data-i="${{idx}}" data-k="view_label">${{options(['frontal_PA','frontal_AP','frontal_unspecified','lateral','other','unknown'],a.view_label)}}</select></label>
<label>Image-specific view evidence <textarea data-i="${{idx}}" data-k="view_evidence">${{esc(a.view_evidence)}}</textarea></label>
<label>Image/report pairing <select data-i="${{idx}}" data-k="pairing_status">${{options(['matches','mismatch','uncertain'],a.pairing_status)}}</select></label>
<label>Selection <select data-i="${{idx}}" data-k="selection_status">${{options(['selected','not_selected','pending'],a.selection_status)}}</select></label>
<label>Reviewer <input data-i="${{idx}}" data-k="reviewer" value="${{esc(a.reviewer)}}"></label>
<label>Reviewed at (ISO date/time) <input data-i="${{idx}}" data-k="reviewed_at" value="${{esc(a.reviewed_at)}}"></label>
<label>Review notes <textarea data-i="${{idx}}" data-k="review_notes">${{esc(a.review_notes)}}</textarea></label></div></section>`}}).join('');
document.querySelectorAll('[data-i]').forEach(el=>el.addEventListener('change',()=>update(rows[Number(el.dataset.i)],el.dataset.k,el.value)));
}}
function csvCell(v){{return '"'+String(v??'').replaceAll('"','""')+'"'}}
document.querySelector('#export').onclick=()=>{{const fields={json.dumps(ANNOTATION_FIELDS)};const out=[fields.join(',')];for(const r of rows){{const a=get(r);out.push(fields.map(f=>csvCell(f in a?a[f]:r[f])).join(','))}}const blob=new Blob(['\\ufeff'+out.join('\\r\\n')],{{type:'text/csv;charset=utf-8'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='view_annotations.csv';a.click();URL.revokeObjectURL(a.href)}};
document.querySelector('#save').onclick=()=>{{const blob=new Blob([JSON.stringify(Object.fromEntries(state),null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='view_review_draft.json';a.click();URL.revokeObjectURL(a.href)}};
document.querySelector('#load').onchange=async e=>{{const d=JSON.parse(await e.target.files[0].text());for(const [k,v] of Object.entries(d))state.set(k,v);render()}};
pageSelect.onchange=render;document.querySelector('#prev').onclick=()=>{{pageSelect.value=Math.max(0,Number(pageSelect.value)-1);render()}};document.querySelector('#next').onclick=()=>{{pageSelect.value=Math.min({max(0,pages-1)},Number(pageSelect.value)+1);render()}};render();
</script></html>'''
    target = output_root / 'review' / 'view_review.html'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page, encoding='utf-8')
    return len(image_rows), target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    xml_root = root / 'NLMCXR_reports' / 'ecgen-radiology'
    image_root = root / 'NLMCXR_png'
    xml_paths = sorted(xml_root.glob('*.xml'))
    if not xml_paths or not image_root.is_dir():
        raise FileNotFoundError('Expected NLMCXR_reports/ecgen-radiology and NLMCXR_png')
    output = args.output.resolve()
    inventory_dir = output / 'inventory'
    review_dir = output / 'review'
    inventory_dir.mkdir(parents=True, exist_ok=False)
    review_dir.mkdir(parents=True, exist_ok=True)

    image_rows, report_rows, issues = [], [], []
    ids_by_hash = defaultdict(list)
    duplicate_reports = Counter()
    for n, xml_path in enumerate(xml_paths, 1):
        rel_xml = xml_path.relative_to(root).as_posix()
        xml_hash = sha256(xml_path)
        try:
            xml = ET.parse(xml_path).getroot()
        except ET.ParseError as error:
            issues.append(dict(report_id='', xml_path=str(xml_path), issue='xml_parse_error', details=str(error)))
            continue
        uid = xml.find('uId')
        report_id = uid.get('id', '').strip() if uid is not None else ''
        duplicate_reports[report_id] += 1
        sections = report_sections(xml)
        findings, impression = sections.get('FINDINGS', ''), sections.get('IMPRESSION', '')
        pieces = [value for value in (findings, impression) if value]
        reference = ' '.join(pieces)
        if findings and impression:
            reference_policy = 'findings+impression'
        elif findings:
            reference_policy = 'findings_only'
        elif impression:
            reference_policy = 'impression_only'
        else:
            reference_policy = 'missing'
        parent_images = xml.findall('.//parentImage')
        extant_count = 0
        report_issue = ''
        if not report_id:
            report_issue = 'missing_report_id'
        if not parent_images:
            report_issue = ';'.join(filter(None, (report_issue, 'no_parent_image')))
        if not reference:
            report_issue = ';'.join(filter(None, (report_issue, 'empty_reference')))
        base_report = dict(report_id=report_id, xml_path=str(xml_path), xml_sha256=xml_hash,
                           image_count_xml=len(parent_images), findings=findings,
                           impression=impression, ground_truth_report=reference,
                           reference_policy=reference_policy, reference_valid=bool(reference),
                           report_issue=report_issue)
        for element in parent_images:
            image_id = (element.get('id') or '').strip()
            figure_id = (element.findtext('figureId') or '').strip()
            caption = ' '.join((element.findtext('caption') or '').split())
            path = image_root / f'{image_id}.png' if image_id else image_root / '__missing_image_id__.png'
            exists = path.is_file()
            decodes, mode, width, height, image_hash, failure = False, '', '', '', '', ''
            if exists:
                extant_count += 1
                try:
                    image_hash = sha256(path)
                    with Image.open(path) as image:
                        mode, width, height = image.mode, image.width, image.height
                        image.verify()
                    decodes = True
                    ids_by_hash[image_hash].append(image_id)
                except (OSError, ValueError) as error:
                    failure = f'image_decode_or_read_error:{type(error).__name__}'
            else:
                failure = 'missing_image_file' if image_id else 'missing_image_id'
            row = dict(report_id=report_id, xml_path=str(xml_path), xml_sha256=xml_hash,
                       figure_id=figure_id, image_id=image_id, image_path=str(path.resolve()),
                       image_exists=exists, image_decodes=decodes, image_mode=mode,
                       image_width=width, image_height=height, image_sha256=image_hash,
                       caption=caption, findings=findings, impression=impression,
                       ground_truth_report=reference, reference_policy=reference_policy,
                       view_label='', view_evidence='', reviewer='', reviewed_at='',
                       selection_status='pending', exclusion_reason=failure)
            image_rows.append(row)
            if failure:
                issues.append(dict(report_id=report_id, xml_path=str(xml_path), image_id=image_id,
                                   issue=failure, details=''))
        base_report['image_count_exists'] = extant_count
        if report_issue:
            issues.append(dict(report_id=report_id, xml_path=str(xml_path), issue=report_issue, details=''))
        report_rows.append(base_report)
        if n % 500 == 0:
            print(f'Inventoried {n}/{len(xml_paths)} XML reports; {len(image_rows)} image references', flush=True)

    duplicates = {digest: sorted(set(ids)) for digest, ids in ids_by_hash.items() if len(set(ids)) > 1}
    duplicate_report_ids = sorted(key for key, count in duplicate_reports.items() if key and count > 1)
    image_issues = [r for r in image_rows if r['exclusion_reason']]
    valid_reports = [r for r in report_rows if r['report_id'] and r['reference_valid']]
    write_csv(inventory_dir / 'images.csv', image_rows, IMAGE_FIELDS)
    write_csv(inventory_dir / 'reports.csv', report_rows, REPORT_FIELDS)
    write_csv(inventory_dir / 'exclusions.csv', issues,
              ['report_id', 'xml_path', 'image_id', 'issue', 'details'])
    annotation_rows = [{key: row.get(key, '') for key in ANNOTATION_FIELDS}
                       for row in image_rows if row['image_exists'] and row['image_decodes']]
    write_csv(review_dir / 'view_annotations_template.csv', annotation_rows, ANNOTATION_FIELDS)
    decodable, review_path = make_review_html(image_rows, output, root)
    (review_dir / 'README.md').write_text(
        '# Indiana view and report-pairing review\n\n'
        f'The review page covers {decodable} decodable images across the complete XML image inventory. '
        'Review images in context with the reference text. A caption such as “PA and LAT” applies to '
        'the report and does not identify the view of each individual image.\n\n'
        'For every image, record an image-specific view label and evidence, whether it matches its '
        'report, reviewer and review time. Mark exactly one verified frontal image as `selected` '
        'for each included report; mark other images `not_selected` and give a reason in review notes. '
        'Use `unknown`, `uncertain`, or `mismatch` when evidence is insufficient; the cohort freezer '
        'will exclude those cases. Save a JSON draft regularly, then download the annotation CSV.\n\n'
        'If opening the HTML directly does not load images, serve the parent of the project directory '
        'so paths can reach the sibling dataset: from the project root run '
        '`python3 -m http.server 8765 --directory ..`, then open '
        '`http://127.0.0.1:8765/CXR_LLaVA_Improvement/result/indiana_full_k80_20261004_inventory02/review/view_review.html` '
        'through the server or an SSH tunnel.\n\n'
        'Save the downloaded file as `review/view_annotations.csv`, then freeze and validate it with:\n\n'
        '```bash\n'
        'python -m scripts.finalize_indiana_cohort \\\n'
        '  --inventory result/indiana_full_k80_20261004_inventory02/inventory/images.csv \\\n'
        '  --annotations result/indiana_full_k80_20261004_inventory02/review/view_annotations.csv \\\n'
        '  --output result/indiana_full_k80_20261004_inventory02/selection/reconstructed_cohort_01\n'
        '```\n', encoding='utf-8')
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'Indiana/OpenI NLMCXR', 'dataset_root': str(root),
        'xml_count': len(xml_paths), 'report_rows': len(report_rows),
        'reports_with_valid_id_and_reference': len(valid_reports),
        'image_references': len(image_rows), 'images_existing': sum(bool(r['image_exists']) for r in image_rows),
        'images_decodable': decodable, 'reports_with_multiple_image_refs': sum(r['image_count_xml'] > 1 for r in report_rows),
        'reports_with_zero_images': sum(r['image_count_xml'] == 0 for r in report_rows),
        'reports_with_empty_reference': sum(not r['reference_valid'] for r in report_rows),
        'image_errors': len(image_issues), 'duplicate_report_ids': duplicate_report_ids,
        'duplicate_image_hash_groups': duplicates,
        'reference_policy_counts': dict(Counter(r['reference_policy'] for r in report_rows)),
        'view_labels_assigned': 0, 'view_review_complete': False,
        'outputs': ['inventory/images.csv', 'inventory/reports.csv', 'inventory/exclusions.csv',
                    'review/view_annotations_template.csv', 'review/view_review.html', 'review/README.md'],
    }
    (inventory_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Open review package: {review_path}')


if __name__ == '__main__':
    main()
