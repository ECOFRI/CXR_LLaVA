import csv
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VR = ROOT / 'research/blip2_cxr_v4'
RUN = VR / 'runs/P0_audit_20261022'
MAN = VR / 'manifests'
REP = VR / 'reports'


def write_csv(name, fields, rows):
    with (MAN / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def file_schema(path):
    with path.open(newline='', errors='replace') as f:
        reader = csv.reader(f)
        return next(reader), sum(1 for _ in reader)


def package_envs():
    raw = (RUN / 'env_direct.log').read_text(errors='replace')
    entries = []
    for chunk in raw.split('ENV ')[1:]:
        lines = chunk.splitlines()
        path = lines[0].strip()
        payload = next((s for s in lines[1:] if s.startswith('{')), None)
        try:
            data = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            data = {}
        row = {'prefix': path, 'python': data.get('python', 'UNKNOWN'),
               'torch_cuda_build': data.get('torch_cuda_build', 'UNKNOWN'),
               'cuda_available': data.get('cuda_available', 'UNKNOWN'),
               'fp16_supported': data.get('fp16_supported', 'UNKNOWN'),
               'bf16_supported': data.get('bf16_supported', 'UNKNOWN'),
               'probe_status': 'DIRECT_OK' if data else 'UNKNOWN'}
        for k in ['torch','transformers','peft','accelerate','bitsandbytes','deepspeed',
                  'lavis','salesforce-lavis','timm','open_clip','nltk','pycocoevalcap',
                  'rouge_score','evaluate','sacrebleu','f1chexbert','chexbert','radgraph']:
            row[k] = data.get('packages', {}).get(k, 'UNKNOWN')
        entries.append(row)
    if not entries:
        entries = [{'prefix': 'UNKNOWN', 'probe_status': 'UNKNOWN'}]
    write_csv('conda_envs.csv', list(entries[0]), entries)
    return entries


def artifact_rows():
    roots = [ROOT / 'CXR_LLAVA_HF', ROOT / 'model_cache', ROOT / 'research/upstream',
             Path.home() / '.cache/huggingface/hub', Path('/storage/shared/MODELS')]
    rows, weights = [], []
    for base in roots:
        if not base.exists():
            rows.append({'name': base.name, 'path': str(base), 'size': 'NOT_FOUND',
                         'revision': 'UNKNOWN', 'license_file': 'UNKNOWN'})
            continue
        children = [base] if base.name == 'CXR_LLAVA_HF' else [p for p in base.iterdir() if p.is_dir()]
        for p in children:
            try:
                size = subprocess.run(['du', '-sh', str(p)], capture_output=True,
                                      text=True, timeout=20).stdout.split()[0]
            except subprocess.TimeoutExpired:
                size = 'TIMEOUT'
            except IndexError:
                size = 'UNKNOWN'
            rev = 'UNKNOWN'
            if (p / '.git').exists():
                try:
                    cp = subprocess.run(['git', '-C', str(p), 'rev-parse', 'HEAD'],
                                        capture_output=True, text=True, timeout=20)
                    if cp.returncode == 0:
                        rev = cp.stdout.strip()
                except subprocess.TimeoutExpired:
                    rev = 'TIMEOUT'
            license_present = any((p / x).exists() for x in ['LICENSE','LICENSE.txt','LICENSE.md'])
            row = {'name': p.name, 'path': str(p), 'size': size, 'revision': rev,
                   'license_file': str(license_present)}
            rows.append(row)
            if p.name.lower().find('cxr') >= 0 or p.name.lower().find('clear') >= 0:
                weights.append({**row, 'checkpoint_files': 'UNKNOWN'})
    write_csv('code_artifacts.csv', list(rows[0]), rows)
    write_csv('checkpoints.csv', list(weights[0]) if weights else
              ['name','path','size','revision','license_file','checkpoint_files'], weights)
    return rows


def main():
    envs = package_envs()
    artifacts = artifact_rows()
    gpu_text = (RUN / 'system_raw.log').read_text(errors='replace')
    gpu = []
    for line in gpu_text.splitlines():
        if re.match(r'^\d+, Tesla K80,', line):
            bits = [x.strip() for x in line.split(',')]
            gpu.append({'index': int(bits[0]), 'name': bits[1], 'memory_total_mib': int(bits[2].split()[0]),
                        'memory_used_mib': int(bits[3].split()[0]), 'driver': bits[4],
                        'compute_capability': '3.7 (K80 architecture; nvidia-smi query unsupported)'})
    (MAN / 'gpu_inventory.json').write_text(json.dumps(gpu, indent=2) + '\n')

    mimic = Path('/storage/shared/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.1.0')
    report = Path('/storage/shared/MIMIC-CXR-Report/download-mimic-cxr-txt/files')
    vindr = Path('/storage/shared/VinDr-CXR')
    ind = ROOT / 'result/indiana_full_k80_20261004_031940_1252388/inference'
    e4 = ROOT / 'research/runs/20261017_E4_B1_evidence_format_gate_v1/E4_B1_reports.jsonl'
    train_h, train_n = file_schema(vindr / 'annotations/image_labels_train.csv')
    test_h, test_n = file_schema(vindr / 'annotations/image_labels_test.csv')
    pred_h, pred_n = file_schema(ind / 'predictions.csv')
    cohort_h, cohort_n = file_schema(ind / 'cohort.csv')
    with e4.open() as f:
        e4_n = sum(1 for _ in f)
    datasets = [
        {'name':'MIMIC-CXR-JPG','version':'2.1.0 directory; 2.0.0 metadata','root':str(mimic),
         'num_images':'377110 metadata rows','num_studies':'UNKNOWN','num_patients':'UNKNOWN',
         'report_available':'separate report tree; sample link YES; completeness UNKNOWN',
         'labels_available':'YES CheXpert/NegBio','view_types':'ViewPosition column','splits':'split CSV 377110 rows',
         'permissions':'READABLE metadata','leakage_risk':'patient/study grouping and pretraining UNKNOWN'},
        {'name':'MIMIC-CXR-Report','version':'UNKNOWN','root':str(report),'num_images':'N/A',
         'num_studies':'UNKNOWN','num_patients':'UNKNOWN','report_available':'YES .txt sample; completeness UNKNOWN',
         'labels_available':'UNKNOWN','view_types':'N/A','splits':'link through subject/study path; full audit pending',
         'permissions':'READABLE sample path','leakage_risk':'linkage/split validation pending'},
        {'name':'VinDr-CXR','version':'UNKNOWN; LICENSE.txt present','root':str(vindr),
         'num_images':'UNKNOWN; label rows are not unique images','num_studies':'UNKNOWN','num_patients':'UNKNOWN',
         'report_available':'NOT_FOUND in shallow inventory','labels_available':f'YES {train_n} train reader rows; {test_n} test image rows',
         'view_types':'UNKNOWN','splits':'train/test','permissions':'READABLE labels',
         'leakage_risk':'patient IDs and version unverified'},
        {'name':'Indiana local baseline','version':'UNKNOWN','root':str(ind),
         'num_images':str(pred_n)+' prediction rows','num_studies':'UNKNOWN','num_patients':'UNKNOWN',
         'report_available':'YES prediction/reference columns','labels_available':'derived labels elsewhere',
         'view_types':'view_label column','splits':'UNKNOWN','permissions':'READABLE local artifact',
         'leakage_risk':'study/image matching and cohort audit pending'}]
    write_csv('datasets.csv', list(datasets[0]), datasets)

    (VR / '.gitignore').write_text('configs/paths.local.yaml\nruns/\nresults/\n')
    (VR / 'README.md').write_text('# BLIP-2 CXR v4 research workspace\n\nP0 inventory only. Research use; no clinical diagnosis. No inference or training was run. See `reports/` and `manifests/`.\n')
    (VR / 'configs/paths.example.yaml').write_text('project_root: UNKNOWN\nindiana_predictions: UNKNOWN\nmimic_jpg_root: UNKNOWN\nmimic_report_root: UNKNOWN\nvindr_root: UNKNOWN\n')
    (VR / 'configs/paths.local.yaml').write_text(f'project_root: {ROOT}\nindiana_predictions: {ind / "predictions.csv"}\nmimic_jpg_root: {mimic}\nmimic_report_root: {report}\nvindr_root: {vindr}\n')
    (REP / '00_system_inventory.md').write_text(
        '# P0 system inventory (2026-10-22)\n\n'
        'Verified: user student4; host ictserver6; Linux 4.19.0-14-amd64; group student4. '
        f'Project root `{ROOT}`. `df -h` reports /storage 813T total, 1.8T available, 100% used; '
        '/home 738G total, 0 available, 100% used. Scheduler commands sbatch/squeue NOT_FOUND; '
        'tmux session count 13. nvcc `/usr/local/cuda-10.2/bin/nvcc`.\n\n'
        f'GPU: {len(gpu)} Tesla K80 devices, 11441 MiB each, 0 MiB used at query, driver 440.33.01. '
        'Running compute process owner/memory: none returned by `nvidia-smi --query-compute-apps`; '
        'compute capability query unsupported by this nvidia-smi, K80 architecture 3.7 not runtime verified.\n\n'
        'Conda: `/home/student4/anaconda3/bin/conda env list`; every `conda run -p` failed '
        '`NoSpaceLeftError`. Direct env interpreters were probed under 60s timeout; see `manifests/conda_envs.csv`. '
        'No packages installed. Repository Git SHA/remotes/branch: UNKNOWN (not required by WO and no Git writes).\n')
    (REP / '01_dataset_inventory.md').write_text(
        '# P0 dataset inventory\n\n'
        'MIMIC-CXR-JPG metadata and split CSVs each have 377110 rows; columns include '
        '`dicom_id,subject_id,study_id` and split `split`. The separate MIMIC-CXR-Report tree '
        'has a sampled study text file under a subject folder, and the JPG tree has '
        'the matching study directory. This establishes a sample path linkage, not coverage. '
        'Free text was not opened. Report completeness, unique studies/patients, and permissions '
        'across all partitions remain UNKNOWN.\n\n'
        f'VinDr label files: train image-reader rows {train_n}, test image rows {test_n}; '
        f'columns include `{train_h[0]}`, `{train_h[1]}` and finding labels. Annotation boxes exist. '
        'Version is UNKNOWN; free-text reports NOT_FOUND in shallow inventory. Do not use report metrics.\n\n'
        f'Indiana local prediction and cohort CSVs each have {pred_n} rows with `report_id,image_id` '
        'and reference/generated report columns. Values were not read. Study matching remains to audit. '
        f'E4 B1 report JSONL has {e4_n} rows; 24-image uniqueness and valid statuses UNKNOWN pending P1.\n\n'
        'See `manifests/datasets.csv` for paths and remaining unknowns. Dataset roots were read only.\n')
    catalog = list(csv.DictReader((VR / 'plan/VLM_MEDICAL_V4_EXPERIMENT_CATALOG.csv').open()))
    lines = ['# Experiment tracker — P0 inventory only','',
             '| Date | Phase | ID | Dataset / split / manifest | Model / checkpoint / repo | Env / device / run | Status | Metrics / CI / runtime / VRAM | Blocker / next action |',
             '|---|---|---|---|---|---|---|---|---|']
    for item in catalog:
        x = item['experiment_id']
        lines.append(f'| 2026-10-22 | P0 | {x} | {item["datasets"]}; split/SHA UNKNOWN | {item["paper"]}; checkpoint/SHA UNKNOWN | UNKNOWN; no run | PLANNED | N/A | P1 linkage/overlap and P3 asset audit; GPU/training authorization where applicable |')
    (REP / '04_experiment_tracker.md').write_text('\n'.join(lines)+'\n')
    (REP / '05_results_and_decisions.md').write_text(
        '# P0 results and decisions\n\nClassification: inventory only; no smoke test, metric-complete evaluation, or benchmark claim. '
        'No model load, GPU inference, training, or metrics were run.\n\n'
        'Decisions: stop GPU work at P0 gate; preserve existing Indiana outputs for later matching; '
        'treat VinDr as label-only until report text is verified; verify MIMIC report/JPG linkage and '
        'coverage before report metrics; resolve env probe/no-space blocker. Historical scores in '
        'the plan are not newly verified here.\n')
    matrix = ['# P0 compatibility matrix','',
              'REUSE means prior local execution evidence, not readiness for a new run. MISSING means no verified local code/weights. INCOMPATIBLE means detected runtime constraint.','',
              '| Env | CXR-LLaVA | CLEAR | BLIP-2 | ClinicalBLIP | R2-LLM | META-CXR | RRG-LLM | Swin-Qwen3 |',
              '|---|---|---|---|---|---|---|---|---|']
    for e in envs:
        p = e['prefix']; short = Path(p).name
        pk = e
        cxr = 'REUSE: prior run manifest' if short in ('cxr-llava','cxr-llava-k80') else 'MISSING: no run evidence'
        clear = 'REUSE: prior run manifest' if short in ('env_clear','cxr-clear-cpu') else 'MISSING: no run evidence'
        if short == 'cxr-clear-cpu':
            clear = 'INCOMPATIBLE: CPU-only for GPU work'
        third = 'MISSING: code/weights unverified'
        matrix.append(f'| `{short}` | {cxr} | {clear} | {third} | {third} | {third} | {third} | {third} | {third} |')
    matrix += ['', 'Proposed environment action: resolve /home space issue and verify per-model code, weights, license and CUDA compatibility before choosing isolated envs. No installation in P0.']
    (REP / 'P0_compat_matrix.md').write_text('\n'.join(matrix)+'\n')
    (REP / 'P1_proposal.md').write_text(
        '# Proposed P1 work order\n\n'
        '1. CPU audit Indiana 3667 prediction/cohort rows: unique report/image keys, nulls, exact '
        'one-to-one linkage, reference source and view policy; inspect E4 B1 264 JSONL statuses, arms, '
        'unique image keys and references. Do not score yet.\n'
        '2. CPU audit MIMIC report/JPG coverage by subject and study path and `subject_id/study_id` '
        'metadata, plus patient/study grouped splits. Use aggregate counts only in reports.\n'
        '3. CPU audit VinDr version/license, label ontology and split overlap; retain classification '
        'scope unless free-text reports are verified.\n'
        '4. Test split overlap, duplicate image/study IDs, and known pretraining exposure; emit '
        'relative-path manifests with controlled identifiers stored only in protected local artifacts.\n\n'
        'Estimated CPU cost: Indiana/E4 seconds to minutes; MIMIC metadata scans and report coverage '
        'minutes to tens of minutes depending on network filesystem. No GPU expected. '
        'Blockers: /home space, full MIMIC report coverage UNKNOWN, checkpoint/license and paper '
        'provenance UNKNOWN.\n')
    (RUN / 'status.json').write_text(json.dumps({'phase':'P0','status':'PARTIAL',
        'reason':'conda run NoSpaceLeftError; several identities and coverage unresolved',
        'gpu_jobs':0,'model_loads':0,'metrics_computed':0}, indent=2)+'\n')
    (RUN / 'stdout.log').write_text('P0 inventory completed without model loading or metrics. '
        'Raw command outputs: system_raw.log, env_raw.log, env_direct.log, dataset_raw.log.\n')


if __name__ == '__main__':
    main()
