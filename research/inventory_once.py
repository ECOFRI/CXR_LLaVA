"""One-time bounded inventory. Existing inventory is reused unless explicitly forced."""
import argparse
import collections
import csv
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from research.common import BASELINE, RESEARCH, ROOT, command, now, sha, write_csv, write_json

DATA = ROOT.parent / 'datasets'
CONDA = Path('/home/student4/anaconda3/bin/conda')


def summarize_tree(path):
    counts = collections.Counter()
    size = 0
    errors = []
    def error(exc):
        errors.append(str(exc))
    for directory, folders, files in os.walk(path, followlinks=False, onerror=error):
        folders[:] = [d for d in folders if d not in ('.git', '__pycache__')]
        for name in files:
            f = Path(directory) / name
            counts[f.suffix.lower()] += 1
            try:
                size += f.stat().st_size
            except OSError as exc:
                errors.append(str(exc))
    return {'file_counts_by_extension': dict(counts), 'logical_bytes': size,
            'errors': errors, 'method': 'bounded metadata traversal; no image binary hashing'}


def csv_summary(path):
    with path.open(newline='', encoding='utf-8-sig', errors='replace') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        count = sum(1 for _ in reader)
    return {'path': str(path), 'rows': count, 'columns': columns, 'sha256': sha(path)}


def inventory(force=False):
    folder = RESEARCH / 'inventory'
    sentinel = folder / 'file_index.json'
    if sentinel.exists() and not force:
        print('Reusing inventory:', sentinel)
        return
    folder.mkdir(parents=True, exist_ok=True)
    date = now()[:10].replace('-', '')
    info = json.loads(command([str(CONDA), 'info', '--envs', '--json'])['stdout'])
    envs = []
    for prefix in info['envs']:
        metadata = Path(prefix) / 'conda-meta'
        packages = []
        for f in metadata.glob('*.json'):
            obj = json.loads(f.read_text())
            packages.append({k: obj.get(k) for k in ('name', 'version', 'build', 'channel')})
        distributions = {}
        for lib in (Path(prefix) / 'lib').glob('python*/site-packages'):
            for package in lib.glob('*.dist-info'):
                name = package.name.rsplit('.dist-info', 1)[0]
                if name.startswith(('torch-', 'torchvision-', 'transformers-', 'numpy-', 'matplotlib-', 'clear', 'huggingface_hub-')):
                    distributions[name] = str(package)
        envs.append({'prefix': prefix, 'conda_packages': packages, 'relevant_pip_metadata': distributions})
    system = {'timestamp': now(), 'project_root': str(ROOT), 'authorized_roots': [str(ROOT), str(DATA), str(ROOT.parent / 'CLEAR')],
              'hostname': platform.node(), 'os': platform.platform(), 'conda_executable': str(CONDA),
              'conda_envs': envs, 'commands': {}}
    for key, args in {'cpu': ['lscpu'], 'ram': ['free', '-b'], 'gpu': ['nvidia-smi'],
                      'gpu_topology': ['nvidia-smi', 'topo', '-m'],
                      'mount': ['df', '-h', str(ROOT)], 'kernel': ['uname', '-a']}.items():
        system['commands'][key] = command(args)
    system['quota'] = command(['quota', '-s']) if shutil.which('quota') else {'status': 'UNVERIFIED', 'reason': 'quota executable absent'}
    probe = ('import json,sys,torch,importlib.metadata as m; '
             'print(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,'
             'arch=torch.cuda.get_arch_list(),gpu_count=torch.cuda.device_count(),'
             'gpu=[dict(name=torch.cuda.get_device_properties(i).name,bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())])))')
    for env in ('cxr-llava-k80', 'env_clear'):
        system['commands'][env] = command([f'/home/student4/anaconda3/envs/{env}/bin/python', '-c', probe])
    write_json(folder / f'system_environment_{date}.json', system)
    lines = ['# Conda inventory', '', f'Timestamp: {now()}; no environments changed.', '', '| Prefix | Relevant installed distributions |', '|---|---|']
    for e in envs:
        lines.append(f"| {e['prefix']} | {', '.join(e['relevant_pip_metadata'])} |")
    (folder / f'conda_envs_{date}.md').write_text('\n'.join(lines) + '\n')
    names = {'Indiana': 'NLMCRX', 'NIH_CXR8': 'CXR8', 'PadChest_sample': 'padChest_datasample',
             'CheXpert_test': 'CheXpert_test_labels', 'CheXpert_train_val': 'CheXpert-v1.0',
             'CheXpert_Plus': 'CheXpert-Plus', 'MIMIC_CXR': 'MIMIC-CXR', 'VinDr_CXR': 'VinDr-CXR', 'ReXGradient': 'ReXGradient'}
    catalog, index = [], {}
    actual_children = sorted(p.name for p in DATA.iterdir())
    for dataset, name in names.items():
        p = DATA / name
        present = p.is_dir()
        tree = summarize_tree(p) if present else {}
        summaries = [csv_summary(f) for f in p.rglob('*.csv') if '.git' not in f.parts] if present else []
        index[dataset] = {'status': 'PRESENT' if present else 'ABSENT_IN_AUTHORIZED_ROOTS', 'tree': tree, 'manifests': summaries}
        images = sum(v for k, v in tree.get('file_counts_by_extension', {}).items() if k in ('.jpg', '.jpeg', '.png', '.dcm'))
        catalog.append({'dataset': dataset, 'status': 'PRESENT' if present else 'ABSENT_IN_AUTHORIZED_ROOTS', 'path': str(p) if present else '',
                        'image_file_count': images if present else '', 'logical_bytes': tree.get('logical_bytes', ''),
                        'splits': 'test only' if dataset == 'CheXpert_test' else 'CSV split candidates; validation pending' if dataset == 'NIH_CXR8' else 'not established',
                        'reference_reports': 'XML available' if dataset == 'Indiana' else 'Report column available' if dataset == 'PadChest_sample' else 'not established / absent',
                        'label_provenance': 'official consensus image/study CSV' if dataset == 'CheXpert_test' else 'manifest metadata; audit required',
                        'access_license': 'local read access; redistribution/license rights not independently verified'})
        print(dataset, catalog[-1]['status'], images, flush=True)
    index['authorized_data_root_children'] = actual_children
    index['mimic_url_manifest_only'] = (DATA / 'mimic_table2_download_urls.txt').is_file()
    index['limitations'] = 'ABSENT refers only to inspected authorized roots. No sibling user directories or entire NFS scanned; no download URLs used.'
    write_csv(folder / 'dataset_catalog.csv', catalog)
    write_json(sentinel, index)
    code = ['# Code / artifact reuse map', '', f'Project: {ROOT}', '']
    for category in ('scripts', 'tests', 'CXR_LLAVA_HF', 'plan', 'reports'):
        code += [f'## {category}', *[f'- {f.relative_to(ROOT)}' for f in sorted((ROOT / category).glob('*')) if f.is_file()]]
    for p in (ROOT / 'codes', ROOT / 'code', ROOT / 'notebooks', ROOT.parent / 'CLEAR'):
        code.append(f'- {p}: {"PRESENT" if p.exists() else "ABSENT"}')
    clear = ROOT.parent / 'CLEAR'
    code += ['', 'CLEAR existing checkout commit: ' + command(['git', '-C', str(clear), 'rev-parse', 'HEAD'])['stdout'].strip(),
             'CheXpert saved source snapshot is authoritative for baseline parser audit; no repeated generation.',
             'CXR vision weights can be loaded independently from original safetensors; do not initialize LLaMA.']
    (folder / 'code_repo_map.md').write_text('\n'.join(code) + '\n')
    (folder / 'data_split_risk.md').write_text('''# Data split risk / entry gate

CheXpert 518 test images are locked: offline auditing and technical shape verification only; no prompt/threshold/probe tuning.
CheXpert train/validation, Plus, MIMIC image dataset, VinDr and ReXGradient were not located under inspected authorized roots.
The MIMIC download URL list alone is not an available image dataset.
NIH CXR8 has train/validation/test CSV candidates and patient keys; existence alone does not pass the data gate.
Before E1: reconcile locally present images, normalize label definitions, audit patient and image overlap across candidate splits and locked test, validate missing-label policy.
PadChest sample contains report and patient/study/label columns; no approved independent split established.
Indiana has reports and prior evaluation but is not automatically a CheXpert validation cohort.
Pretraining overlap with either released encoder is uncertain; do not call these unseen external cohorts without provenance evidence.
No full-image hash pass performed; duplicate image overlap remains unverified beyond frozen CheXpert hashes.
Patient/study identifiers remain in source manifests; public outputs use pseudonyms and local path pointers only.
Data gate: HOLD pending independent split/label/availability validation. No training authorized by this inventory.
''')
    print('Inventory complete:', folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Only use when explicitly authorized to repeat inventory')
    inventory(parser.parse_args().force)
