"""V3 CPU-only train/development analyses of immutable cached representations."""
import argparse
import csv
import importlib.metadata
import json
import os
import pickle
import resource
import shutil
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.common import ROOT, RESEARCH, now, sha, write_csv, write_json

LIMITATION = 'NIH exploratory; potential CXR encoder pretraining overlap; not independent external test.'
LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
          'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule',
          'Pleural Thickening', 'Pneumonia', 'Pneumothorax']
FEATURE_DIRS = {'cxr': RESEARCH / 'runs/20261013T220300_E1_CXR',
                'clear': RESEARCH / 'runs/20261014_E1_CLEAR_CPU'}
ARMS = {'A': ['cxr'], 'B': ['clear'], 'A+B': ['cxr', 'clear']}
C_GRID = [.001, .01, .1, 1.]


def inside(path):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT), f'Outside authorized project: {path}'
    return path


def load_dev():
    """Join selected train/val IDs; test rows are never used in numeric analyses."""
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cohort = inside(RESEARCH / 'inventory/e1_nih/cohort_full.csv')
    # CSV transport includes old test rows; discard them before reading label values.
    rows = [r for r in csv.DictReader(cohort.open()) if r['split'] in ('train', 'val')]
    assert len({r['image_id'] for r in rows}) == 4127 == len(rows)
    historical = json.loads((RESEARCH / 'runs/20261014_E1_NIH_probes/run_manifest.json').read_text())
    assert historical['included_classes'] == LABELS and historical['excluded_classes'] == ['Hernia']
    rows.sort(key=lambda r: (r['split'] != 'train', r['image_id']))
    features, bindings, inputs, cache_indices = {}, {}, {}, {}
    for encoder, directory in FEATURE_DIRS.items():
        directory = inside(directory)
        manifest = json.loads((directory / 'run_manifest.json').read_text())
        assert manifest['status'] == 'COMPLETE' and manifest['checkpoint_integrity'] == 'PASS'
        assert sha(cohort) == manifest['cohort_sha256']
        cache = directory / 'features.npz'
        assert sha(cache) == manifest['cache_sha256'], f'{encoder} cache SHA256 mismatch'
        records = list(csv.DictReader((directory / 'feature_manifest.csv').open()))
        mapping = {r['image_id']: r for r in records}
        assert len(mapping) == len(records) == 4999
        with np.load(cache, allow_pickle=False) as data:
            ids = list(data['image_ids'])
            assert len(ids) == len(set(ids)) == 4999 and data['complete'].all()
            lookup = {str(key): i for i, key in enumerate(ids)}
            indices = [lookup[r['image_id']] for r in rows]
            cache_indices[encoder] = indices
            features[encoder] = {}
            for representation in ('cls', 'patchmean'):
                # NPZ member decompression transports a whole member; slice immediately.
                # No test feature statistics, predictions, or labels are computed.
                selected = data[representation][indices].astype(np.float64)
                assert selected.shape == (4127, 1024 if encoder == 'cxr' else 768)
                assert np.isfinite(selected).all()
                features[encoder][representation] = selected
        for r in rows:
            item = mapping[r['image_id']]
            assert item['split'] == r['split'] and item['patient_id'] == r['patient_id']
        bindings[encoder] = mapping
        inputs[encoder] = {'cache': str(cache), 'cache_sha256': sha(cache),
                           'manifest': str(directory / 'run_manifest.json'),
                           'manifest_sha256': sha(directory / 'run_manifest.json'),
                           'feature_manifest_sha256': sha(directory / 'feature_manifest.csv'),
                           'checkpoint_identity': manifest.get('checkpoint_sha256', manifest.get('checkpoint_revision'))}
    for r in rows:
        assert bindings['cxr'][r['image_id']]['source_sha256'] == bindings['clear'][r['image_id']]['source_sha256']
    ntrain = sum(r['split'] == 'train' for r in rows)
    assert ntrain == 3783
    groups = np.array([r['patient_id'] for r in rows])
    assert len(set(groups[:ntrain])) == 1222 and len(set(groups[ntrain:])) == 57
    assert not set(groups[:ntrain]) & set(groups[ntrain:])
    labels = np.array([[int(r[name]) for name in LABELS] for r in rows], dtype=np.int8)
    assert np.isin(labels, [0, 1]).all()
    alignment = [dict(image_id=r['image_id'], patient_id=r['patient_id'], split=r['split'],
                      source_sha256=bindings['cxr'][r['image_id']]['source_sha256'],
                      cxr_row=cache_indices['cxr'][next_index], clear_row=cache_indices['clear'][next_index])
                 for next_index, r in enumerate(rows)]
    return rows, features, groups, labels, ntrain, alignment, inputs


def folds_for(groups):
    outer = list(GroupKFold(n_splits=3).split(np.zeros(len(groups)), groups=groups))
    inner = []
    for k, (fit, held) in enumerate(outer):
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                           random_state=20261015 + k).split(np.zeros(len(fit)), groups=groups[fit]))
        inner.append((fit[left], fit[right]))
        assert not set(groups[fit]) & set(groups[held])
    left, right = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                       random_state=20261115).split(np.zeros(len(groups)), groups=groups))
    inner.append((left, right))
    for left, right in inner:
        assert not set(groups[left]) & set(groups[right])
    return outer, inner


def scale_blocks(features, arm, fit, evaluate):
    train, evaluation, scalers = [], [], []
    for encoder in ARMS[arm]:
        scaler = StandardScaler().fit(features[encoder]['patchmean'][fit])
        train.append(scaler.transform(features[encoder]['patchmean'][fit]))
        evaluation.append(scaler.transform(features[encoder]['patchmean'][evaluate]))
        scalers.append(scaler)
    return np.concatenate(train, axis=1), np.concatenate(evaluation, axis=1), scalers


def classifier(c):
    return LogisticRegression(C=c, solver='liblinear', penalty='l2', class_weight=None,
                              max_iter=1000, random_state=42)


def metrics(y, score, predicted):
    both = len(np.unique(y)) == 2
    tp = int(((y == 1) & predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    return {'AP': float(average_precision_score(y, score)) if both else float('nan'),
            'AUROC': float(roc_auc_score(y, score)) if both else float('nan'),
            'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float('nan'),
            'precision': tp / (tp + fp) if tp + fp else float('nan'),
            'recall': tp / (tp + fn) if tp + fn else float('nan')}


def cka(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    cross, xx, yy = x.T @ y, x.T @ x, y.T @ y
    return float(np.sum(cross * cross) / np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))


def protocol(phase):
    return {'phase': phase, 'written_at': now(), 'limitation': LIMITATION,
            'cpu_only': True, 'GPU_hours': 0, 'inference': False, 'test_analysis': False,
            'data': 'NIH original TRAIN; original VALIDATION once in E5 as secondary DEV',
            'outer': '3 GroupKFold folds, image-count-balanced patient groups; deterministic sklearn1.5.2',
            'inner': 'one GroupShuffleSplit 25% patients; seeds20261015+outer_fold; full-train seed20261115',
            'labels': LABELS, 'excluded': ['Hernia', 'supplemental labels', 'No Finding'],
            'CKA': 'biased linear feature-space CKA, sample centered, no coordinate standardization; full train and one image/patient sensitivity',
            'ridge': 'alpha100 fixed; source and target standardization fit outer-training only; 3 patient-grouped OOF folds both directions both representations; inverse transform target predictions before variance-weighted R2',
            'E13_error_complementarity': 'DEFERRED_TO_E5_OOF; CKA and ridge do not prove useful complementarity',
            'E5_arms': ARMS, 'E5_representations': 'patchmean only; CLS not evaluated in E5',
            'C_grid': C_GRID, 'C_selection': 'inner validation AP per label, tie smallest C',
            'threshold': 'inner validation F1 grid0.01:0.99; tie closest0.5 then largest; transferred to outer/full refit',
            'scaling': 'independent encoder blocks, inner-fit-only during selection; outer-fit-only at refit',
            'solver': 'L2 liblinear; maxiter1000; class_weightNone; seed42; nonconvergence stops phase',
            'primary': 'pooled train OOF macro sklearn AP across same13 findings; fold variability secondary',
            'bootstrap': '500 shared patient-cluster draws seed20261015; fixed13-label macro only on draws with all labels defined; no fitting; max(A,B) recomputed inside each draw',
            'validation': 'all3 arm models/selections frozen and hashed before original validation scoring once; DEV only',
            'decision': 'fusion minus max(single-arm macro AP); interpret paired uncertainty and fold signs, no numeric uplift cutoff',
            'resource_caps': {'RSS_GiB': 8, 'E13_wall_sec': 3600, 'E5_wall_sec': 7200, 'new_output_bytes': 2 * 1024**3}}


def preflight(out):
    rows, features, groups, y, nt, alignment, inputs = load_dev()
    outer, inner = folds_for(groups[:nt])
    support = []
    for name, ix in [(f'outer{k}_fit', a) for k, (a, b) in enumerate(outer)] + [(f'outer{k}_held', b) for k, (a, b) in enumerate(outer)] + [(f'inner{k}_{role}', a) for k, pair in enumerate(inner) for role, a in zip(('fit', 'select'), pair)] + [('train', np.arange(nt)), ('validation_dev', np.arange(nt, len(rows)))]:
        for j, label in enumerate(LABELS):
            positive = y[ix, j] == 1
            support.append(dict(partition=name, finding=label, images=len(ix), patients=len(set(groups[ix])),
                                positive=int(positive.sum()), negative=int((~positive).sum()),
                                positive_patients=len(set(groups[ix][positive])), limitation=LIMITATION))
            assert positive.any() and (~positive).any(), f'Undefined class support: {name}/{label}'
    saved_support = {(r['split'], r['finding']): r for r in csv.DictReader((RESEARCH / 'inventory/e1_nih/class_support.csv').open())}
    for r in support:
        if r['partition'] in ('train', 'validation_dev'):
            old = saved_support['train' if r['partition'] == 'train' else 'val', r['finding']]
            assert all(int(old[k]) == r[k] for k in ('positive', 'negative', 'patients', 'positive_patients'))
    fold_rows = []
    for i, r in enumerate(alignment[:nt]):
        d = dict(r)
        d['outer_fold'] = next(k for k, (_, held) in enumerate(outer) if i in set(held))
        for k, (fit, select) in enumerate(inner):
            d[f'inner{k}_role'] = 'fit' if i in set(fit) else ('select' if i in set(select) else 'outer_heldout')
        fold_rows.append(d)
    write_csv(out / 'dev_sample_alignment.csv', alignment)
    write_csv(out / 'fold_manifest.csv', fold_rows)
    write_csv(out / 'label_patient_support.csv', support)
    write_json(out / 'input_manifest.json', inputs)
    write_json(out / 'e13_protocol.yaml', protocol('E1.3'))
    write_json(out / 'e5_protocol.yaml', protocol('E5'))  # JSON is valid YAML1.2.
    reserve = out / 'allocation_probe.bin'
    with reserve.open('wb') as stream:
        os.posix_fallocate(stream.fileno(), 0, 128 * 1024**2)
    reserve.unlink()
    timings = []
    for arm in ARMS:
        x, _, _ = scale_blocks(features, arm, np.arange(nt), np.arange(1))
        start = time.perf_counter()
        classifier(1.).fit(x, y[:nt, 0])  # Train-only cost pilot, no score/selection.
        timings.append(time.perf_counter() - start)
    x = StandardScaler().fit_transform(features['cxr']['patchmean'][:nt])
    target = StandardScaler().fit_transform(features['clear']['patchmean'][:nt])
    start = time.perf_counter()
    Ridge(alpha=100., solver='cholesky').fit(x, target)
    ridge_seconds = time.perf_counter() - start
    start = time.perf_counter()
    _ = x.T @ x
    gram_seconds = time.perf_counter() - start
    e13_estimate = 12 * ridge_seconds * 3 + 12 * gram_seconds * 3 + 120
    e5_estimate = max(timings) * 780 * 1.5 + 300
    estimate = {'logistic_fulltrain_fit_sec_by_arm': dict(zip(ARMS, timings)),
                'ridge_fulltrain_fit_sec': ridge_seconds, 'gram_sec': gram_seconds,
                'E13_estimated_wall_sec': e13_estimate, 'E5_estimated_wall_sec': e5_estimate,
                'E5_candidate_and_refit_count': 780, 'estimated_new_outputs_MiB': 80,
                'estimated_peak_RSS_GiB': 2, 'account_quota': 'numeric quota unavailable; bounded128MiB actual allocation PASS',
                'free_disk_GiB': shutil.disk_usage(ROOT).free / 1024**3,
                'available_RAM_GiB': next(int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')) / 1024**2,
                'test_transport_policy': 'Full NPZ bytes verified; only aligned TRAIN/VALIDATION slices returned. Old test labels not accessed; no test statistics/scoring.',
                'GO': e13_estimate <= 3600 and e5_estimate <= 7200}
    write_json(out / 'resource_estimates.json', estimate)
    (out / 'preflight.md').write_text('# G0 cached-feature gate\n\n' + LIMITATION + '\n\n'
        'Matched3783train/1222patients and344validation/57patients. Patient splits disjoint. '
        'Exactly13original labels; no imputation. Joined by image ID/source hash/split, not cache order. '
        'Existing model source/checkpoint identities reused from immutable manifests, not reopened outside root. '
        'No GPU/model inference. Fold and protocol files saved before analysis.\n\n'
        '```json\n' + json.dumps({'inputs': inputs, 'estimates': estimate}, indent=2) + '\n```\n')
    assert estimate['GO'], f'Resource estimate exceeds cap: {estimate}'
    print(json.dumps(estimate), flush=True)


def verify_gate(gate):
    gate = inside(gate)
    assert json.loads((gate / 'resource_estimates.json').read_text())['GO']
    for directory in FEATURE_DIRS.values():
        manifest = json.loads((directory / 'run_manifest.json').read_text())
        assert sha(directory / 'features.npz') == manifest['cache_sha256']
    return gate


def e13(out, gate):
    rows, features, groups, y, nt, alignment, inputs = load_dev()
    folds, _ = folds_for(groups[:nt])
    assert sha(gate / 'e13_protocol.yaml') == json.loads((gate / 'run_manifest.json').read_text())['protocols']['e13']
    shutil.copyfile(gate / 'e13_protocol.yaml', out / 'e13_protocol.yaml')
    write_csv(out / 'e13_sample_alignment.csv', alignment[:nt])
    similarities, cross, residuals = [], [], []
    one_per_patient = np.array([np.flatnonzero(groups[:nt] == p)[0] for p in sorted(set(groups[:nt]))])
    for rep in ('cls', 'patchmean'):
        a, b = features['cxr'][rep][:nt], features['clear'][rep][:nt]
        for sample, indices in [('all_train_images', np.arange(nt)), ('one_image_per_patient', one_per_patient)]:
            value = cka(a[indices], b[indices])
            assert np.isfinite(value) and -1e-12 <= value <= 1 + 1e-12
            similarities.append(dict(representation=rep, sample=sample, images=len(indices), patients=len(set(groups[indices])), linear_CKA=value, limitation=LIMITATION))
        for source, target in [('cxr', 'clear'), ('clear', 'cxr')]:
            x, target_values = features[source][rep][:nt], features[target][rep][:nt]
            predictions = np.empty_like(target_values)
            for k, (fit, held) in enumerate(folds):
                sx, sy = StandardScaler().fit(x[fit]), StandardScaler().fit(target_values[fit])
                model = Ridge(alpha=100., solver='cholesky').fit(sx.transform(x[fit]), sy.transform(target_values[fit]))
                pred = sy.inverse_transform(model.predict(sx.transform(x[held])))
                assert np.isfinite(pred).all()
                predictions[held] = pred
                cross.append(dict(representation=rep, direction=f'{source}_to_{target}', partition=f'outer{k}', images=len(held), patients=len(set(groups[held])), R2_variance_weighted=float(r2_score(target_values[held], pred, multioutput='variance_weighted')), limitation=LIMITATION))
                err = np.sum((pred - target_values[held])**2, axis=1)
                baseline = np.sum((target_values[held] - target_values[fit].mean(0))**2, axis=1)
                for i, loss, base in zip(held, err, baseline):
                    residuals.append(dict(image_id=rows[i]['image_id'], patient_id=groups[i], source_sha256=alignment[i]['source_sha256'], representation=rep, direction=f'{source}_to_{target}', outer_fold=k, squared_error=float(loss), train_mean_baseline_squared_error=float(base)))
                print('E13', rep, source, 'fold', k, flush=True)
            cross.append(dict(representation=rep, direction=f'{source}_to_{target}', partition='pooled_OOF', images=nt, patients=len(set(groups[:nt])), R2_variance_weighted=float(r2_score(target_values, predictions, multioutput='variance_weighted')), limitation=LIMITATION))
    write_csv(out / 'e13_similarity.csv', similarities)
    write_csv(out / 'e13_cross_prediction.csv', cross)
    write_csv(out / 'e13_oof_residuals.csv', residuals)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar([r['representation'] + '/' + r['sample'] for r in similarities], [r['linear_CKA'] for r in similarities])
    axes[0].set_ylabel('Centered linear CKA'); axes[0].tick_params(axis='x', rotation=25)
    pooled = [r for r in cross if r['partition'] == 'pooled_OOF']
    axes[1].bar([r['representation'] + '/' + r['direction'] for r in pooled], [r['R2_variance_weighted'] for r in pooled])
    axes[1].set_ylabel('Patient-fold OOF ridge R²'); axes[1].tick_params(axis='x', rotation=25)
    fig.suptitle(LIMITATION + '\nTRAIN only; neither score proves complementarity', fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'e13_similarity.png', dpi=140); plt.close(fig)
    write_json(out / 'results.json', {'similarity': similarities, 'cross_prediction': pooled,
               'observation': 'INCONCLUSIVE_COMPLEMENTARITY; geometry scores are descriptive',
               'complementarity': 'COMPLEMENTARITY_NOT_YET_PROVEN; DEFERRED_TO_E5_OOF'})


def fitted_arm(features, y, fit, inner, held, arm, tag, selections, convergence):
    left, right = inner
    x, iv, _ = scale_blocks(features, arm, left, right)
    best = []
    for j, label in enumerate(LABELS):
        selected = None
        for c in C_GRID:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = classifier(c).fit(x, y[left, j])
            convergence.append(dict(partition=tag, arm=arm, finding=label, C=c,
                                    n_iter=int(model.n_iter_.max()), warnings='; '.join(str(w.message) for w in caught)))
            assert not caught, f'Candidate convergence warning: {tag}/{arm}/{label}/{c}'
            score = model.predict_proba(iv)[:, 1]
            assert np.isfinite(score).all()
            ap = average_precision_score(y[right, j], score)
            if selected is None or ap > selected[0] + 1e-12:
                selected = (ap, c, score)
        ap, c, score = selected
        candidates = []
        for threshold in np.round(np.arange(.01, 1, .01), 2):
            f1 = metrics(y[right, j], score, score >= threshold)['F1']
            candidates.append((f1, -abs(threshold - .5), threshold))
        threshold = sorted(candidates, reverse=True)[0][2]
        selections.append(dict(partition=tag, arm=arm, finding=label, C=c, threshold=threshold,
                               inner_AP=float(ap), inner_fit_patients='see frozen fold manifest'))
        best.append((c, threshold))
    x, h, scalers = scale_blocks(features, arm, fit, held)
    scores = np.empty((len(held), len(LABELS))); thresholds = []
    models = []
    for j, (c, threshold) in enumerate(best):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); model = classifier(c).fit(x, y[fit, j])
        assert not caught, f'Refit convergence warning: {tag}/{arm}/{LABELS[j]}'
        convergence.append(dict(partition=tag, arm=arm, finding=LABELS[j], C=c, n_iter=int(model.n_iter_.max()), warnings='', stage='refit'))
        models.append(model); thresholds.append(threshold)
    # This function freezes fitted models before held-out prediction is requested.
    return models, scalers, np.array(thresholds), h


def e5(out, gate):
    rows, features, groups, y, nt, alignment, inputs = load_dev()
    outer, inner = folds_for(groups[:nt])
    shutil.copyfile(gate / 'e5_protocol.yaml', out / 'e5_protocol.yaml')
    shutil.copyfile(gate / 'fold_manifest.csv', out / 'e5_fold_manifest.csv')
    assert sha(out / 'e5_protocol.yaml') == json.loads((gate / 'run_manifest.json').read_text())['protocols']['e5']
    scores = {arm: np.empty((len(rows), len(LABELS))) for arm in ARMS}
    thresholds = {arm: np.empty((len(rows), len(LABELS))) for arm in ARMS}
    selections, convergence, final_models = [], [], {}
    for k, (fit, held) in enumerate(outer):
        for arm in ARMS:
            models, scalers, threshold, h = fitted_arm(features, y, fit, inner[k], held, arm, f'outer{k}', selections, convergence)
            for j, model in enumerate(models): scores[arm][held, j] = model.predict_proba(h)[:, 1]
            thresholds[arm][held] = threshold
            print('E5 outer', k, arm, 'complete', flush=True)
            write_json(out / 'progress.json', {'stage': 'TRAIN_CV', 'outer_fold': k, 'arm': arm, 'timestamp': now()})
    # No validation score participates in any choice; fit all arms even if OOF winner differs.
    fit, held = np.arange(nt), np.arange(nt, len(rows))
    for arm in ARMS:
        models, scalers, threshold, h = fitted_arm(features, y, fit, inner[-1], held, arm, 'full_train', selections, convergence)
        final_models[arm] = (models, scalers, threshold, h)
    write_csv(out / 'e5_train_only_selection.csv', selections)
    write_csv(out / 'convergence_log.csv', convergence,
              fields=['partition', 'arm', 'finding', 'C', 'n_iter', 'warnings', 'stage'])
    with (out / 'frozen_dev_models.pkl').open('wb') as stream:
        pickle.dump({arm: value[:3] for arm, value in final_models.items()}, stream)
    write_json(out / 'validation_unlock_manifest.json', {'timestamp': now(), 'selection_sha256': sha(out / 'e5_train_only_selection.csv'),
               'models_sha256': sha(out / 'frozen_dev_models.pkl'), 'protocol_sha256': sha(out / 'e5_protocol.yaml'),
               'rule': 'all arms trained/selected on TRAIN; original validation scored once now; no test accessed'})
    for arm, (models, scalers, threshold, h) in final_models.items():
        for j, model in enumerate(models): scores[arm][held, j] = model.predict_proba(h)[:, 1]
        thresholds[arm][held] = threshold
        assert np.isfinite(scores[arm]).all()
    fold_id = np.full(len(rows), -1, int)
    for k, (_, ix) in enumerate(outer): fold_id[ix] = k
    predictions = []
    for i, r in enumerate(rows):
        for arm in ARMS:
            for j, label in enumerate(LABELS):
                predictions.append(dict(image_id=r['image_id'], patient_id=r['patient_id'], source_sha256=alignment[i]['source_sha256'], partition='train_OOF' if i < nt else 'validation_DEV', outer_fold=int(fold_id[i]), arm=arm, finding=label, GT=int(y[i, j]), score=float(scores[arm][i, j]), threshold=float(thresholds[arm][i, j]), prediction=int(scores[arm][i, j] >= thresholds[arm][i, j])))
    write_csv(out / 'e5_dev_predictions.csv', predictions)
    class_rows, macro_rows = [], []
    partitions = [('train_OOF', np.arange(nt)), ('validation_DEV', np.arange(nt, len(rows)))] + [(f'outer{k}', ix) for k, (_, ix) in enumerate(outer)]
    for partition, ix in partitions:
        for arm in ARMS:
            values = []
            for j, label in enumerate(LABELS):
                v = metrics(y[ix, j], scores[arm][ix, j], scores[arm][ix, j] >= thresholds[arm][ix, j]); values.append(v)
                class_rows.append(dict(partition=partition, arm=arm, finding=label, positive=int(y[ix, j].sum()), negative=int(len(ix) - y[ix, j].sum()), positive_patients=len(set(groups[ix][y[ix, j] == 1])), patients=len(set(groups[ix])), **v, limitation=LIMITATION))
            macro_rows.append(dict(partition=partition, arm=arm, images=len(ix), patients=len(set(groups[ix])), labels=13,
                                   **{key: float(np.mean([v[key] for v in values])) for key in ('AP', 'AUROC', 'F1')}, limitation=LIMITATION))
    write_csv(out / 'e5_per_class_metrics.csv', class_rows)
    write_csv(out / 'e5_macro_comparison.csv', macro_rows)
    increments, error_patterns = [], []
    for partition, ix in partitions:
        selected = {r['arm']: r for r in macro_rows if r['partition'] == partition}
        best = max(('A', 'B'), key=lambda arm: selected[arm]['AP'])
        increments.append(dict(partition=partition, finding='MACRO', best_single=best, delta_AP=selected['A+B']['AP'] - selected[best]['AP'], limitation=LIMITATION))
        for label in LABELS:
            values = {r['arm']: r for r in class_rows if r['partition'] == partition and r['finding'] == label}
            best = max(('A', 'B'), key=lambda arm: values[arm]['AP'])
            increments.append(dict(partition=partition, finding=label, best_single=best, delta_AP=values['A+B']['AP'] - values[best]['AP'], limitation=LIMITATION))
    for j, label in enumerate(LABELS):
        a = scores['A'][:nt, j] >= thresholds['A'][:nt, j]
        b = scores['B'][:nt, j] >= thresholds['B'][:nt, j]
        truth = y[:nt, j].astype(bool)
        error_patterns.append(dict(finding=label, images=nt, patients=1222, disagreements=int((a != b).sum()), both_correct=int(((a == truth) & (b == truth)).sum()), A_only_correct=int(((a == truth) & (b != truth)).sum()), B_only_correct=int(((a != truth) & (b == truth)).sum()), both_wrong=int(((a != truth) & (b != truth)).sum()), limitation=LIMITATION))
    write_csv(out / 'e5_oof_error_patterns.csv', error_patterns)
    rng = np.random.default_rng(20261015); unique = np.unique(groups[:nt]); by_patient = {p: np.flatnonzero(groups[:nt] == p) for p in unique}
    macro_boot, class_boot = [], {label: [] for label in LABELS}
    for repeat in range(500):
        ix = np.concatenate([by_patient[p] for p in rng.choice(unique, len(unique), replace=True)])
        aps = {arm: np.array([average_precision_score(y[ix, j], scores[arm][ix, j]) if len(np.unique(y[ix, j])) == 2 else np.nan for j in range(13)]) for arm in ARMS}
        for j, label in enumerate(LABELS): class_boot[label].append(aps['A+B'][j] - max(aps['A'][j], aps['B'][j]))
        if all(np.isfinite(a).all() for a in aps.values()): macro_boot.append(float(aps['A+B'].mean() - max(aps['A'].mean(), aps['B'].mean())))
        if (repeat + 1) % 100 == 0: print('E5 bootstrap', repeat + 1, flush=True)
    for r in increments:
        if r['partition'] != 'train_OOF': continue
        draws = np.asarray(macro_boot if r['finding'] == 'MACRO' else class_boot[r['finding']]); draws = draws[np.isfinite(draws)]
        r['CI_low'], r['CI_high'] = map(float, np.percentile(draws, [2.5, 97.5])) if len(draws) else (float('nan'), float('nan'))
        r['bootstrap_valid'] = len(draws)
    write_csv(out / 'e5_incremental_benefit.csv', increments,
              fields=['partition', 'finding', 'best_single', 'delta_AP', 'limitation', 'CI_low', 'CI_high', 'bootstrap_valid'])
    macro = next(r for r in increments if r['partition'] == 'train_OOF' and r['finding'] == 'MACRO')
    fold_deltas = [r['delta_AP'] for r in increments if r['partition'].startswith('outer') and r['finding'] == 'MACRO']
    dev_delta = next(r['delta_AP'] for r in increments if r['partition'] == 'validation_DEV' and r['finding'] == 'MACRO')
    if macro['CI_low'] > 0 and all(d > 0 for d in fold_deltas) and dev_delta > 0:
        decision = 'GO_PROPOSE_DUAL_ENCODER_PILOT'
    elif macro['CI_high'] < 0 and all(d < 0 for d in fold_deltas) and dev_delta < 0:
        decision = 'NO_GO_FUSION_FOR_NOW'
    else: decision = 'HOLD_COMPLEX_FUSION'
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, partition in zip(axes, ('train_OOF', 'validation_DEV')):
        values = [r for r in macro_rows if r['partition'] == partition]
        ax.bar([r['arm'] for r in values], [r['AP'] for r in values]); ax.set_title(partition); ax.set_ylabel('Macro sklearn AP')
    fig.suptitle(LIMITATION + '\nNIH TRAIN-CV/DEV, EXPLORATORY', fontsize=9); fig.tight_layout(); fig.savefig(out / 'e5_comparison.png', dpi=140); plt.close(fig)
    write_json(out / 'results.json', {'macro_comparison': macro_rows, 'primary_increment': macro,
               'fold_AP_deltas': fold_deltas, 'secondary_DEV_AP_delta': dev_delta,
               'decision': decision, 'bootstrap_note': 'Conditional on fitted OOF models; no refitting/hyperparameter uncertainty; max comparator recomputed per draw.'})
    # Feature-only evidence: source pixels lie outside the currently authorized project root.
    cases = []
    for i in np.argsort(np.max(np.abs(scores['A+B'][:nt] - scores['B'][:nt]), axis=1))[::-1][:12]:
        j = int(np.argmax(np.abs(scores['A+B'][i] - scores['B'][i])))
        cases.append(dict(image_id=rows[i]['image_id'], patient_id=groups[i], source_sha256=alignment[i]['source_sha256'], finding=LABELS[j], GT=int(y[i, j]), B_score=float(scores['B'][i, j]), fusion_score=float(scores['A+B'][i, j]), evidence='FEATURE_PREDICTION_ONLY_NO_PIXELS_READ', clinical_review='NOT_PERFORMED'))
    write_csv(out / 'e5_feature_evidence.csv', cases)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['g0', 'e13', 'e5'], required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--gate')
    args = parser.parse_args(); out = inside(args.output); out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'results.json').exists(), 'Never overwrite a completed analysis'
    with threadpool_limits(limits=4):
        if args.phase == 'g0': preflight(out)
        elif args.phase == 'e13': e13(out, verify_gate(args.gate))
        else: e5(out, verify_gate(args.gate))
    assert not any(name in sys.modules for name in ('torch', 'cupy', 'transformers')), 'Unexpected model/GPU framework import'
    print('PHASE_COMPLETE', args.phase, 'RSS_GiB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, flush=True)


if __name__ == '__main__': main()
