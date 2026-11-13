"""Independent E1.3 gate/analysis and E5 cost profiling; no E5 benchmark entrypoint."""
import argparse
import json
import os
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.common import ROOT, RESEARCH, now, sha, write_csv, write_json
from research.v3_cached import (LIMITATION, LABELS, ARMS, C_GRID, inside, load_dev,
                               folds_for, classifier, cka, scale_blocks)


def train_data(gate):
    gate = inside(gate)
    old = json.loads((gate / 'input_manifest.json').read_text())
    rows, features, groups, y, nt, alignment, inputs = load_dev()
    assert inputs == old, 'Original cache/manifests changed'
    rows, groups, y, alignment = rows[:nt], groups[:nt], y[:nt], alignment[:nt]
    features = {encoder: {rep: array[:nt] for rep, array in block.items()}
                for encoder, block in features.items()}
    outer, inner = folds_for(groups)
    original = {r['image_id']: r for r in __import__('csv').DictReader((gate / 'fold_manifest.csv').open())}
    for k, (fit, held) in enumerate(outer):
        assert not set(groups[fit]) & set(groups[held])
        for i in held: assert int(original[rows[i]['image_id']]['outer_fold']) == k
    for k, (fit, select) in enumerate(inner):
        for role, ix in [('fit', fit), ('select', select)]:
            for i in ix: assert original[rows[i]['image_id']][f'inner{k}_role'] == role
    return rows, features, groups, y, alignment, inputs, outer, inner


def preflight(out, gate):
    rows, features, groups, y, alignment, inputs, outer, inner = train_data(gate)
    write_json(out / 'protocol.json', {'scope': 'independent E1.3 fresh preflight only',
               'old_gate': str(gate), 'E5_status': 'HOLD_NO_BENCHMARK_AUTHORIZATION',
               'E13_protocol_sha256': sha(gate / 'e13_protocol.yaml'),
               'E5_protocol_sha256': sha(gate / 'e5_protocol.yaml')})
    x = StandardScaler().fit_transform(features['cxr']['patchmean'])
    target = StandardScaler().fit_transform(features['clear']['patchmean'])
    start = time.perf_counter(); Ridge(alpha=100., solver='cholesky').fit(x, target)
    ridge_sec = time.perf_counter() - start
    start = time.perf_counter(); _ = x.T @ x; gram_sec = time.perf_counter() - start
    # 12 ridge fits; extra full/patient/fold CKA Grams and 500 conditional bootstrap draws.
    wall_estimate = 12 * ridge_sec * 4 + 60 * gram_sec * 4 + 180
    ram_available = next(int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')) / 1024**2
    reserve = out / 'allocation_probe.bin'
    with reserve.open('wb') as stream: os.posix_fallocate(stream.fileno(), 0, 64 * 1024**2)
    reserve.unlink()
    result = {'E13_estimated_wall_sec': wall_estimate, 'E13_estimated_peak_RSS_GiB': 2,
              'observed_preflight_peak_RSS_GiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
              'ridge_full_train_timing_sec': ridge_sec, 'Gram_timing_sec': gram_sec,
              'available_RAM_GiB': ram_available, 'allocation_probe_MiB': 64,
              'quota_note': 'Numeric account quota unavailable; bounded allocation PASS, not a quota guarantee.',
              'E13_GO': wall_estimate <= 3600 and ram_available >= 2,
              'E5_GATE': 'INDEPENDENT_HOLD_PENDING_COST_PROFILING_AND_EXPLICIT_BENCHMARK_AUTHORIZATION',
              'train_images': len(rows), 'train_patients': len(set(groups)),
              'input_manifest': inputs, 'source_data_scope': 'TRAIN-only calculations; cached member transport/hash includes immutable full cache; no original test evaluation'}
    write_json(out / 'resource_preflight.json', result)
    assert result['E13_GO'], 'Independent E13 resource gate failed'
    print(json.dumps(result), flush=True)


def e13(out, gate, fresh):
    assert json.loads((fresh / 'resource_preflight.json').read_text())['E13_GO']
    rows, features, groups, y, alignment, inputs, outer, inner = train_data(gate)
    shutil.copyfile(gate / 'e13_protocol.yaml', out / 'e13_protocol.yaml')
    write_json(out / 'uncertainty_protocol.json', {'frozen_at': now(), 'bootstrap_unit': 'TRAIN patient',
               'draws': 500, 'seed': 20261015, 'R2': 'variance-weighted original target space; SSE and target moments aggregated by patient',
               'CI': 'percentile95; conditional on fitted OOF models; no ridge refitting',
               'CKA_uncertainty': 'one-image-per-patient and three heldout-fold sensitivity; no inferential CKA CI',
               'selection': 'ridge alpha100 fixed, no label tuning; no E5 performance evaluation'})
    write_csv(out / 'e13_sample_alignment.csv', alignment)
    similarities, cross, residuals = [], [], []
    unique = np.unique(groups)
    patient_indices = [np.flatnonzero(groups == p) for p in unique]
    first = np.array([ix[0] for ix in patient_indices])
    rng = np.random.default_rng(20261015)
    counts = np.stack([np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique)) for _ in range(500)]).astype(np.float64)
    for rep in ('cls', 'patchmean'):
        for sample, ix in [('all_train_images', np.arange(len(rows))), ('one_image_per_patient', first)] + [(f'outer{k}', held) for k, (_, held) in enumerate(outer)]:
            value = cka(features['cxr'][rep][ix], features['clear'][rep][ix])
            assert np.isfinite(value) and -1e-12 <= value <= 1 + 1e-12
            similarities.append(dict(representation=rep, sample=sample, images=len(ix), patients=len(set(groups[ix])), linear_CKA=value, limitation=LIMITATION))
        for source, target in [('cxr', 'clear'), ('clear', 'cxr')]:
            x, target_values = features[source][rep], features[target][rep]
            predictions = np.empty_like(target_values)
            for k, (fit, held) in enumerate(outer):
                sx, sy = StandardScaler().fit(x[fit]), StandardScaler().fit(target_values[fit])
                model = Ridge(alpha=100., solver='cholesky').fit(sx.transform(x[fit]), sy.transform(target_values[fit]))
                pred = sy.inverse_transform(model.predict(sx.transform(x[held])))
                assert np.isfinite(pred).all(); predictions[held] = pred
                value = float(r2_score(target_values[held], pred, multioutput='variance_weighted'))
                cross.append(dict(representation=rep, direction=f'{source}_to_{target}', partition=f'outer{k}', images=len(held), patients=len(set(groups[held])), R2_variance_weighted=value, limitation=LIMITATION))
                for i, loss in zip(held, np.sum((pred - target_values[held])**2, axis=1)):
                    residuals.append(dict(image_id=rows[i]['image_id'], patient_id=groups[i], source_sha256=alignment[i]['source_sha256'], representation=rep, direction=f'{source}_to_{target}', outer_fold=k, squared_error=float(loss)))
                print('E13', rep, source, 'fold', k, 'done', flush=True)
            error = np.sum((predictions - target_values)**2, axis=1)
            patient_error = np.array([error[ix].sum() for ix in patient_indices])
            patient_sum = np.stack([target_values[ix].sum(0) for ix in patient_indices])
            patient_squares = np.array([np.sum(target_values[ix]**2) for ix in patient_indices])
            patient_size = np.array([len(ix) for ix in patient_indices])
            total_sum = counts @ patient_sum
            denominator = counts @ patient_squares - np.sum(total_sum**2, axis=1) / (counts @ patient_size)
            draws = 1 - (counts @ patient_error) / denominator
            assert np.isfinite(draws).all() and (denominator > 0).all()
            point = float(r2_score(target_values, predictions, multioutput='variance_weighted'))
            algebraic = 1 - error.sum() / np.sum((target_values - target_values.mean(0))**2)
            assert abs(point - algebraic) < 1e-10, 'R2 sufficient-statistic sanity check failed'
            cross.append(dict(representation=rep, direction=f'{source}_to_{target}', partition='pooled_OOF', images=len(rows), patients=len(unique), R2_variance_weighted=point, CI_low=float(np.percentile(draws, 2.5)), CI_high=float(np.percentile(draws, 97.5)), bootstrap_valid=len(draws), limitation=LIMITATION))
    write_csv(out / 'e13_similarity.csv', similarities)
    write_csv(out / 'e13_cross_prediction.csv', cross, fields=['representation', 'direction', 'partition', 'images', 'patients', 'R2_variance_weighted', 'CI_low', 'CI_high', 'bootstrap_valid', 'limitation'])
    write_csv(out / 'e13_oof_residuals.csv', residuals)
    full = [r for r in similarities if r['sample'] == 'all_train_images']
    pooled = [r for r in cross if r['partition'] == 'pooled_OOF']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar([r['representation'] for r in full], [r['linear_CKA'] for r in full]); axes[0].set_ylabel('TRAIN centered linear CKA')
    labels = [r['representation'] + '/' + r['direction'] for r in pooled]
    values = np.array([r['R2_variance_weighted'] for r in pooled])
    axes[1].bar(labels, values)
    axes[1].errorbar(np.arange(4), values, yerr=[np.maximum(0, values - np.array([r['CI_low'] for r in pooled])), np.maximum(0, np.array([r['CI_high'] for r in pooled]) - values)], fmt='none', color='black', capsize=4)
    axes[1].set_ylabel('Patient-fold OOF R² / conditional95%CI'); axes[1].tick_params(axis='x', rotation=25)
    fig.suptitle(LIMITATION + '\nTRAIN only; geometry is not predictive complementarity', fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'e13_similarity.png', dpi=150); plt.close(fig)
    write_json(out / 'results.json', {'similarities': similarities, 'cross_prediction': pooled,
               'observation': 'DESCRIPTIVE_GEOMETRY_ONLY', 'complementarity': 'COMPLEMENTARITY_NOT_YET_PROVEN',
               'images': len(rows), 'patients': len(unique), 'folds': 3})


def profile(out, gate):
    start = time.perf_counter()
    rows, features, groups, y, alignment, inputs, outer, inner = train_data(gate)
    # Representative sizes fixed by original fold sizes, not scores or timing outcomes.
    outer_typical = sorted(range(3), key=lambda k: len(inner[k][0]))[1]
    outer_fit = outer[outer_typical][0]
    scenarios = {'outer_inner': inner[outer_typical][0], 'full_train_inner': inner[-1][0],
                 'outer_refit': outer_fit, 'full_train_refit': np.arange(len(rows))}
    write_json(out / 'profiling_protocol.json', {'written_at': now(), 'benchmark_authorized': False,
               'TRAIN_only': True, 'no_predict_calls': True, 'no_performance_metrics': True,
               'original_E5_protocol_sha256': sha(gate / 'e5_protocol.yaml'),
               'arms': ARMS, 'C_grid': C_GRID, 'common_label': 'Atelectasis', 'rare_label': 'Edema',
               'profile_fits': 36, 'future_benchmark_fits': 780,
               'scenarios': {k: {'images': len(ix), 'patients': len(set(groups[ix]))} for k, ix in scenarios.items()},
               'projection': 'arm/C-specific measured inner costs; scale outer sizes linearly; refit C1 adjusted by observed C timing ratio; conservative rare-class factor; safety1.5 +300s overhead',
               'cap': {'wall_sec': 600, 'RSS_GiB': 8}, 'partial_profile_policy': 'HOLD if incomplete; do not fill unmeasured cells'})
    support, measured = [], []
    for scenario, ix in scenarios.items():
        for j, label in enumerate(LABELS):
            positive = y[ix, j] == 1
            support.append(dict(scenario=scenario, finding=label, images=len(ix), patients=len(set(groups[ix])), positive=int(positive.sum()), negative=int((~positive).sum()), positive_patients=len(set(groups[ix][positive]))))
            assert positive.any() and (~positive).any(), f'Unsupported profile class: {scenario}/{label}'
    write_csv(out / 'class_support_constraints.csv', support)
    for scenario, ix in scenarios.items():
        for arm in ARMS:
            x, _, _ = scale_blocks(features, arm, ix, np.arange(1))
            jobs = [('Atelectasis', c) for c in (C_GRID if scenario.endswith('inner') else [1.])]
            if scenario.endswith('inner'): jobs += [('Edema', 1.)]
            for label, c in jobs:
                if time.perf_counter() - start > 540:
                    write_csv(out / 'fit_timings.csv', measured)
                    write_json(out / 'results.json', {'gate': 'HOLD_INCOMPLETE_PROFILE', 'fits': len(measured), 'E5_executed': False})
                    return
                j = LABELS.index(label)
                clock = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always'); model = classifier(c).fit(x, y[ix, j])
                elapsed = time.perf_counter() - clock
                assert np.isfinite(model.coef_).all() and not caught, f'Invalid/convergence fit: {scenario}/{arm}/{label}/{c}'
                measured.append(dict(scenario=scenario, arm=arm, finding=label, C=c, images=len(ix), patients=len(set(groups[ix])), width=x.shape[1], fit_sec=elapsed, n_iter=int(model.n_iter_.max()), positive=int(y[ix, j].sum()), warnings=''))
                write_csv(out / 'fit_timings.csv', measured)
                print('PROFILE', len(measured), '/36', scenario, arm, label, c, elapsed, flush=True)
    assert len(measured) == 36
    lookup = {(r['scenario'], r['arm'], r['finding'], r['C']): r['fit_sec'] for r in measured}
    costs = []
    for arm in ARMS:
        rare_factor = max(1., *(lookup[s, arm, 'Edema', 1.] / lookup[s, arm, 'Atelectasis', 1.] for s in ('outer_inner', 'full_train_inner')))
        c_refit_factor = max(1., *(lookup[s, arm, 'Atelectasis', c] / lookup[s, arm, 'Atelectasis', 1.] for s in ('outer_inner', 'full_train_inner') for c in C_GRID))
        for k in range(4):
            scenario = 'outer_inner' if k < 3 else 'full_train_inner'
            ratio = len(inner[k][0]) / len(scenarios[scenario])
            candidates = 13 * sum(lookup[scenario, arm, 'Atelectasis', c] for c in C_GRID) * ratio
            refit_scenario = 'outer_refit' if k < 3 else 'full_train_refit'
            refit_size = len(outer[k][0]) if k < 3 else len(rows)
            refits = 13 * lookup[refit_scenario, arm, 'Atelectasis', 1.] * refit_size / len(scenarios[refit_scenario]) * c_refit_factor
            costs.append(dict(arm=arm, partition=f'outer{k}' if k < 3 else 'full_train', candidate_fits=52, refit_fits=13, rare_class_factor=rare_factor, C_refit_factor=c_refit_factor, raw_cost_sec=candidates + refits, adjusted_cost_sec=(candidates + refits) * rare_factor))
    write_csv(out / 'cost_projection.csv', costs)
    estimated = sum(r['adjusted_cost_sec'] for r in costs) * 1.5 + 300
    write_json(out / 'results.json', {'gate': 'E5_RESOURCE_GATE_ELIGIBLE' if estimated <= 7200 else 'HOLD_RESOURCE_ESTIMATE',
               'estimated_E5_wall_sec': estimated, 'estimated_E5_minutes': estimated / 60,
               'margin_factor': 1.5, 'overhead_sec': 300, 'profile_fits': 36,
               'projected_fits': sum(r['candidate_fits'] + r['refit_fits'] for r in costs),
               'cost_by_arm_sec_before_global_margin': {arm: sum(r['adjusted_cost_sec'] for r in costs if r['arm'] == arm) for arm in ARMS},
               'E5_executed': False, 'E5_modeling_protocol_changed': False,
               'decision_scope': 'resource eligibility only; E5 remains HOLD_FOR_NEW_EXPLICIT_AUTHORIZATION',
               'limitations': 'Timing variability, unprofiled label costs and shared-host load remain uncertain; wall guard120min still required for any authorized benchmark.'})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['preflight', 'e13', 'profile'], required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--gate', required=True); parser.add_argument('--fresh')
    args = parser.parse_args(); out, gate = inside(args.output), inside(args.gate)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out.mkdir(parents=True, exist_ok=True); assert not (out / 'results.json').exists()
    with threadpool_limits(limits=4):
        if args.phase == 'preflight': preflight(out, gate)
        elif args.phase == 'e13': e13(out, gate, inside(args.fresh))
        else: profile(out, gate)
    assert not any(k in sys.modules for k in ('torch', 'cupy', 'transformers', 'jax'))
    print('CPU_ONLY_PHASE_COMPLETE', args.phase, flush=True)


if __name__ == '__main__': main()
