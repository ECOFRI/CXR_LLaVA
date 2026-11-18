"""Write the required 18-section analysis from completed E1 artifacts."""
import argparse
import json
from pathlib import Path

from research.common import ROOT, now, sha


def run(args):
    out = Path(args.output)
    dirs = {'cxr': Path(args.cxr), 'clear': Path(args.clear), 'probe': out}
    manifests = {key: json.loads((value / 'run_manifest.json').read_text())
                 for key, value in dirs.items()}
    assert all(m['status'] == 'COMPLETE' for m in manifests.values())
    probe = manifests['probe']
    report = ROOT / 'reports' / f'{now()[:10]}_E1_NIH_exploratory_analysis.md'
    assert not report.exists(), 'Preserve existing analysis report'
    sections = []

    def add(number, title, body):
        sections.append(f'## {number}. {title}\n\n{body}')

    add(1, 'Experiment Information',
        f"E1 frozen-feature matched probes; COMPLETE, exploratory metric-complete evaluation. "
        f"Written {now()}. Repository commit {manifests['cxr']['commit']}. "
        f"Run directories: `{dirs['cxr']}`, `{dirs['clear']}`, `{out}`. "
        'Scripts: research/e1_features.py, e1_job.py, e1_probe.py, e1_review.py, e1_analysis_report.py.')
    add(2, 'Objective', 'Compare native CLS and spatial patch-mean representations with matched '
        'linear probes. No encoder gradients or LLM training. NIH exploratory; potential CXR '
        'encoder pretraining overlap; not independent external test.')
    add(3, 'Survey / Technical Motivation', 'Saved paper crosswalk: '
        '`research/inventory/paper_dataset_crosswalk_20261013.md`. CXR-LLaVA Table 1 includes NIH '
        'in encoder pretraining. Individual local-image membership remains unknown. Native '
        'representations have different dimensions and native preprocessing; the result tests '
        'this complete representation/probe setup, not isolated architecture superiority.')
    add(4, 'Environment', '\n\n'.join(
        f"{k}: `{manifests[k]['environment']}`, PyTorch {manifests[k]['torch']}, "
        f"FP32, batch 1; {'K80 GPU0' if k == 'cxr' else 'CPU only'}."
        for k in ('cxr', 'clear')) + '\n\nCPU process versions are saved in environment manifests; '
        'the completed system inventory is reused. Hardware differs, so efficiency ranking is unsupported.')
    add(5, 'Dependency Setup', 'Existing stable cxr-llava-k80 and isolated cxr-clear-cpu were reused. '
        'No installation, base-environment change or in-place dependency update was performed for E1. '
        'CLEAR emits expected CPU xFormers-unavailable warnings; official PyTorch path executes.')
    add(6, 'Code / Notebook Changes', 'Added project research orchestration, provenance checks, '
        'compact caches, matched probes and local evidence/report generation. Original CXR-LLaVA, '
        'CLEAR implementation and checkpoint weights were preserved. Source/checkpoint hashes '
        'are pinned in feature manifests.')
    add(7, 'Experiment Configuration', '4999 local NIH images: train 3783 / 1222 patients, validation '
        '344 / 57 patients, test 872 / 56 patients. Original NIH14 labels matched source metadata '
        'in all 69986 binary comparisons. Hernia excluded by frozen support rule; 13 pathologies '
        'retained. Five supplemental labels and No Finding excluded from primary ontology. '
        'No cross-split image/patient ID overlap. All 4999 encoded source-file hashes are distinct '
        '(saved-hash audit); this does not exclude differently encoded duplicates. '
        'Bounded 300-image pixel screen is not exhaustive '
        'duplicate validation. CXR CLS/patchmean 1024-D; CLEAR 768-D; register tokens excluded '
        'from spatial mean. Native inputs 512 grayscale / 448 RGB. Three warm-up forwards. '
        'Seed42; protocol SHA256 `' + probe['protocol_sha256'] + '`.')
    add(8, 'Code Executed', 'Exact commands: feature/process manifests and `reproduce.sh` in the '
        'probe directory. StandardScaler fit on train only. L2 liblinear logistic regression, '
        'C in [0.001,0.01,0.1,1], highest validation AP; ties lowest C. Threshold selected on '
        'validation F1 over 0.01–0.99; ties closest 0.5 then largest. All models and selections '
        'hashed before test scores were computed once. No CheXpert test tuning.')
    add(9, 'Output', 'Feature caches and test predictions saved; all native shapes and numeric '
        'checks passed. Deterministic endpoint repeats have maximum absolute difference <=1e-6. '
        'Candidate convergence warnings/iteration counts: `convergence_log.csv`. Raw process '
        'logs and measured CPU resource records retained.')
    add(10, 'Output Analysis', 'Finite native features and checkpoint integrity passed. Only CLS '
        'and patchmean were cached; no full patch sequence or 4096-D LLM-projector cache. Test '
        'metrics use the same eligible findings and source cohort for both encoders. Patient '
        'bootstrap resamples preserve repeated images and pairing across encoders.')
    limitation = '**NIH exploratory; potential CXR encoder pretraining overlap; not independent external test.**'
    table = [limitation, '', '| Encoder / representation | Macro AUROC | Macro AP | Macro F1 |',
             '|---|---:|---:|---:|']
    for key, value in probe['summary'].items():
        table.append(f"| {key} | {value['AUROC']:.6f} | {value['AUPRC']:.6f} | {value['F1']:.6f} |")
    table += ['', 'Per-class 95% patient-cluster intervals and paired CLEAR−CXR differences: '
              '`e1_metrics.csv`, `paired_cluster_differences.csv`; 1000 shared bootstrap draws. '
              'One-class ranking draws excluded; valid zero F1 preserved. Undefined denominators '
              'produce NaN, with valid-draw counts saved.', '',
              limitation, '', '| Resource | CXR K80 | CLEAR CPU |', '|---|---:|---:|']
    for label, key in [('Runner wall seconds (includes model load)', 'wall_sec'), ('Mean native forward seconds', 'mean_forward_sec'),
                       ('p95 forward seconds', 'p95_forward_sec'), ('Peak RSS GiB', 'peak_rss_gib'),
                       ('GPU-hours', 'actual_gpu_hours')]:
        table.append(f"| {label} | {manifests['cxr'][key]:.6f} | {manifests['clear'][key]:.6f} |")
    clear_resources = json.loads((dirs['clear'] / 'process_resources.json').read_text())
    table += ['', f"CLEAR full process wall (including imports): {clear_resources['process_wall_sec']:.6f} s; "
              f"CPU user {clear_resources['cpu_user_sec']:.6f} s, system {clear_resources['cpu_system_sec']:.6f} s. "
              'CXR final CPU-seconds/full-parent-process accounting was not recorded; the saved live '
              'observation is not final accounting. Its runner wall and GPU/RSS counters are measured.']
    add(11, 'Quantitative Results', '\n'.join(table))
    add(12, 'Qualitative Results', '`e1_disagreements_contact_sheet.png` displays distinct '
        'CLS prediction disagreements selected by a declared score-gap rule; source SHA256 and '
        'dataset labels are linked in `image_evidence_manifest.csv`. These are dataset-label '
        'disagreements, not verified clinical findings. Expert review NOT PERFORMED. Native '
        'preprocessing sheets are saved separately; shared 32×32 grids do not establish spatial alignment.')
    add(13, 'Comparison with Baseline', 'The matched NIH experiment compares four representation '
        'conditions under one probe protocol. It is not directly comparable to locked CheXpert '
        'report-generation F1. Differences can reflect pretraining, preprocessing, dimension and '
        'probe interaction. K80 versus CPU timings cannot establish which encoder is faster.')
    add(14, 'Problems Encountered', 'Potential CXR pretraining exposure, convenience-subset selection, '
        'AP/PA distribution shift and only 56 test patients limit generalization. Spatial fusion '
        'remains NOT_YET_VALIDATED. Full CLEAR concept E2 remains BLOCKED_CONCEPT_ASSETS_OR_RESOURCE; '
        'direct E2 was completed separately. Storage account quota is unverified; observed resource '
        'bounds are not a quota guarantee.')
    add(15, 'Conclusion', 'E1 completed frozen extraction and matched probe evaluation on the '
        'audited local NIH cohort. Measured metrics are exploratory representation evidence. '
        'They do not establish independent external performance, clinical validity or better '
        'report generation. H1 general superiority remains inconclusive; H2 concept utility '
        'is untested by this experiment.')
    add(16, 'Next Experiment', 'Review per-class uncertainty and image evidence. Identify a '
        'pretraining-independent development/evaluation cohort before stronger claims. Validate '
        'source-coordinate alignment before spatial fusion. Any expensive adapter/LoRA work '
        'requires a compute estimate and explicit approval.')
    add(17, 'Reproducibility Checklist', 'Saved protocol, source cohort, source/checkpoint hashes, '
        'feature caches, environment versions, seeds, validation selections, fitted probes, test '
        'unlock manifest, test predictions, convergence logs, patient bootstrap intervals, resource '
        'records, plots and source-linked image evidence. `reproduce.sh` describes fresh-output '
        'commands; current test predictions must be reused for offline report revisions.')
    add(18, 'Short Lab Update', 'Completed E1 frozen native CLS/patchmean extraction and matched '
        'linear probes with strict train/validation/test separation. E2 direct contrastive '
        'CheXpert evaluation is also complete. GO for reviewing these bounded exploratory '
        'results; HOLD independent superiority claims, full concept pipeline and spatial fusion '
        'until their missing evidence is resolved.')
    report.write_text('# E1 NIH exploratory analysis\n\n' + '\n\n'.join(sections) + '\n')
    print(report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr', required=True)
    parser.add_argument('--clear', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
