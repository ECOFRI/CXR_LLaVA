"""Write a template-structured analysis from completed Indiana full-run artifacts."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def fmt(value, digits=4):
    if value in ('', None, 'None'):
        return 'undefined'
    return f'{float(value):.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--gt-labeling', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, selection, gt, output = [p.resolve() for p in
                                  (args.run, args.selection, args.gt_labeling, args.output)]
    if output.exists():
        raise FileExistsError(f'Do not overwrite an existing experiment report: {output}')
    if (run / 'status.txt').read_text().strip() != 'SUCCESS':
        raise ValueError('Run has not been marked SUCCESS')
    summary = json.loads((run / 'metrics/summary.json').read_text())
    if summary['generated_success'] != summary['cohort_n']:
        raise ValueError('Full report requires complete inference; write a partial report separately')
    metrics = read_csv(run / 'metrics/pathology_metrics.csv')
    support = read_csv(gt / 'support.csv')
    support_by_name = {row['pathology']: row for row in support}
    selection_config = json.loads((selection / 'selection/cohort_config.json').read_text())
    inference_config = json.loads((run / 'inference/run_config.json').read_text())
    gt_provenance = json.loads((gt / 'provenance.json').read_text())
    source = json.loads((selection / 'review/source_provenance.json').read_text())
    if (selection_config['cohort_sha256'] != summary['cohort_sha256']
            or gt_provenance['cohort_sha256'] != summary['cohort_sha256']
            or inference_config['cohort_sha256'] != summary['cohort_sha256']):
        raise ValueError('Selection, GT, inference and metrics cohort hashes differ')
    comparison = [
        '| Bệnh lý | GT +/− | Joint coverage | P / R / F1 local | 95% CI local | F1 paper (95% CI) | Delta |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in metrics:
        gt_row = support_by_name[row['pathology']]
        comparison.append(
            f"| {row['pathology']} | {gt_row['positive']}/{gt_row['negative']} | "
            f"{row['joint_definite']}/{row['n_cohort']} ({fmt(row['coverage_all'], 3)}) | "
            f"{fmt(row['precision'], 3)} / {fmt(row['recall'], 3)} / {fmt(row['f1'], 3)} | "
            f"({fmt(row['ci95_low'], 3)}, {fmt(row['ci95_high'], 3)}) | "
            f"{fmt(row['paper_table4_f1'], 2)} ({fmt(row['paper_table4_ci95_low'], 2)}, "
            f"{fmt(row['paper_table4_ci95_high'], 2)}) | {fmt(row['delta_local_minus_paper'], 3)} |"
        )
    omit = [
        '| Bệnh lý | TP | FP | TN | FN | GT+ bị bỏ vì generation failed | GT+ pred uncertain | GT+ pred unmentioned | Bootstrap undefined |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in metrics:
        omit.append(f"| {row['pathology']} | {row['tp']} | {row['fp']} | {row['tn']} | {row['fn']} | "
                    f"{row['positive_generation_failed']} | {row['positive_pred_uncertain']} | "
                    f"{row['positive_pred_unmentioned']} | {row['bootstrap_undefined']} |")
    repo_status_count = len(inference_config.get('git_status', '').splitlines())
    report = f'''# Indiana full cohort K80 analysis — reconstructed CXR-LLaVA evaluation

## 1. Experiment Information

- **Experiment ID:** `{run.name}`
- **Date:** {datetime.now().date().isoformat()}
- **Researcher:** student4 / project team
- **Project:** CXR_LLaVA_Improvement
- **Repository:** https://github.com/hanhpm/CXR_LLaVA_Improvement
- **Branch:** master
- **Commit:** `{inference_config.get('git_commit', 'unknown')}`
- **Notebook / Script:** `scripts/run_indiana_full_k80.sh`, `scripts/k80_full_inference.py`, `scripts/evaluate_indiana_full.py`
- **Status:** completed reconstructed-cohort evaluation; research-only

## 2. Objective

Đánh giá checkpoint phát hành ECOFRI/CXR-LLAVA-v2 bằng report generation trên cohort Indiana/OpenI frontal dựng lại từ dữ liệu local; CheXpert gán nhãn cả reference và report sinh ra rồi tính precision/recall/F1 theo từng bệnh lý.

## 3. Survey / Technical Motivation

- Paper v3/Table 4: https://arxiv.org/pdf/2310.18341v3
- NLM image-level frontal/lateral lists: {source['source_page']}
- Checkpoint: https://huggingface.co/ECOFRI/CXR-LLAVA-v2
- Official CheXpert labeler: https://github.com/stanfordmlgroup/chexpert-labeler

NLM ghi rõ nhãn view được tạo bằng việc xem từng ảnh. Cohort này dùng nhãn đó thay cho đoán suffix/caption; paper không công bố manifest ID hoặc cách tính average F1 chi tiết trong artifact đang có.

## 4. Environment

| Component | Configuration |
|---|---|
| GPU | four Tesla K80 selected by `CUDA_VISIBLE_DEVICES` |
| Torch / Transformers | 1.12.1+cu102 / 4.36.2 |
| Model | ECOFRI/CXR-LLAVA-v2, revision `{inference_config['model_revision']}` |
| Precision | FP32 across four GPUs; no quantization |
| CheXpert / NegBio | `{gt_provenance['labeler_revision']}` / `{gt_provenance['negbio_revision']}` |

Package freezes and exact source/checkpoint hashes are under `{run.name}/snapshots/` and `{run.name}/inference/run_config.json`.

## 5. Dependency Setup

Existing `cxr-llava-k80` and `chexpert-label` environments were reused. The pinned official sample gate passed before GT and generated labeling. Reproduce with `bash scripts/run_indiana_full_k80.sh --check` followed by `bash scripts/run_indiana_full_k80.sh --background` from the project root.

## 6. Code / Notebook Changes

`scripts/inventory_indiana.py` inventoried every XML/image; `scripts/select_indiana_nlm_views.py` matched the NLM labels and selected one frontal image/report; `scripts/label_indiana_gt.py` labeled references; `scripts/k80_full_inference.py` saved atomic per-case checkpoints and independent per-case RNG seeds; `scripts/evaluate_indiana_full.py` aligned labels and calculated metrics. The inference runner retains the K80 FP32 compatibility adaptations from the 50-case run.

## 7. Experiment Configuration

| Parameter | Value |
|---|---|
| XML reports / local images | {selection_config['n_xml_reports']} / {selection_config['n_images']} |
| Official frontal candidates / selected reports | {selection_config['n_frontal_candidates']} / {selection_config['n_cohort']} |
| NLM label conflicts / unlisted local images | {len(source['local_conflict_ids'])} / {len(source['local_unlisted_ids'])} |
| Selection | one frontal/report, lexicographically smallest image ID |
| Reference | original FINDINGS + IMPRESSION when available |
| Precision / batch | FP32 / 1 |
| Generation | official report helper defaults temperature 0.2, top_p 0.8 |
| Seed | {inference_config['seed']}, per-case SHA-256 derived; differs from the earlier global RNG stream |
| Bootstrap | {summary['bootstrap_iterations']} study-level resamples, seed {summary['bootstrap_seed']} |

## 8. Code Executed

```bash
bash scripts/run_indiana_full_k80.sh --check
bash scripts/run_indiana_full_k80.sh --background
```

The run performed a 10-case gate, stopped after one case, resumed without rewriting it, then generated all {summary['cohort_n']} reports.

## 9. Output

- Full pipeline status: `{run.name}/status.txt`.
- Raw per-case predictions and logs: `{run.name}/inference/`.
- Official labels and metrics: `{run.name}/metrics/`.
- GT-only labels and support: `{gt}`.
- Review queue: `{run.name}/metrics/review_queue.csv`.

Warnings and errors remain in the stage logs. Generated text is unverified model output and has research use only.

## 10. Output Analysis

The sample gates and GT/generated row, ID and text alignment passed. Completion was {summary['generated_success']}/{summary['cohort_n']}. Direct image/report links come from XML and image ID prefix agreement; a new independent visual report-pairing review was not performed. NLM view labels are external human annotations, and eight local images with absent/conflicting labels were excluded.

## 11. Quantitative Results

Macro F1 over {summary['f1_pathologies_defined']}/7 defined pathologies: {fmt(summary['macro_f1_defined_only'])}. Mean latency: {fmt(summary['mean_latency_sec_per_success'], 2)} sec/success. Peak allocated VRAM on the most-used device: {fmt(summary['peak_vram_gb_max_device'], 2)} GiB.

{chr(10).join(comparison)}

Joint-definite F1 uses only pairs where GT and generated labels are both 0 or 1; -1/blank are preserved and excluded. Coverage and omissions expose the denominator. The paper column is a descriptive reference, not a matched-cohort significance comparison.

{chr(10).join(omit)}

Bootstrap resamples complete report/study rows then reapplies each pathology's joint-definite mask. Undefined replicates are counted. The paper's exact CI aggregation and average F1 calculation remain unverified.

## 12. Qualitative Results

Individual image/reference/prediction findings need review using `{run.name}/metrics/review_queue.csv`. No report is interpreted here as a verified clinical finding.

## 13. Comparison with Baseline

The earlier 10/50 runs used first existing XML images without view verification. This run uses NLM image-level view labels and one image/report; the resulting metric differences mix cohort and selection changes and cannot be treated as model improvement.

## 14. Problems Encountered

- {len(source['local_conflict_ids'])} local IDs are present in both NLM view lists and {len(source['local_unlisted_ids'])} in neither; all were excluded.
- {selection_config['n_multiple_frontal_reports']} reports have more than one eligible frontal image; deterministic image ID ordering selected one.
- Paper test set has 3,689 pairs, local cohort has {selection_config['n_cohort']}; official ID split and exact average-F1 method are unresolved.

## 15. Conclusion

The released model completed report generation and official CheXpert evaluation on {summary['cohort_n']} Indiana reports selected with NLM image-level frontal labels. Per-pathology F1/CI, support and coverage are recorded with explicit masking and omission counts. The dataset selection, per-case RNG policy, K80 precision adaptation and unresolved paper aggregation limit direct Table 4 comparability. This is a completed reconstructed-cohort evaluation, not exact paper reproduction.

## 16. Next Experiment

- [ ] Review the highest-priority FP/FN and positive omissions with images and original reports.
- [ ] Verify the paper's original case IDs, decoding, masking and average-F1 implementation if source artifacts become available.
- [ ] Investigate NLM conflict/unlisted images separately without altering this frozen cohort.

## 17. Reproducibility Checklist

- [x] Repository, commit, scripts snapshot, environments and checkpoint hashes captured.
- [x] Cohort and source label hashes, exclusions, seed policy and raw outputs captured.
- [x] Official sample gates, GT/generated alignment and bootstrap settings recorded.
- [ ] Independent visual report-pairing review and original paper split/method verified.

Checkout status at run start: {repo_status_count} changed/untracked paths (exact list in `inference/run_config.json`).

## 18. Short Lab Update

> **Objective:** evaluate the released CXR-LLaVA model on a reconstructed Indiana frontal cohort. **Survey:** NLM image-level view labels and paper Table 4. **Implementation:** 4 K80, FP32, frozen XML-linked cohort, atomic resume and official CheXpert labeling. **Experiment:** {summary['generated_success']}/{summary['cohort_n']} reports. **Result:** macro F1 {fmt(summary['macro_f1_defined_only'])} across {summary['f1_pathologies_defined']}/7 defined pathologies, with per-class coverage/CI recorded. **Issue:** paper's exact split/average method and independent pairing review are unresolved. **Next:** review errors and seek original paper method.
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
