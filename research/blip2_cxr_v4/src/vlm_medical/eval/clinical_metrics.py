"""CheXpert label policies and confusion-matrix metrics."""

import math

PATHOLOGIES = ('Cardiomegaly', 'Consolidation', 'Edema', 'Lung Opacity',
               'Pleural Effusion', 'Pneumonia', 'Pneumothorax')
POLICIES = ('historical', 'uncertain_as_positive', 'uncertain_as_negative', 'blank_as_negative')


def label(value, policy):
    if policy not in POLICIES:
        raise ValueError(f'Unknown policy: {policy}')
    value = str(value).strip()
    if value == '':
        return 0 if policy == 'blank_as_negative' else None
    number = float(value)
    if number not in (-1, 0, 1):
        raise ValueError(f'Invalid CheXpert label: {value}')
    if number == -1:
        return 1 if policy == 'uncertain_as_positive' else (0 if policy == 'uncertain_as_negative' else None)
    return int(number)


def confusion(pairs):
    counts = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'paired': 0}
    for ground_truth, predicted in pairs:
        if ground_truth is None or predicted is None:
            continue
        counts['paired'] += 1
        key = ('tp' if predicted == 1 else 'fn') if ground_truth == 1 else ('fp' if predicted == 1 else 'tn')
        counts[key] += 1
    return counts


def enrich(counts, cohort_n):
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else None
    return {**counts, 'precision': precision, 'recall': recall, 'f1': f1,
            'support': tp + fn, 'coverage': counts['paired'] / cohort_n if cohort_n else None}


def summarize(per_class):
    defined = [item['f1'] for item in per_class.values() if item['f1'] is not None]
    tp = sum(item['tp'] for item in per_class.values())
    fp = sum(item['fp'] for item in per_class.values())
    fn = sum(item['fn'] for item in per_class.values())
    denominator = 2 * tp + fp + fn
    return {'macro_f1': sum(defined) / len(defined) if defined else None,
            'micro_f1': 2 * tp / denominator if denominator else None,
            'defined_classes': len(defined), 'classes': len(per_class)}


def require_continuous_scores(scores):
    values = list(scores)
    if not values or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError('AUROC requires finite continuous scores')
    if set(values).issubset({0, 1}):
        raise ValueError('AUROC refuses binary-only predictions')
    return values
