"""Binary classification metrics from continuous scores."""

import math


def validate_scores(values):
    values = list(values)
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('Scores must be finite')
    if set(values).issubset({0.0, 1.0}):
        raise ValueError('AUROC/AUPRC require continuous scores, not binary predictions')
    return values


def auroc(labels, scores):
    scores = validate_scores(scores)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(zip(scores, labels))
    positive_rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        positive_rank_sum += avg_rank * sum(y for _, y in ranked[i:j])
        i = j
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels, scores):
    scores = validate_scores(scores)
    positives = sum(labels)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    area = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        group = ranked[i:j]
        new_tp = tp + sum(y for _, y in group)
        new_fp = fp + len(group) - sum(y for _, y in group)
        area += (new_tp - tp) / positives * (new_tp / (new_tp + new_fp))
        tp, fp = new_tp, new_fp
        i = j
    return area


def threshold_metrics(labels, scores, threshold):
    validate_scores(scores)
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must lie in [0,1]')
    predictions = [score >= threshold for score in scores]
    tp = sum(y == 1 and p for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p for y, p in zip(labels, predictions))
    fn = sum(y == 1 and not p for y, p in zip(labels, predictions))
    tn = sum(y == 0 and not p for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': precision,
            'recall': recall, 'f1': f1, 'positive_support': tp + fn,
            'negative_support': tn + fp}
