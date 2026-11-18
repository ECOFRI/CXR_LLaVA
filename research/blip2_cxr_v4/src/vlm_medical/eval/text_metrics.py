"""CPU text metrics with explicit, dependency-free normalization."""

import math
import re
from collections import Counter

TOKEN_PATTERN = re.compile(r'[a-z0-9]+')


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a, b):
    previous = [0] * (len(b) + 1)
    for word in a:
        current = [0]
        for index, other in enumerate(b, 1):
            current.append(previous[index - 1] + 1 if word == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def case_statistics(reference, generated):
    ref = tokenize(reference)
    hyp = tokenize(generated)
    matches = []
    totals = []
    for n in range(1, 5):
        ref_counts = _ngrams(ref, n)
        hyp_counts = _ngrams(hyp, n)
        matches.append(sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items()))
        totals.append(sum(hyp_counts.values()))
    lcs = _lcs_length(ref, hyp)
    precision = lcs / len(hyp) if hyp else 0.0
    recall = lcs / len(ref) if ref else 0.0
    rouge = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'ref_len': len(ref), 'hyp_len': len(hyp), 'matches': matches,
            'totals': totals, 'rouge_l': rouge}


def corpus_scores(stats):
    if not stats:
        raise ValueError('No valid reports')
    ref_len = sum(row['ref_len'] for row in stats)
    hyp_len = sum(row['hyp_len'] for row in stats)
    matches = [sum(row['matches'][n] for row in stats) for n in range(4)]
    totals = [sum(row['totals'][n] for row in stats) for n in range(4)]
    bp = math.exp(min(0.0, 1.0 - ref_len / hyp_len)) if hyp_len else 0.0
    scores = {}
    for n in range(1, 5):
        ratios = [matches[i] / totals[i] if totals[i] else 0.0 for i in range(n)]
        scores[f'bleu{n}'] = bp * math.exp(sum(math.log(p) for p in ratios) / n) if all(ratios) else 0.0
    scores['rouge_l'] = sum(row['rouge_l'] for row in stats) / len(stats)
    return scores
