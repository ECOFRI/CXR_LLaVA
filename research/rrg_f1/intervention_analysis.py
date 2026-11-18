"""Aggregate A4 token controls; no ground truth is used to define directions."""
import json
import random
from research.rrg_f1.cohort import BASE, PRIVATE, atomic_json
from research.rrg_f1.labeler import LABELS, label_texts


def main():
    from research.rrg_f1.data import rows_for, read_feature
    # This entry point uses the Qwen CPU env; the labeler itself is the pinned legacy subprocess.
    cases = [(family, i) for family in ('mimic', 'indiana') for i in range(50)]
    scores = [read_feature(f, i, rows_for(f)[i])[1].tolist() for f, i in cases]
    baseline = []
    for family, index in cases:
        position = index if family == 'mimic' else 500 + index
        baseline.append(json.loads((PRIVATE / 'A4/predictions' / ('%04d.json' % position)).read_text()))
    variants = {'baseline': baseline}
    for mode in ('negative', 'shuffled'):
        variants[mode] = [json.loads((PRIVATE / ('A4/intervention_' + mode) / ('%04d.json' % i)).read_text()) for i in range(100)]
    texts, entries = [], []
    labeled = {}
    for mode, rows in variants.items():
        labeled[mode] = [[-2] * 14 for _ in rows]
        for i, row in enumerate(rows):
            if row['generated_report'].strip():
                texts.append(row['generated_report'])
                entries.append((mode, i))
    values = label_texts(texts, PRIVATE / 'intervention_label_chunks', chunk_size=25)
    for (mode, i), row in zip(entries, values):
        labeled[mode][i] = row
    permutation = list(range(100))
    random.Random(42).shuffle(permutation)
    summary = {'status': 'PASS', 'cases': 100, 'selection': 'first50 MIMIC + first50 Indiana P4 dev', 'modes': {},
               'interpretation': 'exploratory behavior control on exposed dev; proxies only, not clinical correctness'}
    for mode in ('negative', 'shuffled'):
        concordant, opposite, neutral, definite = 0, 0, 0, 0
        per_class = {}
        for j, label in enumerate(LABELS):
            before, after, paired = 0, 0, 0
            for i in range(100):
                a, b = labeled['baseline'][i][j], labeled[mode][i][j]
                if a not in (0, 1) or b not in (0, 1):
                    continue
                paired += 1
                before += a
                after += b
                moved_score = 0. if mode == 'negative' else scores[permutation[i]][j]
                direction = (moved_score - scores[i][j]) * (b - a)
                definite += 1
                concordant += direction > 0
                opposite += direction < 0
                neutral += direction == 0
            per_class[label] = {'paired_definite': paired, 'baseline_positive': before, 'control_positive': after,
                                'positive_count_delta': after - before}
        summary['modes'][mode] = {'changed_reports': sum(a['generated_report'] != b['generated_report'] for a, b in zip(baseline, variants[mode])),
            'per_class': per_class, 'paired_definite_class_cases': definite, 'direction_concordant': concordant,
            'direction_opposite': opposite, 'neutral': neutral,
            'direction_concordance_among_label_changes': concordant / (concordant + opposite) if concordant + opposite else None}
    atomic_json(BASE / 'intervention_summary.json', summary)


if __name__ == '__main__':
    main()
