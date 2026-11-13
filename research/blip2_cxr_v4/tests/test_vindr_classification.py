import sys
import json
import tempfile
from pathlib import Path

VR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VR / 'src'))
sys.path.insert(0, str(VR / 'scripts/eval'))
from vlm_medical.eval.classification_metrics import auroc, average_precision, threshold_metrics
from evaluate_classification import evaluate, load_thresholds


def test_known_scores():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.6, 0.9]
    assert auroc(labels, scores) == 1.0
    assert average_precision(labels, scores) == 1.0
    counts = threshold_metrics(labels, scores, 0.5)
    assert (counts['tp'], counts['fp'], counts['fn'], counts['tn']) == (2, 0, 0, 2)
    for fn in (auroc, average_precision):
        try:
            fn(labels, [0, 0, 1, 1])
        except ValueError:
            pass
        else:
            raise AssertionError('Binary-only scores must be rejected')
    with tempfile.TemporaryDirectory(dir=VR / '.cache/tmp') as folder:
        path = Path(folder) / 'thresholds.json'
        path.write_text(json.dumps({'selection_split': 'test', 'locked_before_test': True,
                                    'thresholds': {'Finding': 0.5}}))
        try:
            load_thresholds(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Test-derived threshold must be rejected')
        path.write_text(json.dumps({'selection_split': 'validate', 'locked_before_test': True,
                                    'thresholds': {'Finding': 0.5}}))
        assert load_thresholds(path) == {'Finding': 0.5}
    label_rows = [{'image_id': str(i), 'Finding': str(y)} for i, y in enumerate(labels)]
    score_rows = [{'image_id': str(i), 'finding': 'Finding', 'score': str(x)}
                  for i, x in enumerate(scores)]
    result = evaluate(label_rows, score_rows, {'Finding': 0.5}, {}, 10, 42)
    assert result['n_images'] == 4 and result['per_class']['Finding']['auroc'] == 1.0


if __name__ == '__main__':
    test_known_scores()
