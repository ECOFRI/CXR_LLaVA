import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlm_medical.eval.clinical_metrics import require_continuous_scores


def test_auc_rejects_binary():
    for values in ([0, 1], [1, 1], []):
        try:
            require_continuous_scores(values)
        except ValueError:
            pass
        else:
            raise AssertionError('Binary-only or empty values must be rejected')
    assert require_continuous_scores([0.1, 0.9]) == [0.1, 0.9]


if __name__ == '__main__':
    test_auc_rejects_binary()
