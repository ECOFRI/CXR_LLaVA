import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlm_medical.eval.text_metrics import case_statistics, corpus_scores


def test_bleu_known_examples():
    exact = corpus_scores([case_statistics('Clear lungs and normal heart.', 'clear lungs and normal heart')])
    assert all(abs(exact[f'bleu{n}'] - 1.0) < 1e-12 for n in range(1, 5))
    assert exact['rouge_l'] == 1.0
    disjoint = corpus_scores([case_statistics('clear lungs', 'enlarged heart')])
    assert all(disjoint[f'bleu{n}'] == 0.0 for n in range(1, 5))


if __name__ == '__main__':
    test_bleu_known_examples()
