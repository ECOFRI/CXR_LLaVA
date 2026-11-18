import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/eval'))
from evaluate_reports import audit


def test_reference_alignment():
    cohort = [{'report_id': 'study', 'image_id': 'image', 'ground_truth_report': 'findings impression',
               'reference_policy': 'findings+impression', 'view_label': 'frontal_unspecified'}]
    pred = [{'cohort_index': '0', 'report_id': 'study', 'image_id': 'image',
             'ground_truth_report': 'findings impression', 'generated_report': 'text', 'status': 'success'}]
    gt = [{'Reports': 'findings impression'}]
    generated = [{'Reports': 'text'}]
    _, valid, result = audit(pred, cohort, cohort, gt, generated)
    assert valid == [0]
    assert result['reference_section_policies'] == {'findings+impression': 1}
    bad_source = [dict(cohort[0], ground_truth_report='different')]
    try:
        audit(pred, cohort, bad_source, gt, generated)
    except ValueError:
        pass
    else:
        raise AssertionError('Different reference section text must fail alignment')


if __name__ == '__main__':
    test_reference_alignment()
