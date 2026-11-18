import importlib.util
from pathlib import Path

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paired_bootstrap', root/'scripts/eval/paired_bootstrap.py')
boot = importlib.util.module_from_spec(spec); spec.loader.exec_module(boot)
spec = importlib.util.spec_from_file_location('compare_runs', root/'scripts/eval/compare_runs.py')
import sys
sys.path.insert(0,str(root/'scripts/eval'))
compare = importlib.util.module_from_spec(spec); spec.loader.exec_module(compare)

def test_refuse_mismatch():
    rows=[{'arm':'C0','image_id':'a'},{'arm':'C0','image_id':'b'},{'arm':'F1','image_id':'a'}]
    try: compare.strict_cohort(rows,'C0')
    except ValueError: return
    raise AssertionError('mismatch accepted')

def test_study_grouping():
    samples=boot.draws(['a','b'],1000,42)
    assert len(samples)==1000 and all(len(x)==2 for x in samples)
    reports={'a':[1,2,3],'b':[4,5]}
    assert any(sum(len(reports[k]) for k in sample)==6 for sample in samples)

if __name__=='__main__':
    test_refuse_mismatch();test_study_grouping();print('PASS')
