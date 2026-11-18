import importlib.util
from pathlib import Path

script=Path(__file__).resolve().parents[1]/'scripts/eval/evaluate_reports.py'
spec=importlib.util.spec_from_file_location('evaluate_reports',script)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
empty={'Cardiomegaly':{'paired':0,'support':0,'tp':0,'fp':0,'fn':0,'tn':0,'f1':None}}
assert module.clinical_summary_or_na(empty)['micro_f1'] is None
false_positive={'Pneumothorax':{'paired':2,'support':0,'tp':0,'fp':2,'fn':0,'tn':0,'f1':0.0}}
result=module.clinical_summary_or_na(false_positive)
assert result['micro_f1'] is None and result['macro_f1'] is None
assert result['paired_labels']==2 and result['paired_positive_support']==0
print('PASS')
