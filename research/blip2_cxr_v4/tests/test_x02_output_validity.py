import importlib.util
from pathlib import Path

script=Path(__file__).resolve().parents[1]/'scripts/experiments/infer_blip2.py'
spec=importlib.util.spec_from_file_location('infer_blip2',script)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not module.valid_generated_report('')
assert not module.valid_generated_report(' \n\t ')
assert not module.valid_generated_report(None)
assert module.valid_generated_report('No focal opacity.')
print('PASS')
