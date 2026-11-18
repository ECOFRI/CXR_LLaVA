import importlib.metadata as metadata
import json
import sys

names = ['torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'deepspeed',
         'lavis', 'salesforce-lavis', 'timm', 'open_clip', 'nltk', 'pycocoevalcap',
         'rouge_score', 'evaluate', 'sacrebleu', 'f1chexbert', 'chexbert', 'radgraph']
result = {'python': sys.version.split()[0], 'packages': {}}
for name in names:
    try:
        result['packages'][name] = metadata.version(name.replace('_', '-'))
    except metadata.PackageNotFoundError:
        result['packages'][name] = 'MISSING'
try:
    import torch
    result['torch_cuda_build'] = torch.version.cuda or 'CPU'
    result['cuda_available'] = torch.cuda.is_available()
    if result['cuda_available']:
        cap = torch.cuda.get_device_capability()
        result['fp16_supported'] = cap[0] >= 5
        result['bf16_supported'] = torch.cuda.is_bf16_supported()
    else:
        result['fp16_supported'] = 'UNKNOWN'
        result['bf16_supported'] = 'UNKNOWN'
except (ImportError, OSError, RuntimeError) as exc:
    result['torch_error'] = type(exc).__name__
print(json.dumps(result, sort_keys=True))
