"""Validate the new isolated CPU environment without image/checkpoint inference."""
import importlib.metadata as metadata
import json
import os
import re
import sys
from pathlib import Path

import torch
import torchvision
from packaging.requirements import Requirement

from research.common import RESEARCH, ROOT, sha, write_json


def validate():
    assert sys.version_info >= (3, 10)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert torch.__version__ == '2.3.1+cpu'
    assert torchvision.__version__ == '0.18.1+cpu'
    assert torch.version.cuda is None and not torch.cuda.is_initialized()
    repo = ROOT.parent / 'CLEAR'
    source = (repo / 'pyproject.toml').read_text()
    block = re.search(r'dependencies\s*=\s*\[(.*?)\]', source, re.S).group(1)
    checks = []
    for text in re.findall(r'"([^"]+)"', block):
        requirement = Requirement(text)
        version = metadata.version(requirement.name)
        assert requirement.specifier.contains(version), (text, version)
        checks.append({'requirement': text, 'installed': version, 'status': 'PASS'})
    sys.path.insert(0, str(repo / 'src'))
    import clear
    with torch.inference_mode():
        attention = torch.nn.functional.scaled_dot_product_attention(torch.zeros(1, 1, 2, 32),
                     torch.zeros(1, 1, 2, 32), torch.zeros(1, 1, 2, 32))
        assert torch.isfinite(attention).all()
        assert torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([.5]), .5).tolist() == [0]
    result = {'status': 'PASS', 'python': sys.version, 'prefix': sys.prefix, 'requirements': checks,
              'torch': torch.__version__, 'torchvision': torchvision.__version__, 'torch_cuda': torch.version.cuda,
              'cuda_initialized': torch.cuda.is_initialized(), 'synthetic_CPU_SDPA': 'PASS',
              'torchvision_CPU_binary_op': 'PASS', 'official_CLEAR_import': 'PASS',
              'source_pyproject_sha256': sha(repo / 'pyproject.toml'), 'image_inference_count': 0}
    write_json(RESEARCH / 'environment_setup/clear_cpu_20261013/runtime_validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    validate()
