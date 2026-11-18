"""Record exact CLEAR K80 blockers without environment changes or model inference."""
import json
import sys
import traceback
from pathlib import Path

import torch

from research.common import RESEARCH, ROOT, command, finish_run, new_run, sha, write_json


def preflight():
    run, manifest = new_run('E0A_CLEAR_PREFLIGHT')
    clear_repo = ROOT.parent / 'CLEAR'
    dino = RESEARCH / 'upstream/dinov2'
    sys.path.insert(0, str(clear_repo / 'src'))
    sys.path.insert(0, str(dino))
    import clear
    from dinov2.layers.attention import Attention
    checkpoints = list(Path('/home/student4/.cache/huggingface/hub/models--peterhan91--CLEAR/snapshots').glob('*/best_model.pt'))
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    facts = {'clear_source_commit': command(['git', '-C', str(clear_repo), 'rev-parse', 'HEAD'])['stdout'].strip(),
             'dinov2_source_commit': command(['git', '-C', str(dino), 'rev-parse', 'HEAD'])['stdout'].strip(),
             'clear_metadata_requirement': 'Python>=3.10, torch>=2.0, torchvision>=0.15',
             'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'compiled_arch': torch.cuda.get_arch_list(),
             'checkpoint_path': str(checkpoint), 'checkpoint_revision': checkpoint.parent.name,
             'checkpoint_sha256': sha(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
             'clear_import': 'PASS', 'sdpa_available': hasattr(torch.nn.functional, 'scaled_dot_product_attention'),
             'source_attention_sha256': sha(dino / 'dinov2/layers/attention.py'),
             'environment_modified': False, 'checkpoint_inference_executed': False}
    try:
        with torch.inference_mode():
            Attention(dim=32, num_heads=4).eval()(torch.zeros(1, 2, 32))
    except AttributeError as exc:
        facts.update(attention_preflight='BLOCKED', exact_error=f'{type(exc).__name__}: {exc}')
        (run / 'stderr.log').write_text(traceback.format_exc())
    else:
        facts['attention_preflight'] = 'PASS'
    write_json(run / 'compatibility.json', facts)
    manifest.update(env=sys.prefix, command=sys.argv, runner_sha256=sha(__file__), dataset='none: synthetic CPU attention input',
                    checkpoint=facts['checkpoint_revision'], blocker=facts.get('exact_error', ''),
                    next_action='Approve separate CPU-compatible Python>=3.10/PyTorch>=2 environment; stable K80 env stays unchanged')
    finish_run(run, manifest, 'BLOCKED' if facts['attention_preflight'] == 'BLOCKED' else 'COMPLETE')
    print(json.dumps({'run': str(run), **facts}, indent=2))


if __name__ == '__main__':
    preflight()
