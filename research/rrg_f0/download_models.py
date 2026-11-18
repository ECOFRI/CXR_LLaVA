"""Download public weights only. Never opens MIMIC files or sends local data."""
import json
from pathlib import Path

from research.rrg_f0.core import ROOT, EXPECTED_CLEAR_SHA, sha256

REVISIONS = {'peterhan91/CLEAR': '142f169c8b006b0f28cf55c6a237c3003db5ed8f',
             'Qwen/Qwen3-1.7B': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'}


def main():
    from huggingface_hub import hf_hub_download, snapshot_download
    result = {'status': 'RUNNING', 'models': {}, 'gpu_seconds': 0}
    out = ROOT / 'research/rrg_f0/model_download.json'
    if out.exists():
        raise FileExistsError('Preserve existing download evidence; inspect before rerunning')
    try:
        for repo, revision in REVISIONS.items():
            if repo == 'peterhan91/CLEAR':
                path = Path(hf_hub_download(repo, 'best_model.pt', revision=revision))
                digest = sha256(path)
                result['models'][repo] = {'revision': revision, 'path': str(path), 'sha256': digest}
                if digest != EXPECTED_CLEAR_SHA:
                    raise RuntimeError('STOP_CLEAR_SHA_MISMATCH')
            else:
                path = snapshot_download(repo, revision=revision,
                                         allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'], max_workers=2)
                result['models'][repo] = {'revision': revision, 'path': str(path)}
        result['status'] = 'PASS'
    except (OSError, RuntimeError, ValueError) as exc:
        result['status'] = 'STOP_MODEL_DOWNLOAD_OR_SHA'
        result['error_type'] = type(exc).__name__
        raise
    finally:
        out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
