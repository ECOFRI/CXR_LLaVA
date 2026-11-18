"""Shared frozen CLEAR path, identical pooling and direct paired scores to F0."""
import sys
import torch
from torch.nn import functional as F

from research.rrg_f0.core import ROOT, EXPECTED_CLEAR_SHA, pool_grid, sha256
from research.rrg_f0.download_models import REVISIONS


def checkpoint_path(repo, filename=None):
    path = ROOT / 'model_cache/rrg_f0/huggingface/hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / REVISIONS[repo]
    return path / filename if filename else path


class FrozenClear:
    def __init__(self, labels, device='cuda'):
        checkpoint = checkpoint_path('peterhan91/CLEAR', 'best_model.pt')
        if sha256(checkpoint) != EXPECTED_CLEAR_SHA:
            raise RuntimeError('STOP_CLEAR_SHA_MISMATCH')
        sys.path.insert(0, str(ROOT.parent / 'CLEAR/src'))
        torch.hub.set_dir(str(ROOT / 'research/cache/torch_hub'))
        from clear import load_pretrained, tokenize
        self.model, self.preprocess = load_pretrained(checkpoint_path=checkpoint, local_files_only=True, device=device)
        self.model.float().eval().requires_grad_(False)
        self.device = device
        prompts = [s for name in labels for s in (name.lower(), 'no ' + name.lower())]
        with torch.inference_mode():
            self.text = F.normalize(self.model.encode_text(tokenize(prompts).to(device)), dim=-1)

    @torch.inference_mode()
    def encode(self, inputs):
        output = self.model.visual.backbone.forward_features(inputs.to(self.device))
        patches = output['x_norm_patchtokens']
        if patches.shape[1:] != (1024, 768):
            raise RuntimeError('Unexpected CLEAR patch grid')
        pooled = pool_grid(patches).half().cpu()
        projected = F.normalize(self.model.visual.projection(output['x_norm_clstoken']), dim=-1)
        scores = (projected @ self.text.T).reshape(len(inputs), 14, 2).softmax(-1)[..., 0].float().cpu()
        if not torch.isfinite(pooled).all() or not torch.isfinite(scores).all():
            raise RuntimeError('Nonfinite CLEAR features')
        return pooled, scores
