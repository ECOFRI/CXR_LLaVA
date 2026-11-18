"""Frozen FP16 Qwen with FP32 visual/finding bridge and optional rank-16 LoRA."""
import torch
from torch import nn


class FindingBridge(nn.Module):
    def __init__(self, hidden, labels=14):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.finding_embeddings = nn.Embedding(labels, hidden)
        self.score_mlp = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.auxiliary = nn.Linear(768, labels)

    def forward(self, patches, scores):
        patches, scores = patches.float(), scores.float()
        visual = self.projector(patches)
        ids = torch.arange(scores.shape[-1], device=scores.device)
        finding = self.finding_embeddings(ids)[None] + self.score_mlp(scores[..., None])
        return torch.cat((visual, finding), dim=1), self.auxiliary(patches.mean(1))


class RRG(nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.bridge = FindingBridge(llm.config.hidden_size)

    def prefix(self, patches, scores):
        return self.bridge(patches, scores)[0].to(self.llm.get_input_embeddings().weight.dtype)

    def forward(self, patches, scores, ids, mask, labels, auxiliary_targets):
        prefix, auxiliary = self.bridge(patches, scores)
        outputs = self.language_output(prefix, ids, mask, labels)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(auxiliary, auxiliary_targets.float())
        return outputs.loss.float() + 0.1 * bce, outputs.loss.detach().float(), bce.detach()

    def language_output(self, prefix, ids, mask, labels):
        embeddings = self.llm.get_input_embeddings()(ids)
        embeddings = torch.cat((prefix.to(embeddings.dtype), embeddings), dim=1)
        prefix_mask = torch.ones(prefix.shape[:2], dtype=mask.dtype, device=mask.device)
        prefix_labels = torch.full(prefix.shape[:2], -100, dtype=labels.dtype, device=labels.device)
        outputs = self.llm(inputs_embeds=embeddings, attention_mask=torch.cat((prefix_mask, mask), 1),
                           labels=torch.cat((prefix_labels, labels), 1), use_cache=False)
        return outputs
