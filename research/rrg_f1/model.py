"""F1 arms share F0 projection/token implementation and language-loss masking."""
import torch
from torch import nn
from research.rrg_f0.model import FindingBridge, RRG


class ArmBridge(FindingBridge):
    def __init__(self, hidden, arm):
        super().__init__(hidden)
        self.arm = arm
        self.auxiliary = nn.Linear(hidden, 14)
        self.finding_embeddings.requires_grad_(arm == 'A4')
        self.score_mlp.requires_grad_(arm == 'A4')
        self.auxiliary.requires_grad_(arm != 'A2')

    def forward(self, patches, scores):
        visual = self.projector(patches.float())
        logits = self.auxiliary(visual.mean(1)) if self.arm != 'A2' else None
        if self.arm == 'A4':
            ids = torch.arange(14, device=scores.device)
            finding = self.finding_embeddings(ids)[None] + self.score_mlp(scores.float()[..., None])
            visual = torch.cat((visual, finding), 1)
        return visual, logits


class ArmModel(RRG):
    def __init__(self, llm, arm, prior):
        nn.Module.__init__(self)
        self.llm = llm
        self.arm = arm
        torch.manual_seed(42)
        self.bridge = ArmBridge(llm.config.hidden_size, arm)
        self.register_buffer('prior_log_odds', torch.logit(prior.float()))

    def forward(self, patches, scores, ids, mask, labels, auxiliary_targets, auxiliary_mask):
        prefix, logits = self.bridge(patches, scores)
        output = self.language_output(prefix, ids, mask, labels)
        bce = output.loss.new_zeros(())
        if logits is not None:
            values = nn.functional.binary_cross_entropy_with_logits(logits + self.prior_log_odds, auxiliary_targets.float(), reduction='none')
            bce = (values * auxiliary_mask).sum() / auxiliary_mask.sum().clamp_min(1)
        return output.loss.float() + 0.1 * bce, output.loss.detach().float(), bce.detach()
