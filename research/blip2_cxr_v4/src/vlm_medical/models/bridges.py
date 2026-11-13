"""Small trainable feature-to-OPT bridges; synthetic CPU checks need no backbone."""
import torch
from torch import nn


class LinearBridge(nn.Module):
    def __init__(self, feature_dim, language_dim, n_query=32):
        super().__init__()
        self.proj = nn.Linear(feature_dim, language_dim)
        self.n_query = n_query

    def forward(self, features):
        if features.ndim != 3:
            raise ValueError('features must be [batch,tokens,channels]')
        pooled = features.mean(dim=1, keepdim=True)
        return self.proj(pooled).expand(-1, self.n_query, -1)


class QueryBridge(nn.Module):
    def __init__(self, feature_dim, language_dim, hidden_dim=128, n_query=32):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, n_query, hidden_dim))
        self.key_value = nn.Linear(feature_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.proj = nn.Linear(hidden_dim, language_dim)

    def forward(self, features):
        if features.ndim != 3:
            raise ValueError('features must be [batch,tokens,channels]')
        query = self.query.expand(features.size(0), -1, -1)
        kv = self.key_value(features)
        attended, _ = self.attention(query, kv, kv, need_weights=False)
        return self.proj(attended)


def make_bridge(kind, feature_dim, language_dim, n_query=32):
    if kind == 'linear':
        return LinearBridge(feature_dim, language_dim, n_query)
    if kind == 'query':
        return QueryBridge(feature_dim, language_dim, n_query=n_query)
    raise ValueError('Unknown bridge: ' + kind)

class LoRALinear(nn.Module):
    """Local FP32 adapter on a frozen OPT projection."""
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.b = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        original = self.base(x)
        adapted = self.b(self.a(x.float())) * self.scale
        return original + adapted.to(original.dtype)


def inject_opt_lora(language_model):
    replaced = 0
    for module in language_model.modules():
        for name in ('q_proj', 'v_proj'):
            layer = getattr(module, name, None)
            if isinstance(layer, nn.Linear):
                setattr(module, name, LoRALinear(layer))
                replaced += 1
    if not replaced:
        raise RuntimeError('No OPT q_proj/v_proj layers found for LoRA')
    return replaced

class FP32Projection(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer.float()
        self.layer.requires_grad_(True)

    def forward(self, features):
        return self.layer(features.float()).to(torch.float16)


class FP32QFormer(nn.Module):
    def __init__(self, qformer):
        super().__init__()
        self.inner = qformer.float()
        self.inner.requires_grad_(True)

    def forward(self, *args, **kwargs):
        for name in ('query_embeds', 'encoder_hidden_states'):
            if name in kwargs and kwargs[name] is not None:
                kwargs[name] = kwargs[name].float()
        return self.inner(*args, **kwargs)
