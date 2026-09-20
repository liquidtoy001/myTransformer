"""Transformer 层：pre-norm + 残差直连。

    h = x + Attn(RMSNorm(x))
    h = h + SwiGLU(RMSNorm(h))

残差路径上不做任何缩放——缩放放在初始化里（见 transformer.py 的
_init_weights），这是 GPT-2 的做法：残差分支的输出投影按 1/sqrt(2·n_layers)
缩小初始值，防止残差流方差随深度线性累积。
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import Attention
from .config import ModelConfig
from .mlp import build_mlp
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # 命名对齐 HuggingFace Llama
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = build_mlp(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, new_cache = self.self_attn(self.input_layernorm(x), cos, sin, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache
