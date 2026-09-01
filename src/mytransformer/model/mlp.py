"""SwiGLU 前馈网络。

down(silu(gate(x)) * up(x))。比 GeLU 多一个矩阵，所以 ffn_hidden 取
(8/3)·d_model 而非 4·d_model，保持参数量相当。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, h = cfg.d_model, cfg.ffn_hidden
        # 命名对齐 HuggingFace Llama
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
