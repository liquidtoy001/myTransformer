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


class GeLUMLP(nn.Module):
    """消融 A2 的对照：普通两层前馈 down(gelu(up(x)))。

    比 SwiGLU 少一个矩阵，所以同样的 ffn_hidden 下参数更少。要和 SwiGLU 公平对比，
    配置里把 ffn_hidden 放大到 1.5 倍（SwiGLU 3 个矩阵 vs 这里 2 个），让参数量相当。
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, h = cfg.d_model, cfg.ffn_hidden
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x), approximate="tanh"))


def build_mlp(cfg: ModelConfig) -> nn.Module:
    if cfg.activation == "swiglu":
        return SwiGLU(cfg)
    if cfg.activation == "gelu":
        return GeLUMLP(cfg)
    raise ValueError(f"activation 只能是 swiglu 或 gelu，得到 {cfg.activation!r}")
