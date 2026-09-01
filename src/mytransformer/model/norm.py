"""RMSNorm。

实现与 HuggingFace LlamaRMSNorm 完全一致（包括 fp32 中间计算、以及
在乘 weight 之前先转回输入 dtype 这个细节），否则数值对齐测试过不了。
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # 方差在 fp32 下算：bf16 的动态范围够，但精度不够，平方和会明显掉位
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x.to(torch.float32) * torch.rsqrt(var + self.eps)
        return self.weight * x.to(in_dtype)

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)}, eps={self.eps}"
