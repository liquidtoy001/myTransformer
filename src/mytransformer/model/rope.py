"""旋转位置编码（RoPE）。

采用 GPT-NeoX / HuggingFace Llama 的 "rotate_half" 约定：把 head_dim 前后
对半切开做旋转。**不是** GPT-J 的相邻元素交错约定——两者数学等价但权重
排布不同，混用会导致数值对齐失败且很难查。

cos/sin 用 fp32 预计算并缓存，前向时再转到输入 dtype。
"""

from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把最后一维对半切开，返回 (-x2, x1)。"""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 q/k 施加旋转。v 不参与。

    q: (B, n_heads,    T, head_dim)
    k: (B, n_kv_heads, T, head_dim)
    cos/sin: (T, head_dim) —— 在头维度上广播
    """
    cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        # persistent=False：这些是可重算的常量，不进 checkpoint
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cached_len = seq_len

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 [offset, offset+seq_len) 这段位置的 cos/sin。

        offset 用于增量解码：KV cache 里已有 offset 个 token 时，新 token 的
        位置从 offset 开始。
        """
        end = offset + seq_len
        if end > self._cached_len:
            # 上下文扩展（退火期 2048 -> 4096）时会走到这里
            self._build_cache(end)
            self.cos_cached = self.cos_cached.to(self.cos_cached.device)
            self.sin_cached = self.sin_cached.to(self.sin_cached.device)
        return self.cos_cached[offset:end], self.sin_cached[offset:end]
