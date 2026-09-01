"""分组查询注意力（GQA）。

n_kv_heads < n_heads：KV cache 缩小 n_rep 倍。本项目 20 Q / 5 KV，n_rep=4。
走 F.scaled_dot_product_attention，PyTorch 会自动选 FlashAttention 后端。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .rope import apply_rope


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, T, hd) -> (B, n_kv*n_rep, T, hd)。

    等价于 torch.repeat_interleave(x, n_rep, dim=1)，但用 expand 避免真实拷贝。
    """
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep

        d = cfg.d_model
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        bias = cfg.attention_bias
        # 命名对齐 HuggingFace Llama，便于 state_dict 直接互换做数值对齐
        self.q_proj = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=bias)
        self.k_proj = nn.Linear(d, kv_dim, bias=bias)
        self.v_proj = nn.Linear(d, kv_dim, bias=bias)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, d, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v) if kv_cache is not None else None

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # 增量解码时 q 只有 1 个 token 而 k 有多个，is_causal 语义会出错，
        # 此时不需要 mask（新 token 本来就能看到全部历史）
        is_causal = q.shape[2] == k.shape[2] and t > 1
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out), new_cache
