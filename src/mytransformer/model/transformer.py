"""完整模型。

层次结构与权重命名对齐 HuggingFace Llama，因此可以与
`transformers.LlamaForCausalLM` 直接互换 state_dict 做数值对齐测试
（见 tests/test_hf_parity.py）。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .block import TransformerBlock
from .config import ModelConfig
from .norm import RMSNorm
from .rope import RotaryEmbedding


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)
        # 残差分支的输出投影额外缩放，必须在 apply 之后单独做
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for block in self.layers:
            with torch.no_grad():
                block.self_attn.o_proj.weight.mul_(scale)
                block.mlp.down_proj.weight.mul_(scale)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> dict:
        """input_ids: (B, T)。targets 给了就同时返回 loss。"""
        _, t = input_ids.shape
        offset = kv_caches[0][0].shape[2] if kv_caches else 0
        cos, sin = self.rotary(t, offset)
        cos, sin = cos.to(input_ids.device), sin.to(input_ids.device)

        x = self.embed_tokens(input_ids)

        new_caches = [] if kv_caches is not None else None
        for i, block in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = block(x, cos, sin, cache)
            if new_caches is not None:
                new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 标准的下一 token 预测：logits[..., :-1] 对 targets[..., 1:]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                targets[:, 1:].reshape(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss, "kv_caches": new_caches}

    # ---- 工具方法 ----

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def param_groups(self, weight_decay: float) -> list[dict]:
        """2D 权重衰减，1D（RMSNorm/bias）不衰减。

        嵌入是 2D 但通常也不衰减，这里显式排除。
        """
        embed_ids = {id(self.embed_tokens.weight), id(self.lm_head.weight)}
        decay, nodecay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 and id(p) not in embed_ids else nodecay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
