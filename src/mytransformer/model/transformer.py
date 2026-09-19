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
from torch.utils.checkpoint import checkpoint

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
        *,
        shift: bool = True,
        ce_chunk: int = 0,
        z_loss: float = 0.0,
    ) -> dict:
        """input_ids: (B, T)。targets 给了就同时返回 loss。

        shift=True：HF 约定，targets 与 input_ids 对齐，内部错开一位。
        shift=False：targets 已经是下一个 token（训练时加载器直接给 x、y），T 个位置全部算 loss。
        ce_chunk>0：分块计算 lm_head + 交叉熵并逐块做激活重算，峰值显存只剩一块的 logits；
                    此时不返回完整 logits——训练用不到它，而它恰恰是显存大头。
        z_loss>0：额外加 z_loss · mean(logsumexp²)，抑制 logits 整体漂移（PaLM）。
        """
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

        h = self.norm(x)
        if targets is None:
            return {"logits": self.lm_head(h), "loss": None, "kv_caches": new_caches}

        h_pred, y = (h[:, :-1], targets[:, 1:]) if shift else (h, targets)
        h_pred = h_pred.reshape(-1, h_pred.size(-1))
        y = y.reshape(-1)
        n_valid = (y != -100).sum().clamp(min=1)

        if ce_chunk > 0:
            # 每块单独做激活重算：只保存该块的输入 h（N×d），反向时再算一遍 logits。
            # 只切块不重算的话，autograd 仍会为每块保留反向用的 logits，显存一点没省。
            logits = None
            total = sum(
                checkpoint(self._loss_sum, h_pred[i : i + ce_chunk], y[i : i + ce_chunk], z_loss, use_reentrant=False)
                for i in range(0, h_pred.size(0), ce_chunk)
            )
        else:
            logits = self.lm_head(h)
            flat = (logits[:, :-1] if shift else logits).reshape(-1, logits.size(-1))
            total = self._loss_from_logits(flat, y, z_loss)

        return {"logits": logits, "loss": total / n_valid, "kv_caches": new_caches}

    def _loss_sum(self, h: torch.Tensor, y: torch.Tensor, z_loss: float) -> torch.Tensor:
        return self._loss_from_logits(self.lm_head(h), y, z_loss)

    @staticmethod
    def _loss_from_logits(logits: torch.Tensor, y: torch.Tensor, z_loss: float) -> torch.Tensor:
        """返回整块的 loss 之和（不是均值），由调用方统一除以有效 token 数。"""
        # bf16/fp16 升到 fp32 再做 softmax；fp32/fp64 保持原精度（不能把 fp64 降成 fp32）
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        total = F.cross_entropy(logits, y, ignore_index=-100, reduction="sum")
        if z_loss > 0:
            lse = torch.logsumexp(logits, dim=-1)
            total = total + z_loss * lse[y != -100].pow(2).sum()
        return total

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
