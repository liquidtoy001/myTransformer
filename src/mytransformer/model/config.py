"""模型配置。

唯一的超参真源是 configs/model/*.yaml，这里只负责解析和校验。
参数量推导见 docs/02-architecture.md §1.3。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 49152
    d_model: int = 1280
    n_layers: int = 26
    n_heads: int = 20
    n_kv_heads: int = 5
    head_dim: int = 64
    ffn_hidden: int = 3456
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    attention_bias: bool = False
    dropout: float = 0.0
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError(
                f"d_model({self.d_model}) 必须等于 n_heads({self.n_heads}) × "
                f"head_dim({self.head_dim}) = {self.n_heads * self.head_dim}"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads({self.n_heads}) 必须能被 n_kv_heads({self.n_kv_heads}) 整除"
            )

    @property
    def n_rep(self) -> int:
        """GQA 中每个 KV 头要复制多少次以匹配 Q 头。"""
        return self.n_heads // self.n_kv_heads

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        # yaml 里还有 name / init / expected_params 等非模型字段，这里忽略
        return cls(**{k: v for k, v in raw.items() if k in known})

    # ---- 参数量：不构造模型也能算，用于开跑前核对预算 ----

    def count_params(self) -> dict[str, int]:
        d, kv = self.d_model, self.n_kv_heads * self.head_dim

        attn = d * d * 2 + d * kv * 2  # Wq, Wo, Wk, Wv
        mlp = 3 * d * self.ffn_hidden  # gate, up, down
        norms = 2 * d  # 每层两个 RMSNorm
        per_layer = attn + mlp + norms

        # 最终 RMSNorm 也算非嵌入参数，保证 total - embedding == non_embedding
        non_embedding = per_layer * self.n_layers + d
        embedding = self.vocab_size * d
        total = non_embedding + embedding
        if not self.tie_embeddings:
            total += embedding

        return {
            "per_layer": per_layer,
            "non_embedding": non_embedding,
            "embedding": embedding,
            "total": total,
            "bf16_bytes": total * 2,
        }
