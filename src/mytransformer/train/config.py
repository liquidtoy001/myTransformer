"""训练配置。唯一的超参真源是 configs/train/*.yaml。

严格校验：yaml 里出现未知字段直接报错。拼错一个键名（比如 warmup_step）
会被悄悄忽略、用上默认值，这类错误往往要到训完才发现。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    model: str                              # 模型配置 yaml
    train_data: str                         # 训练分片的 glob
    val_data: str | None = None             # 验证分片的 glob；匹配到多个文件时每个文件单独报 loss
    tokenizer: str | None = None            # 分词器文件；设置后训练前核对分片 meta.json 里的指纹
    run_dir: str = "runs/default"

    # ---- batch ----
    seq_len: int = 2048
    global_batch_tokens: int = 524288       # 与硬件无关的常量；用梯度累积吸收显存差异
    micro_batch_size: int = 16
    max_steps: int = 1000

    # ---- 优化器 ----
    lr: float = 7e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    optimizer: str = "adamw"                # adamw | muon（消融 A1）
    muon_lr: float = 0.02                   # Muon 的学习率量级和 AdamW 不同，单列
    muon_momentum: float = 0.95

    # ---- 学习率调度 ----
    schedule: str = "wsd"
    warmup_steps: int = 100
    decay_start: int | None = None          # 默认 90%
    min_lr_ratio: float = 0.0

    # ---- 精度与性能 ----
    dtype: str = "bfloat16"                 # bfloat16 | float32
    compile: bool = False
    ce_chunk: int = 4096                    # 分块交叉熵每块的 token 数；0 表示不分块
    z_loss: float = 1e-4

    # ---- 日志与存档 ----
    seed: int = 0
    log_every: int = 10
    eval_every: int = 250
    eval_batches: int = 8
    ckpt_every: int = 1000
    ckpt_keep: int = 3
    ckpt_keep_every: int = 0                # 这个整数倍的 step 永久保留（中间 checkpoint 评测用）
    peak_tflops: float | None = None        # 不填则按 GPU 型号查表

    # 本 run 没有任何 checkpoint 时，从这个 checkpoint 起步（WSD 分叉用）
    init_from: str | None = None

    extra: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)
        if self.dtype not in ("bfloat16", "float32"):
            raise ValueError(f"dtype 只能是 bfloat16 或 float32，得到 {self.dtype!r}")

    def grad_accum_steps(self, world_size: int = 1) -> int:
        per_step = self.micro_batch_size * self.seq_len * world_size
        if self.global_batch_tokens % per_step:
            raise ValueError(
                f"global_batch_tokens={self.global_batch_tokens} 不能被 "
                f"micro_batch_size × seq_len × world_size = {per_step} 整除"
            )
        return self.global_batch_tokens // per_step

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)} - {"extra"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"{path} 里有未知字段：{sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        d["betas"] = list(self.betas)
        return d
