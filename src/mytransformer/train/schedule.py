"""学习率调度：WSD（主线）与 cosine（消融 A5 的对照组）。

WSD = warmup → stable（恒定峰值）→ decay（线性降到 min）。
选它而不选 cosine，是因为稳定段 LR 恒定，可以在任意一步分叉做退火：
缩放律的第 4 个点就是从主训练的 step 9000 分叉、做 900 步衰减得到的——
那就是一个 decay_start=9000、total_steps=9900 的 WSD，不需要任何特殊逻辑。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LRSchedule:
    peak_lr: float
    warmup_steps: int
    total_steps: int
    kind: str = "wsd"                 # "wsd" | "cosine"
    decay_start: int | None = None    # 仅 wsd：稳定段持续到这一步；默认总步数的 90%
    min_lr_ratio: float = 0.0         # 衰减终点 = peak_lr × min_lr_ratio

    def __post_init__(self) -> None:
        if self.kind not in ("wsd", "cosine"):
            raise ValueError(f"未知的调度类型 {self.kind!r}")
        if not 0 <= self.warmup_steps <= self.total_steps:
            raise ValueError("warmup_steps 必须在 [0, total_steps] 内")
        if self.kind == "wsd" and not self.warmup_steps <= self._decay_start <= self.total_steps:
            raise ValueError("decay_start 必须在 [warmup_steps, total_steps] 内")

    @property
    def _decay_start(self) -> int:
        return self.decay_start if self.decay_start is not None else int(0.9 * self.total_steps)

    def __call__(self, step: int) -> float:
        peak, floor = self.peak_lr, self.peak_lr * self.min_lr_ratio
        if step < self.warmup_steps:
            return peak * (step + 1) / self.warmup_steps
        if step >= self.total_steps:
            return floor

        if self.kind == "wsd":
            start = self._decay_start
            if step < start:
                return peak
            frac = (step - start) / max(1, self.total_steps - start)
            return peak + (floor - peak) * frac

        frac = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * frac))
