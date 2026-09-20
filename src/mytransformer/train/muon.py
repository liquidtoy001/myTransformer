"""Muon 优化器（消融 A1 的对照组）。

Keller Jordan 2024（https://kellerjordan.github.io/posts/muon/）。一句话：
**先按动量算出更新量，再把这个更新矩阵"正交化"，然后才加到权重上。**

为什么这样做有用：一个矩阵的梯度往往被少数几个方向主导（奇异值差几个数量级），
沿它更新等于只在那几个方向上走。正交化把所有奇异值拉到 1 附近，各个方向走得一样多。

正交化用 Newton-Schulz 迭代做——**不求 SVD**：SVD 在 GPU 上慢，而且我们不需要精确的
正交矩阵，只要奇异值都落在 1 附近就够了。迭代只有矩阵乘法，5 步、bf16 精度足够。

**只对二维权重用 Muon**：嵌入、lm_head、RMSNorm 的缩放参数仍然用 AdamW。
嵌入的每一行是独立的词向量，"矩阵方向"没有意义；一维参数根本没法正交化。

两个优化器的学习率不是一回事（Muon 通常大一个数量级），所以 `TrainConfig.muon_lr` 单列，
调度曲线按同一个比例同时缩放两者。
"""

from __future__ import annotations

import torch


def orthogonalize(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz 五次迭代：把矩阵的奇异值都推向 1，奇异向量不变。

    迭代式 X ← aX + b(XXᵀ)X + c(XXᵀ)²X 的系数取自原实现，它让 [0,1] 区间内的奇异值
    快速收敛到 1 附近（并不精确到 1，够用即可）。先按谱范数归一化保证收敛。
    行多于列时先转置：迭代里算的是 XXᵀ，取短边那侧更省。
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """只处理二维参数。其余参数交给 AdamW（见 `split_params`）。"""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, weight_decay: float = 0.0, ns_steps: int = 5) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(f"Muon 只接受二维参数，收到 {tuple(p.shape)}")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                update = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                update = orthogonalize(update, group["ns_steps"])
                # 正交化后每个元素的大小只由形状决定，和梯度大小无关；
                # 这个系数让不同形状的矩阵每步走的"相对幅度"大致一致（原实现的做法）
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * scale)
        return loss


def split_params(model) -> tuple[list, list]:
    """(给 Muon 的二维权重, 给 AdamW 的其余参数)。

    嵌入和 lm_head 即使是二维也归 AdamW：它们的每一行是一个词向量，
    对整个矩阵做正交化没有意义（原论文同样这样处理）。
    """
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embedding = "embed" in name or "lm_head" in name
        (muon if p.ndim == 2 and not is_embedding else adamw).append(p)
    return muon, adamw
