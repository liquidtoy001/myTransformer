"""checkpoint 的原子写入、查找与清理。

原子写入：先写 ckpt_XXXXXXXX.pt.tmp，写完再 os.replace 成正式文件名。
同一文件系统内的 rename 是原子的——作业在写到一半时被杀，
最坏也只是留下一个 .tmp，上一份完好的 checkpoint 不受影响。

内容只含张量、dict、list、数字和字符串，因此可以用 weights_only=True 安全加载。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch

_NAME = re.compile(r"^ckpt_(\d{8})\.pt$")


def path_for(run_dir: Path, step: int) -> Path:
    return run_dir / f"ckpt_{step:08d}.pt"


def save(run_dir: Path, step: int, payload: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    final = path_for(run_dir, step)
    tmp = final.with_name(final.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, final)
    return final


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    if not run_dir.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in run_dir.iterdir() if (m := _NAME.match(p.name))]
    return sorted(found)


def latest(run_dir: Path) -> Path | None:
    ckpts = list_checkpoints(run_dir)
    return ckpts[-1][1] if ckpts else None


def load(path: Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=True)


FINAL_NAME = "final.pt"


def finalize(run_dir: Path, payload: dict) -> Path:
    """训练完成后：写一个只含权重的 final.pt，再删掉所有完整 checkpoint。

    完整 checkpoint 里优化器状态（AdamW 的两个动量）是权重的 2 倍大，只有续跑时才需要。
    训完之后留着它只是占空间：39M 参数的模型，完整 checkpoint 470 MB，只存权重 157 MB。
    Rangpur 的 home 只有 16 GB，P5 的十几个实验必须这样才放得下。

    先写 final.pt 再删旧文件：写到一半被杀，旧的完整 checkpoint 还在，下次还能续上。
    """
    final = run_dir / FINAL_NAME
    tmp = final.with_name(final.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, final)
    for _, p in list_checkpoints(run_dir):
        p.unlink()
    return final


def final_path(run_dir: Path) -> Path | None:
    p = run_dir / FINAL_NAME
    return p if p.exists() else None


def prune(run_dir: Path, keep: int, keep_every: int = 0) -> list[Path]:
    """只保留最新的 keep 份；step 是 keep_every 整数倍的永久保留（供中间 checkpoint 评测）。

    同时清掉残留的 .tmp（上次写到一半被杀留下的）。
    """
    removed = []
    ckpts = list_checkpoints(run_dir)
    for step, path in ckpts[:-keep] if keep > 0 else []:
        if keep_every and step % keep_every == 0:
            continue
        path.unlink()
        removed.append(path)
    for tmp in run_dir.glob("ckpt_*.pt.tmp"):
        tmp.unlink()
        removed.append(tmp)
    return removed
