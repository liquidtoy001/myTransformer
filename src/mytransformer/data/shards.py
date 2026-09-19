"""分词后语料的读写：flat uint16 分片 + 可续跑的顺序加载器。

格式（见 docs/05-data.md §2.6）：每个 .bin 是纯 uint16 数组，无 header；
文档之间用 <|endoftext|> 分隔，训练时跨文档拼接成定长序列，不做 padding。

加载器的状态只有三个整数（epoch、分片序号、分片内偏移），随 checkpoint 保存。
续跑后产出的 batch 与不中断时逐个相同——这是接力训练的前提。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DTYPE = np.uint16  # 词表 32768 < 65536


def write_shard(path: str | Path, tokens: np.ndarray) -> None:
    arr = np.asarray(tokens)
    if arr.size and (arr.min() < 0 or arr.max() > np.iinfo(DTYPE).max):
        raise ValueError("token id 超出 uint16 范围")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr.astype(DTYPE).tofile(path)


@dataclass
class LoaderState:
    epoch: int = 0
    shard: int = 0   # 在本 epoch 洗牌后的分片顺序中的位置
    offset: int = 0  # 分片内偏移（token 数）


class ShardLoader:
    """按顺序读分片，产出 (x, y)，y 是 x 的下一个 token。

    - 每步读 B×T+1 个连续 token，切成 B 行（相邻行首尾重叠 1 个，nanoGPT 做法）
    - 分片顺序每个 epoch 用 seed+epoch 重新洗牌；数据不够时自然进入下一 epoch（多轮训练）
    - 多卡：rank r 读 offset + r×B×T 处；所有 rank 共享同一个 offset，
      换分片的判断按最后一个 rank 做，保证各 rank 在同一步换分片、永不错位
    """

    def __init__(
        self,
        paths: list[str | Path],
        batch_size: int,
        seq_len: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ) -> None:
        if not paths:
            raise ValueError("没有数据分片")
        self.paths = sorted(Path(p) for p in paths)
        self.B, self.T = batch_size, seq_len
        self.rank, self.world = rank, world_size
        self.seed = seed
        self.state = LoaderState()
        self._mmaps: dict[Path, np.memmap] = {}

        window = self.B * self.T * self.world + 1
        usable = [p for p in self.paths if self._data(p).size >= window]
        if not usable:
            raise ValueError(f"所有分片都比一步所需的 {window} 个 token 短")
        self.paths = usable

    # ---- 公开接口 ----

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.state, x, y = self._read(self.state)
        return x, y

    def peek(self) -> tuple[torch.Tensor, torch.Tensor]:
        """看下一个 batch 但不前进。续跑自检用：存档时记下，恢复后比对。"""
        _, x, y = self._read(self.state)
        return x, y

    def state_dict(self) -> dict:
        return {"epoch": self.state.epoch, "shard": self.state.shard, "offset": self.state.offset}

    def load_state_dict(self, d: dict) -> None:
        self.state = LoaderState(**d)

    @property
    def epoch(self) -> int:
        return self.state.epoch

    @property
    def total_tokens(self) -> int:
        return sum(self._data(p).size for p in self.paths)

    # ---- 内部 ----

    def _data(self, path: Path) -> np.memmap:
        if path not in self._mmaps:
            self._mmaps[path] = np.memmap(path, dtype=DTYPE, mode="r")
        return self._mmaps[path]

    def _order(self, epoch: int) -> list[Path]:
        if getattr(self, "_order_cache", (None,))[0] != epoch:
            perm = np.random.default_rng(self.seed + epoch).permutation(len(self.paths))
            self._order_cache = (epoch, [self.paths[i] for i in perm])
        return self._order_cache[1]

    def _normalize(self, epoch: int, shard: int, offset: int) -> tuple[int, int, int]:
        """把指针推进到下一个放得下一整步的位置（按最后一个 rank 判断）。

        每次读完立刻推进，让状态始终指向"下一个 batch 真正从哪里读"：
        否则指针会停在已读空的分片末尾，epoch 也会滞后一步。
        """
        need = self.B * self.T * self.world + 1
        while offset + need > self._data(self._order(epoch)[shard]).size:
            shard, offset = shard + 1, 0
            if shard == len(self.paths):
                epoch, shard = epoch + 1, 0
        return epoch, shard, offset

    def _read(self, s: LoaderState) -> tuple[LoaderState, torch.Tensor, torch.Tensor]:
        """纯函数：给定状态，返回（新状态, x, y）。不修改 self.state。"""
        bt = self.B * self.T
        epoch, shard, offset = self._normalize(s.epoch, s.shard, s.offset)
        data = self._data(self._order(epoch)[shard])
        start = offset + self.rank * bt
        buf = torch.from_numpy(data[start : start + bt + 1].astype(np.int64))
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        return LoaderState(*self._normalize(epoch, shard, offset + bt * self.world)), x, y


def eval_batches(paths: list[str | Path], batch_size: int, seq_len: int, n: int):
    """验证集：总是从头取固定的 n 个 batch，与训练加载器的状态无关。"""
    loader = ShardLoader(paths, batch_size, seq_len, seed=0)
    return [loader.next_batch() for _ in range(n)]
