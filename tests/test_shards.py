from __future__ import annotations

import numpy as np
import pytest
import torch

from mytransformer.data.shards import ShardLoader, eval_batches, write_shard


def make_shards(tmp_path, sizes, start=0):
    paths, v = [], start
    for i, n in enumerate(sizes):
        p = tmp_path / f"shard_{i:03d}.bin"
        write_shard(p, np.arange(v, v + n) % 60000)  # 递增序列：一眼能看出读的是哪一段
        paths.append(p)
        v += n
    return paths


def test_x_y_are_next_tokens(tmp_path):
    loader = ShardLoader(make_shards(tmp_path, [1000]), batch_size=3, seq_len=8)
    x, y = loader.next_batch()
    assert x.shape == y.shape == (3, 8) and x.dtype == torch.int64
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y[:-1, -1], x[1:, 0])  # 相邻行首尾衔接
    assert torch.equal(x.flatten(), torch.arange(24))


def test_peek_does_not_advance(tmp_path):
    loader = ShardLoader(make_shards(tmp_path, [1000]), batch_size=2, seq_len=8)
    px, py = loader.peek()
    x, y = loader.next_batch()
    assert torch.equal(px, x) and torch.equal(py, y)


def test_resume_reproduces_stream(tmp_path):
    """续跑的前提：恢复状态后的 batch 序列与不中断时逐个相同，包括跨分片、跨 epoch。"""
    paths = make_shards(tmp_path, [100, 150, 90, 120])
    a = ShardLoader(paths, batch_size=2, seq_len=8, seed=7)
    for _ in range(9):
        a.next_batch()
    saved = a.state_dict()
    expected = [a.next_batch() for _ in range(40)]  # 足够跨越多个分片和 epoch

    b = ShardLoader(paths, batch_size=2, seq_len=8, seed=7)
    b.load_state_dict(saved)
    for (ex, ey), (gx, gy) in zip(expected, (b.next_batch() for _ in range(40)), strict=True):
        assert torch.equal(ex, gx) and torch.equal(ey, gy)
    assert a.epoch >= 1, "测试应覆盖到 epoch 回绕"


def test_epoch_reshuffles_shard_order(tmp_path):
    paths = make_shards(tmp_path, [17] * 8)
    loader = ShardLoader(paths, batch_size=1, seq_len=16, seed=0)
    firsts = {}
    while loader.epoch < 3:
        ep = loader.epoch
        x, _ = loader.next_batch()
        firsts.setdefault(ep, []).append(int(x[0, 0]))
    assert firsts[0] != firsts[1] != firsts[2], "每个 epoch 的分片顺序应重新洗牌"
    assert sorted(firsts[0]) == sorted(firsts[1]), "每个 epoch 都完整走一遍所有分片"


def test_ranks_partition_the_stream(tmp_path):
    """world=2 时 rank r 在第 s 步读到的，恰好是单卡第 s×2+r 步读到的——数据不重不漏。"""
    bt = 2 * 8
    paths = make_shards(tmp_path, [4 * bt + 1])
    single = ShardLoader(paths, 2, 8)
    ref = [single.next_batch()[0] for _ in range(4)]
    r0 = ShardLoader(paths, 2, 8, rank=0, world_size=2)
    r1 = ShardLoader(paths, 2, 8, rank=1, world_size=2)
    for s in range(2):
        assert torch.equal(r0.next_batch()[0], ref[2 * s])
        assert torch.equal(r1.next_batch()[0], ref[2 * s + 1])
    assert r0.state_dict() == r1.state_dict(), "各 rank 的状态必须始终一致"


def test_short_shards_are_skipped(tmp_path):
    paths = make_shards(tmp_path, [5, 1000])
    loader = ShardLoader(paths, batch_size=2, seq_len=8)
    assert len(loader.paths) == 1
    with pytest.raises(ValueError, match="都比"):
        ShardLoader(make_shards(tmp_path / "t", [5, 6]), batch_size=2, seq_len=8)


def test_eval_batches_are_fixed(tmp_path):
    paths = make_shards(tmp_path, [1000])
    a = eval_batches(paths, 2, 8, 3)
    b = eval_batches(paths, 2, 8, 3)
    assert all(torch.equal(x1, x2) for (x1, _), (x2, _) in zip(a, b, strict=True))


def test_write_shard_rejects_out_of_range(tmp_path):
    with pytest.raises(ValueError):
        write_shard(tmp_path / "bad.bin", np.array([70000]))
