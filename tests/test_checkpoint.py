from __future__ import annotations

import pytest
import torch

from mytransformer.train import checkpoint as ck


def test_atomic_save_leaves_no_tmp(tmp_path):
    p = ck.save(tmp_path, 5, {"x": torch.ones(3)})
    assert p.name == "ckpt_00000005.pt"
    assert not list(tmp_path.glob("*.tmp"))
    assert torch.equal(ck.load(p)["x"], torch.ones(3))


def test_latest_ignores_partial_writes(tmp_path):
    """写到一半被杀留下的 .tmp 不能被当成最新 checkpoint。"""
    ck.save(tmp_path, 100, {"step": 100})
    ck.save(tmp_path, 200, {"step": 200})
    (tmp_path / "ckpt_00000300.pt.tmp").write_bytes(b"truncated")
    assert ck.load(ck.latest(tmp_path))["step"] == 200


def test_crash_during_save_keeps_previous(tmp_path, monkeypatch):
    """原子写入的真正意义：写到一半被杀，上一份 checkpoint 必须完好可用。

    这个性质不主动模拟故障是测不出来的——把 save 改成直接写正式文件名，
    其它测试照样全过。
    """
    ck.save(tmp_path, 100, {"step": 100})

    def killed_midway(obj, f):
        with open(f, "wb") as fh:
            fh.write(b"half-written")  # 只写了一半
        raise KeyboardInterrupt("模拟作业被杀")

    monkeypatch.setattr(torch, "save", killed_midway)
    with pytest.raises(KeyboardInterrupt):
        ck.save(tmp_path, 200, {"step": 200})
    monkeypatch.undo()

    assert ck.load(ck.latest(tmp_path))["step"] == 100


def test_latest_on_missing_dir(tmp_path):
    assert ck.latest(tmp_path / "nope") is None


def test_prune_keeps_recent_and_milestones(tmp_path):
    for s in range(1000, 11000, 1000):
        ck.save(tmp_path, s, {"step": s})
    (tmp_path / "ckpt_00011000.pt.tmp").write_bytes(b"x")
    ck.prune(tmp_path, keep=3, keep_every=4000)
    left = [s for s, _ in ck.list_checkpoints(tmp_path)]
    assert left == [4000, 8000, 9000, 10000]
    assert not list(tmp_path.glob("*.tmp"))


def test_payload_loads_with_weights_only(tmp_path):
    """RNG 状态、嵌套 dict、字符串都要能在 weights_only=True 下加载。"""
    payload = {
        "rng": {"torch": torch.get_rng_state()},
        "loader": {"epoch": 1, "shard": 2, "offset": 3},
        "meta": {"git": "abc123", "config": {"lr": 7e-4, "betas": [0.9, 0.95]}},
    }
    p = ck.save(tmp_path, 1, payload)
    got = ck.load(p)
    assert got["loader"] == payload["loader"] and got["meta"] == payload["meta"]
