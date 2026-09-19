from __future__ import annotations

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
