from __future__ import annotations

import math

import pytest

from mytransformer.train.schedule import LRSchedule


def test_wsd_shape():
    s = LRSchedule(peak_lr=1.0, warmup_steps=10, total_steps=100)  # 默认 90 步开始衰减
    assert s(0) == pytest.approx(0.1)
    assert s(9) == pytest.approx(1.0)               # warmup 最后一步到达峰值
    assert all(s(i) == 1.0 for i in range(10, 90))  # 稳定段恒定
    assert s(95) == pytest.approx(0.5)              # 衰减段线性
    assert s(100) == 0.0 and s(10_000) == 0.0


def test_wsd_is_monotone_in_decay():
    s = LRSchedule(peak_lr=7e-4, warmup_steps=720, total_steps=18000, decay_start=16200)
    lrs = [s(i) for i in range(16200, 18001)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_wsd_fork_matches_main_run_until_fork():
    """缩放律第 4 点：从 step 9021 分叉做 900 步衰减。分叉前两条调度必须完全一样。"""
    main = LRSchedule(peak_lr=7e-4, warmup_steps=720, total_steps=18000, decay_start=16200)
    fork = LRSchedule(peak_lr=7e-4, warmup_steps=720, total_steps=9921, decay_start=9021)
    assert all(main(i) == fork(i) for i in range(9021))
    assert fork(9921) == 0.0 and main(9921) == 7e-4


def test_cosine_endpoints():
    s = LRSchedule(peak_lr=1.0, warmup_steps=0, total_steps=100, kind="cosine", min_lr_ratio=0.1)
    assert s(0) == pytest.approx(1.0)
    assert s(50) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi / 2)))
    assert s(100) == pytest.approx(0.1)


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        LRSchedule(peak_lr=1.0, warmup_steps=10, total_steps=100, kind="linear")
    with pytest.raises(ValueError):
        LRSchedule(peak_lr=1.0, warmup_steps=50, total_steps=100, decay_start=20)
