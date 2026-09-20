"""P5 分析脚本：缩放律拟合能还原已知的参数，显著性判据按 2σ 工作。

用"答案已知"的方式检验：按 L(N) = 1.9 + 12·N^(−0.28) 造三个点，拟合必须把这三个数找回来。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_p5", ROOT / "scripts" / "analyze_p5.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def write_run(root: Path, name: str, val_loss: float, steps: int = 1507, grad_norm: float = 1.0) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    per = {k: val_loss + off for k, off in
           [("en_web", -0.3), ("zh_web", 0.4), ("code", 0.1), ("math", -0.1)]}
    recs = [{"type": "train", "step": s, "tokens": s * 524288, "loss": val_loss + 1.0,
             "lr": 2e-3, "grad_norm": grad_norm, "step_s": 2.0, "tok_s": 1e5}
            for s in range(20, steps + 1, 20)]
    recs.append({"type": "eval", "step": steps, "tokens": steps * 524288,
                 "val_loss": val_loss, "val": per})
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in recs), "utf-8")


def test_fit_recovers_known_scaling_law(mod):
    ns = np.array([9.44e6, 22.55e6, 74.34e6])
    losses = 1.9 + 12 * ns ** -0.28
    fit = mod.fit_scaling(ns, losses)
    assert fit["alpha"] == pytest.approx(0.28, abs=0.01)
    assert fit["L_inf"] == pytest.approx(1.9, abs=0.01)
    assert fit["A"] == pytest.approx(12, rel=0.05)


def test_extrapolation_is_monotone(mod):
    """模型越大，预测 loss 越低，且不会低过 L∞。"""
    fit = {"alpha": 0.28, "L_inf": 1.9, "A": 12.0}
    preds = [mod.predict(fit, n) for n in (1e7, 1e8, 1e9, 1e12)]
    assert preds == sorted(preds, reverse=True)
    assert preds[-1] > fit["L_inf"]


def test_ablation_table_flags_only_effects_above_two_sigma(mod, tmp_path):
    for i, v in enumerate([3.4102, 3.4185, 3.4131]):   # σ ≈ 0.0042，2σ ≈ 0.0084
        write_run(tmp_path, f"base_seed{i}", v)
    write_run(tmp_path, "a1_muon", 3.3712)             # 明显更好
    write_run(tmp_path, "a5_cosine", 3.4150)           # 噪声以内
    runs = {d.name: mod.read_run(d) for d in tmp_path.iterdir()}
    text = "\n".join(mod.ablation_report(runs))

    assert "标准差 **0.0042**" in text
    a1 = next(line for line in text.splitlines() if "A1" in line)
    a5 = next(line for line in text.splitlines() if "A5" in line)
    assert "**是**" in a1 and "-0.0427" in a1
    assert "否（噪声内）" in a5
    assert "A2" in text and "未跑" in text            # 没跑的实验要显式标出来，不能悄悄消失


def test_report_refuses_to_fit_with_too_few_points(mod, tmp_path):
    write_run(tmp_path, "ladder_s1", 3.0)
    runs = {d.name: mod.read_run(d) for d in tmp_path.iterdir()}
    assert "无法拟合" in "\n".join(mod.scaling_report(runs))


def test_seed_variance_needs_at_least_two_runs(mod, tmp_path):
    write_run(tmp_path, "base_seed0", 3.4)
    runs = {d.name: mod.read_run(d) for d in tmp_path.iterdir()}
    assert "种子方差还没跑完" in "\n".join(mod.ablation_report(runs))
