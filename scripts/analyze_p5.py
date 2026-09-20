"""P5 结果分析：种子方差、消融表、缩放律拟合与外推（读 runs/p5/*/metrics.jsonl）。

    python scripts/analyze_p5.py                    # 写 reports/ablation.md 和 reports/scaling.md
    python scripts/analyze_p5.py --runs runs/p5     # 指定目录（从 Rangpur scp 回来的）

**为什么先算种子方差**：同样的配置换个随机种子，val loss 本来就会差一点。
比这个差异还小的消融结果，说明不了任何事。这一步几乎没人做，但它决定了消融表里哪几行能下结论。

**缩放律**：用三个点拟合 L(N) = L∞ + A·N^(−α)，N 取**非嵌入参数**。
拟合方式：α 在网格上扫，每个 α 下 (L∞, A) 用最小二乘解——三个点、两个线性参数，
不需要 scipy，也不会陷进局部最优。三个点拟三个参数本来就勉强，报告里要写清这一点。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mytransformer.model import ModelConfig
from mytransformer.train.config import TrainConfig

CONFIGS = Path("configs/train/p5")
ABLATIONS = {
    "a1_muon": "A1 优化器：Muon vs AdamW",
    "a2_gelu": "A2 激活：GeLU vs SwiGLU（参数量相同）",
    "a3_mha": "A3 注意力：MHA vs GQA（MHA 多 13.9% 参数）",
    "a4_mix_v2": "A4 数据：中文 35% vs 24%",
    "a5_cosine": "A5 调度：cosine vs WSD",
    "a6_qknorm_no_zloss": "A6 稳定性：QK-Norm 且无 z-loss vs 无 QK-Norm 且有 z-loss",
}


def read_run(run_dir: Path) -> dict | None:
    """取这个 run 的最后一次评测，以及训练用时、最大梯度范数。"""
    f = run_dir / "metrics.jsonl"
    if not f.exists():
        return None
    evals, trains = [], []
    for line in f.read_text("utf-8").splitlines():
        r = json.loads(line)
        (evals if r.get("type") == "eval" else trains).append(r)
    if not evals:
        return None
    last = evals[-1]
    return {
        "name": run_dir.name,
        "val_loss": last["val_loss"],
        "per_subset": last.get("val", {}),
        "step": last["step"],
        "tokens": last["tokens"],
        "grad_norm_max": max((r.get("grad_norm", 0) or 0) for r in trains) if trains else 0.0,
        # step_s 是一个记录窗口内的平均每步秒数；乘总步数就是这次训练的大致耗时
        "hours": (float(np.mean([r["step_s"] for r in trains])) * last["step"] / 3600) if trains else 0.0,
    }


def fit_scaling(ns: np.ndarray, losses: np.ndarray) -> dict:
    """L(N) = L∞ + A·N^(−α)。α 扫网格，每个 α 下线性最小二乘解 (L∞, A)。"""
    best = None
    for alpha in np.linspace(0.01, 1.5, 3000):
        x = ns.astype(float) ** (-alpha)
        design = np.stack([np.ones_like(x), x], axis=1)
        coef, *_ = np.linalg.lstsq(design, losses, rcond=None)
        resid = float(np.sum((design @ coef - losses) ** 2))
        if best is None or resid < best["resid"]:
            best = {"alpha": float(alpha), "L_inf": float(coef[0]), "A": float(coef[1]), "resid": resid}
    return best


def predict(fit: dict, n: float) -> float:
    return fit["L_inf"] + fit["A"] * n ** (-fit["alpha"])


def ablation_report(runs: dict[str, dict]) -> list[str]:
    seeds = [runs[f"base_seed{i}"]["val_loss"] for i in range(3) if f"base_seed{i}" in runs]
    lines = ["# P5 消融实验", ""]
    if len(seeds) < 2:
        return lines + ["种子方差还没跑完（需要 base_seed0/1/2），消融结果暂时无法判断显著性。", ""]

    mean, std = float(np.mean(seeds)), float(np.std(seeds, ddof=1))
    spread = max(seeds) - min(seeds)
    lines += [
        "## 一、先看噪声：同配置换种子能差多少", "",
        f"`base_seed0/1/2` 三次完全相同的训练，只换随机种子：val loss "
        f"{' / '.join(f'{s:.4f}' for s in seeds)}",
        "",
        f"- 均值 **{mean:.4f}**，标准差 **{std:.4f}**，极差 {spread:.4f}",
        f"- **判据：消融的差异小于 {2 * std:.4f}（2σ）就不下结论**，这条线以下的行只能写成看不出差别",
        "",
        "## 二、消融表", "",
        "| 实验 | val loss | 与基线之差 | 是否超过 2σ | 中文 val | 英文 val | 备注 |",
        "|---|---:|---:|---|---:|---:|---|",
        f"| 基线（3 个种子均值） | {mean:.4f} | — | — | — | — | ladder_s2 × 0.79B tokens |",
    ]
    for key, label in ABLATIONS.items():
        r = runs.get(key)
        if not r:
            lines.append(f"| {label} | 未跑 | | | | | |")
            continue
        delta = r["val_loss"] - mean
        sig = "**是**" if abs(delta) > 2 * std else "否（噪声内）"
        zh = np.mean([v for k, v in r["per_subset"].items() if k.startswith("zh")]) if r["per_subset"] else float("nan")
        en = np.mean([v for k, v in r["per_subset"].items() if k.startswith("en")]) if r["per_subset"] else float("nan")
        note = "梯度范数峰值 %.1f" % r["grad_norm_max"] if key == "a6_qknorm_no_zloss" else ""
        lines.append(f"| {label} | {r['val_loss']:.4f} | {delta:+.4f} | {sig} | {zh:.4f} | {en:.4f} | {note} |")
    lines += ["", "> 下游任务分（HellaSwag 等）在 P8 统一评测：39M 的模型在这些任务上基本是随机水平，"
                  "这里用 val loss 和分子集 val loss 做判断。", ""]
    return lines


def scaling_report(runs: dict[str, dict]) -> list[str]:
    points = []
    for i in (1, 2, 3):
        r = runs.get(f"ladder_s{i}")
        if not r:
            continue
        m = ModelConfig.from_yaml(TrainConfig.from_yaml(CONFIGS / f"ladder_s{i}.yaml").model)
        points.append((m.count_params()["non_embedding"], r["val_loss"], f"ladder_s{i}", r["tokens"]))
    lines = ["# P5 缩放律", ""]
    if len(points) < 3:
        return lines + [f"三个点还没跑完（当前 {len(points)} 个），无法拟合。", ""]

    ns = np.array([p[0] for p in points], dtype=float)
    ls = np.array([p[1] for p in points])
    fit = fit_scaling(ns, ls)
    target = ModelConfig.from_yaml("configs/model/250m.yaml").count_params()["non_embedding"]
    lines += [
        "## 一、三个点", "",
        "| 模型 | 非嵌入参数 | 训练 tokens | val loss | 拟合值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for (n, loss, name, toks) in points:
        lines.append(f"| {name} | {n / 1e6:.2f}M | {toks / 1e6:.0f}M | {loss:.4f} | {predict(fit, n):.4f} |")
    hours = sum(runs[f"ladder_s{i}"]["hours"] for i in (1, 2, 3) if f"ladder_s{i}" in runs)
    lines.append(f"\n三个点合计约 {hours:.1f} A100 小时。")
    lines += [
        "",
        "## 二、拟合", "",
        "```",
        f"L(N) = {fit['L_inf']:.4f} + {fit['A']:.3f} · N^(-{fit['alpha']:.4f})",
        f"残差平方和 {fit['resid']:.2e}",
        "```",
        "",
        f"**外推到 250M 模型**（非嵌入 {target / 1e6:.1f}M，按同样的 20 tokens/参数）："
        f"预测 val loss ≈ **{predict(fit, target):.3f}**",
        "",
        "> 三个点拟三个参数，拟合必然贴得很紧，但这不代表外推准。P6 训完 250M 后回来对照，"
        "差多少要如实写进报告——这一条比拟合本身更有价值。", "",
    ]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/p5")
    args = ap.parse_args()
    runs = {}
    for d in sorted(Path(args.runs).glob("*")):
        r = read_run(d)
        if r:
            runs[d.name] = r
    if not runs:
        raise SystemExit(f"{args.runs} 下没有找到任何 metrics.jsonl（先从 Rangpur scp 回来）")
    print(f"读到 {len(runs)} 个 run：{', '.join(sorted(runs))}")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/ablation.md").write_text("\n".join(ablation_report(runs)), "utf-8")
    Path("reports/scaling.md").write_text("\n".join(scaling_report(runs)), "utf-8")
    print("已写 reports/ablation.md、reports/scaling.md")


if __name__ == "__main__":
    main()
