"""P5 结果分析：种子方差、消融表、缩放律拟合与外推（读 runs/p5/*/metrics.jsonl）。

    python scripts/analyze_p5.py                    # 写 reports/ablation.md 和 reports/scaling.md
    python scripts/analyze_p5.py --runs runs/p5     # 指定目录（从 Rangpur scp 回来的）

**为什么先算种子方差**：同样的配置换个随机种子，val loss 本来就会差一点。
比这个差异还小的消融结果，说明不了任何事。这一步几乎没人做，但它决定了消融表里哪几行能下结论。

**为什么要分子集看**：总 val loss 是 7 个子集的等权平均。一个改动让中文大幅变好、英文略微变差，
总数照样会"变好"（A4 就是这样）。所以每个子集单独算差值，并且用**这个子集自己**的种子噪声判断——
各子集的噪声能差 7 倍（code 0.011，en_web_dclm 0.0015）。

**缩放律**：拟合 L(x) = L∞ + A·x^(−α)。α 在网格上扫，每个 α 下 (L∞, A) 用最小二乘解，不需要 scipy。
但三个点拟三个参数必然"完全拟合"，残差为 0 什么也证明不了。所以同时给出几种拟合（横轴用非嵌入参数、
总参数、计算量；有无不可约项），**看外推结果是否稳定**。不稳定就如实写"预测不了"。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mytransformer.model import ModelConfig
from mytransformer.train.config import TrainConfig

CONFIGS = Path("configs/train/p5")
HAND_MARKER = "<!-- 以下为手写，重新生成报告时保留 -->"
ABLATIONS = {
    "a1_muon": "A1 优化器：Muon vs AdamW",
    "a2_gelu": "A2 激活：GeLU vs SwiGLU（参数量相同）",
    "a3_mha": "A3 注意力：MHA vs GQA（MHA 多 13.9% 参数）",
    "a4_mix_v2": "A4 数据：中文 35% vs 24%",
    "a5_cosine": "A5 调度：cosine vs WSD",
    "a6_qknorm_no_zloss": "A6 稳定性：QK-Norm 且无 z-loss vs 无 QK-Norm 且有 z-loss",
}
ALPHA_GRID = (0.01, 1.5)
PLAUSIBLE_ALPHA = (0.05, 0.6)   # 文献里 L(N)、L(C) 的指数大致在这个范围（Kaplan 0.076，Chinchilla 约 0.34）


def read_run(run_dir: Path) -> dict | None:
    """取这个 run 的最后一次评测，以及训练用时、梯度范数。"""
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
    # warmup 期间梯度范数本来就大，只看第 200 步之后的峰值才能反映训练是否平稳
    late = [r.get("grad_norm", 0) or 0 for r in trains if r["step"] > 200]
    return {
        "name": run_dir.name,
        "val_loss": last["val_loss"],
        "per_subset": last.get("val", {}),
        "step": last["step"],
        "tokens": last["tokens"],
        "grad_norm_max": max((r.get("grad_norm", 0) or 0) for r in trains) if trains else 0.0,
        "grad_norm_late_max": max(late) if late else 0.0,
        # step_s 是一个记录窗口内的平均每步秒数；乘总步数就是这次训练的大致耗时
        "hours": (float(np.mean([r["step_s"] for r in trains])) * last["step"] / 3600) if trains else 0.0,
    }


def fit_scaling(ns: np.ndarray, losses: np.ndarray) -> dict:
    """L(N) = L∞ + A·N^(−α)。α 扫网格，每个 α 下线性最小二乘解 (L∞, A)。"""
    best = None
    for alpha in np.linspace(*ALPHA_GRID, 3000):
        x = ns.astype(float) ** (-alpha)
        design = np.stack([np.ones_like(x), x], axis=1)
        coef, *_ = np.linalg.lstsq(design, losses, rcond=None)
        resid = float(np.sum((design @ coef - losses) ** 2))
        if best is None or resid < best["resid"]:
            best = {"alpha": float(alpha), "L_inf": float(coef[0]), "A": float(coef[1]), "resid": resid}
    return best


def fit_power(xs: np.ndarray, losses: np.ndarray) -> dict:
    """两参数幂律 L = A·x^(−α)（没有不可约项），在对数坐标里就是一条直线。
    三个点拟两个参数还剩一个自由度，残差能说明"数据像不像幂律"。"""
    slope, intercept = np.polyfit(np.log(xs), np.log(losses), 1)
    resid = np.exp(np.polyval([slope, intercept], np.log(xs))) - losses
    return {"alpha": float(-slope), "logA": float(intercept), "max_resid": float(np.abs(resid).max())}


def predict(fit: dict, n: float) -> float:
    return fit["L_inf"] + fit["A"] * n ** (-fit["alpha"])


def predict_power(fit: dict, x: float) -> float:
    return float(np.exp(fit["logA"] - fit["alpha"] * np.log(x)))


def _keep_hand_written(path: Path, lines: list[str]) -> None:
    """报告里 HAND_MARKER 之后是手写的结论和注意事项，重新生成时保留。"""
    lines = lines + ["", HAND_MARKER]
    old = path.read_text("utf-8") if path.exists() else ""
    if HAND_MARKER in old:
        lines.append(old.split(HAND_MARKER, 1)[1].lstrip("\n"))
    path.write_text("\n".join(lines), "utf-8")


# ---------------------------------------------------------------- 消融


def ablation_report(runs: dict[str, dict]) -> list[str]:
    seed_runs = [runs[f"base_seed{i}"] for i in range(3) if f"base_seed{i}" in runs]
    seeds = [r["val_loss"] for r in seed_runs]
    lines = ["# P5 消融实验", ""]
    if len(seeds) < 2:
        return lines + ["种子方差还没跑完（需要 base_seed0/1/2），消融结果暂时无法判断显著性。", ""]

    mean, std = float(np.mean(seeds)), float(np.std(seeds, ddof=1))
    subsets = list(seed_runs[0]["per_subset"])
    sub_mean = {s: float(np.mean([r["per_subset"][s] for r in seed_runs])) for s in subsets}
    sub_std = {s: float(np.std([r["per_subset"][s] for r in seed_runs], ddof=1)) for s in subsets}

    def group(values: dict, prefix: str) -> float:
        vals = [v for k, v in values.items() if k.startswith(prefix)]
        return float(np.mean(vals)) if vals else float("nan")

    base_zh, base_en = group(sub_mean, "zh"), group(sub_mean, "en")
    base_gn = max(r["grad_norm_late_max"] for r in seed_runs)
    lines += [
        "## 一、先看噪声：同配置换种子能差多少", "",
        f"`base_seed0/1/2` 三次完全相同的训练，只换随机种子：val loss "
        f"{' / '.join(f'{s:.4f}' for s in seeds)}",
        "",
        f"- 均值 **{mean:.4f}**，标准差 **{std:.4f}**，极差 {max(seeds) - min(seeds):.4f}",
        f"- **判据：消融的差异小于 {2 * std:.4f}（2σ）就不下结论**，这条线以下的行只能写成看不出差别",
        "",
        "各子集的噪声差别很大，分子集比较时用各自的 σ：", "",
        "| 子集 | 基线均值 | σ（3 个种子） |",
        "|---|---:|---:|",
        *[f"| {s} | {sub_mean[s]:.4f} | {sub_std[s]:.4f} |" for s in subsets],
        "",
        "## 二、消融表（总 val loss）", "",
        "| 实验 | val loss | 与基线之差 | 是否超过 2σ | 中文 val | 英文 val | 第 200 步后梯度范数峰值 |",
        "|---|---:|---:|---|---:|---:|---:|",
        f"| 基线（3 个种子均值） | {mean:.4f} | — | — | {base_zh:.4f} | {base_en:.4f} | ≤ {base_gn:.2f} |",
    ]
    for key, label in ABLATIONS.items():
        r = runs.get(key)
        if not r:
            lines.append(f"| {label} | 未跑 | | | | | |")
            continue
        delta = r["val_loss"] - mean
        sig = "**是**" if abs(delta) > 2 * std else "否（噪声内）"
        zh, en = group(r["per_subset"], "zh"), group(r["per_subset"], "en")
        lines.append(f"| {label} | {r['val_loss']:.4f} | {delta:+.4f} | {sig} | {zh:.4f} | {en:.4f} | "
                     f"{r['grad_norm_late_max']:.2f} |")

    lines += ["", "## 三、分子集的差值", "",
              "每格是该实验减去基线均值；**加粗**表示超过这个子集自己的 2σ。负数表示更好。", "",
              "| 实验 | " + " | ".join(subsets) + " |",
              "|---|" + "---:|" * len(subsets)]
    for key, label in ABLATIONS.items():
        r = runs.get(key)
        if not r or not r["per_subset"]:
            continue
        cells = []
        for s in subsets:
            d = r["per_subset"][s] - sub_mean[s]
            cells.append(f"**{d:+.3f}**" if abs(d) > 2 * sub_std[s] else f"{d:+.3f}")
        lines.append(f"| {label.split('：')[0]} | " + " | ".join(cells) + " |")
    lines += ["", "> 下游任务分（HellaSwag 等）在 P8 统一评测：39M 的模型在这些任务上基本是随机水平，"
                  "这里用 val loss 和分子集 val loss 做判断。", "",
              "> 梯度范数每 `log_every` 步才记一次，两次记录之间的尖峰看不到，这一列是抽样出来的峰值。"]
    return lines


# ---------------------------------------------------------------- 缩放律


def scaling_report(runs: dict[str, dict]) -> list[str]:
    points = []
    for i in (1, 2, 3):
        r = runs.get(f"ladder_s{i}")
        if not r:
            continue
        cfg = TrainConfig.from_yaml(CONFIGS / f"ladder_s{i}.yaml")
        m = ModelConfig.from_yaml(cfg.model)
        n = m.count_params()
        points.append({"name": f"ladder_s{i}", "loss": r["val_loss"], "tokens": r["tokens"],
                       "non_embedding": n["non_embedding"], "total": n["total"],
                       "flops": m.flops_per_token(cfg.seq_len) * r["tokens"], "steps": cfg.max_steps,
                       "warmup": cfg.warmup_steps, "hours": r["hours"]})
    lines = ["# P5 缩放律", ""]
    if len(points) < 3:
        return lines + [f"三个点还没跑完（当前 {len(points)} 个），无法拟合。", ""]

    big = ModelConfig.from_yaml("configs/model/250m.yaml")
    bn = big.count_params()
    big_tokens = 20 * bn["non_embedding"]
    target = {"non_embedding": bn["non_embedding"], "total": bn["total"],
              "flops": big.flops_per_token(2048) * big_tokens}
    ls = np.array([p["loss"] for p in points])

    lines += [
        "## 一、三个点", "",
        "| 模型 | 非嵌入参数 | 总参数 | 训练 tokens | 步数（其中 warmup） | val loss |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {p['name']} | {p['non_embedding'] / 1e6:.2f}M | {p['total'] / 1e6:.2f}M | {p['tokens'] / 1e6:.0f}M | "
          f"{p['steps']}（{p['warmup']}，{p['warmup'] / p['steps']:.0%}） | {p['loss']:.4f} |" for p in points],
        "",
        f"三个点合计约 {sum(p['hours'] for p in points):.1f} A100 小时。",
        "",
        "## 二、几种拟合并排比较", "",
        f"外推目标：250M 模型（非嵌入 {bn['non_embedding'] / 1e6:.1f}M、总参数 {bn['total'] / 1e6:.1f}M，"
        f"按同样的 20 tokens/参数训练 {big_tokens / 1e9:.2f}B tokens）。",
        "",
        "| 横轴 | 形式 | α | L∞ | 拟合残差 | 外推 250M | 可信吗 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    preds = []
    for key, label in [("non_embedding", "非嵌入参数"), ("total", "总参数"), ("flops", "计算量 C")]:
        xs = np.array([p[key] for p in points], dtype=float)
        f3 = fit_scaling(xs, ls)
        p3 = predict(f3, target[key])
        at_edge = abs(f3["alpha"] - ALPHA_GRID[1]) < 1e-3 or abs(f3["alpha"] - ALPHA_GRID[0]) < 1e-3
        ok3 = PLAUSIBLE_ALPHA[0] <= f3["alpha"] <= PLAUSIBLE_ALPHA[1] and not at_edge
        note3 = "α 撞到搜索上限" if at_edge else ("α 在文献范围内" if ok3 else "α 远超文献范围")
        lines.append(f"| {label} | L∞ + A·x^−α | {f3['alpha']:.3f} | {f3['L_inf']:.3f} | 0（必然） | {p3:.3f} | {note3} |")
        f2 = fit_power(xs, ls)
        p2 = predict_power(f2, target[key])
        note2 = "残差太大，不是幂律" if f2["max_resid"] > 0.05 else "拟合尚可"
        lines.append(f"| {label} | A·x^−α | {f2['alpha']:.3f} | — | ±{f2['max_resid']:.3f} | {p2:.3f} | {note2} |")
        preds += [p3, p2]
    spread = max(preds) - min(preds)
    lines += [
        "",
        f"**外推结果的范围：{min(preds):.2f} – {max(preds):.2f}（相差 {spread:.2f}）。**",
        "",
        ("这个范围太宽，**用这三个点预测不了 250M 的 loss**。原因分析见下面的手写部分。"
         if spread > 0.1 else "几种拟合给出的外推相近，预测可信度尚可。"),
        "",
        "> 三个点拟三个参数，残差必然为 0，这本身什么也证明不了。P6 训完 250M 后回来对照。",
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
    _keep_hand_written(Path("reports/ablation.md"), ablation_report(runs))
    _keep_hand_written(Path("reports/scaling.md"), scaling_report(runs))
    print("已写 reports/ablation.md、reports/scaling.md")


if __name__ == "__main__":
    main()
