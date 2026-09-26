"""按**实测**的每步耗时，算出 P5 每个实验要几个作业、每个作业申请多长时间。

    python scripts/rangpur/p5_plan.py            # 打印计划表
    python scripts/rangpur/p5_plan.py --lines    # 给 submit_p5.sh 用：每行 "配置 作业分钟数 训练分钟数 作业数"

为什么要单独算：一开始按"MFU 25%"估时，实际 ladder_s1 只有 11%，作业在 303/360 步被截断。
**估时只能用实测数**（Rangpur A100-PCIE-40GB，2026-09-24，作业 613670/613679）。
ladder_s3 还没跑过，按 FLOPs 比例从 ladder_s2 推算，并留了余量；跑过之后把实测值填回来。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from mytransformer.model import ModelConfig
from mytransformer.train.config import TrainConfig

P5 = Path("configs/train/p5")
MEASURED = {"ladder_s1": 2.84, "ladder_s2": 3.37}  # 秒/步
MEASURED_BATCH = 524288                             # 上面两个数是在这个 global batch 下测的
JOB_LIMIT_MIN = 240       # 单个作业最多申请 4 小时（接力比一次申请 12 小时更容易排上）
STARTUP_MIN = 10          # 启动、编译、最终评测、存档
MARGIN = 1.15             # 每步耗时的余量：别人的作业、NFS 抖动都会让它变慢


def seconds_per_step(cfg: TrainConfig) -> float:
    model = ModelConfig.from_yaml(cfg.model)
    name = Path(cfg.model).stem
    if name in MEASURED:
        base = MEASURED[name]
    else:
        # 没实测过的模型：按每 token 的 FLOPs 从 ladder_s2 推算（假设 MFU 相同，大模型实际会略高，偏保守）
        ref = ModelConfig.from_yaml("configs/model/ladder_s2.yaml")
        base = MEASURED["ladder_s2"] * model.flops_per_token(cfg.seq_len) / ref.flops_per_token(cfg.seq_len)
    if cfg.optimizer == "muon":
        base *= 1.05  # Newton-Schulz 迭代的额外开销
    # 实测值都是在每步 524288 tokens 下测的；batch 变小，每步的累积次数按比例减少
    return base * cfg.global_batch_tokens / MEASURED_BATCH


def plan() -> list[tuple[str, int, int, int, float, str]]:
    rows = []
    for f in sorted(P5.glob("*.yaml")):
        if f.name.startswith("_"):
            continue
        cfg = TrainConfig.from_yaml(f)
        train_min = cfg.max_steps * seconds_per_step(cfg) * MARGIN / 60
        budget = JOB_LIMIT_MIN - STARTUP_MIN - 5            # 每个作业里用于训练的分钟数
        jobs = max(1, math.ceil(train_min / budget))
        job_min = min(JOB_LIMIT_MIN, math.ceil(train_min / jobs) + STARTUP_MIN + 5)
        rows.append((str(f).replace("\\", "/"), job_min, job_min - STARTUP_MIN, jobs, train_min / 60,
                     cfg.run_dir))
    # 提交顺序：先补 ladder_s1/s2 的最终评测（几分钟），再跑种子和消融，最长的 ladder_s3 放最后
    order = {"ladder_s1": 0, "ladder_s2": 1, "base_seed0": 2, "base_seed1": 3, "base_seed2": 4, "ladder_s3": 99}
    rows.sort(key=lambda r: (order.get(Path(r[0]).stem, 10), r[0]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", action="store_true",
                    help="每行：配置 作业分钟数 训练分钟数 作业数 run_dir")
    args = ap.parse_args()
    rows = plan()
    if args.lines:
        for cfg, job_min, train_min, jobs, _, run_dir in rows:
            print(cfg, job_min, train_min, jobs, run_dir)
        return
    total = 0.0
    print(f"{'配置':42s} {'作业数':>4s} {'每个作业':>8s} {'训练时长':>8s}")
    for cfg, job_min, _, jobs, hours, _ in rows:
        total += jobs * job_min / 60
        print(f"{cfg:42s} {jobs:4d} {job_min:6d} 分 {hours:6.2f} 时")
    print(f"合计申请约 {total:.1f} A100 小时")


if __name__ == "__main__":
    main()
