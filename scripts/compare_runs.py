"""比较两个 run 的 metrics.jsonl：同一步的训练 loss 差多少。

    python scripts/compare_runs.py runs/ddp_check/g1 runs/ddp_check/g2 --rtol 0.02

用途：多卡验收（1 卡 vs 2 卡、中断续跑 vs 不中断）。GPU 上 bf16 + 非确定性 kernel，
做不到逐位相同，所以比相对误差；本机 CPU 的逐位测试见 tests/test_ddp.py。
退出码 0 表示所有共同的 step 都在容差内。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def losses(run: Path) -> dict[int, float]:
    out = {}
    for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("type") == "train":
            out[r["step"]] = r["loss"]  # 续跑时同一步可能记了两次，以后一次为准
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--rtol", type=float, default=0.02)
    ap.add_argument("--from-step", type=int, default=1, help="只比这一步及之后")
    args = ap.parse_args()

    la, lb = losses(args.a), losses(args.b)
    steps = sorted(s for s in la.keys() & lb.keys() if s >= args.from_step)
    if not steps:
        print("两个 run 没有共同的 step")
        return 1
    worst = max(steps, key=lambda s: abs(la[s] - lb[s]) / abs(la[s]))
    rel = abs(la[worst] - lb[worst]) / abs(la[worst])
    print(f"比较 {len(steps)} 步（{steps[0]}–{steps[-1]}）；最大相对差在 step {worst}："
          f"{la[worst]:.4f} vs {lb[worst]:.4f}（{rel:.2%}，容差 {args.rtol:.0%}）")
    print(f"最后一步 {steps[-1]}：{la[steps[-1]]:.4f} vs {lb[steps[-1]]:.4f}")
    return 0 if rel <= args.rtol else 1


if __name__ == "__main__":
    sys.exit(main())
