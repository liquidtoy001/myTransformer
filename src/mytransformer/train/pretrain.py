"""预训练入口。

    python -m mytransformer.train.pretrain --config configs/train/smoke_s1.yaml
    python -m mytransformer.train.pretrain --config ... --max-minutes 225   # Rangpur：时限 4h 减去余量

多次运行同一个配置是安全的：有 checkpoint 就续跑，已经训完就直接退出。
"""

from __future__ import annotations

import argparse
import sys

from .config import TrainConfig
from .trainer import Trainer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="预训练（可续跑）")
    ap.add_argument("--config", required=True, help="configs/train/*.yaml")
    ap.add_argument("--max-minutes", type=float, help="时间预算：快用完时存档退出")
    ap.add_argument("--device", help="cuda / cpu，默认有 GPU 就用 GPU")
    ap.add_argument("--run-dir", help="覆盖配置里的 run_dir")
    ap.add_argument("--max-steps", type=int, help="覆盖配置里的 max_steps")
    args = ap.parse_args(argv)

    cfg = TrainConfig.from_yaml(args.config, run_dir=args.run_dir, max_steps=args.max_steps)
    reason = Trainer(cfg, device=args.device, max_minutes=args.max_minutes).fit()
    print(f"退出原因：{reason}", flush=True)
    return 0  # 无论训完还是提前存档退出都返回 0：sbatch 接力用 afterany，不依赖退出码


if __name__ == "__main__":
    sys.exit(main())
