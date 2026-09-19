"""预训练入口。

    python -m mytransformer.train.pretrain --config configs/train/smoke_s1.yaml
    python -m mytransformer.train.pretrain --config ... --max-minutes 225        # Rangpur：4h 时限减去余量
    python -m mytransformer.train.pretrain --config ... --set compile=true --set lr=1e-3

多次运行同一个配置是安全的：有 checkpoint 就续跑，已经训完就直接退出。
"""

from __future__ import annotations

import argparse
import sys

import yaml

from .config import TrainConfig
from .trainer import Trainer


def parse_overrides(items: list[str]) -> dict:
    """把 ["compile=true", "lr=1e-3"] 解析成 {"compile": True, "lr": 0.001}。

    值按 YAML 解析（true/false/数字/列表都能用）。PyYAML 不认 "1e-3" 这种不带小数点的
    科学计数法、会当成字符串，所以字符串再尝试一次 float。
    """
    out = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--set 需要 键=值 的形式，得到 {item!r}")
        value = yaml.safe_load(raw)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        out[key.strip()] = value
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="预训练（可续跑）")
    ap.add_argument("--config", required=True, help="configs/train/*.yaml")
    ap.add_argument("--max-minutes", type=float, help="时间预算：快用完时存档退出")
    ap.add_argument("--device", help="cuda / cpu，默认有 GPU 就用 GPU")
    ap.add_argument("--set", action="append", default=[], metavar="键=值",
                    help="覆盖配置里的字段，可重复；字段名拼错会直接报错")
    args = ap.parse_args(argv)

    cfg = TrainConfig.from_yaml(args.config, **parse_overrides(args.set))
    reason = Trainer(cfg, device=args.device, max_minutes=args.max_minutes).fit()
    print(f"退出原因：{reason}", flush=True)
    return 0  # 无论训完还是提前存档退出都返回 0：sbatch 接力用 afterany，不依赖退出码


if __name__ == "__main__":
    sys.exit(main())
