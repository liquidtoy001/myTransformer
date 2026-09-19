"""生成可学习的合成数据，用来在分词器和真实数据就绪之前验证训练框架。

数据是一条稀疏马尔可夫链：每个 token 恰好有 branch 个不同的后继，等概率随机选一个。
所以一个完美的模型 loss 能降到 ln(branch)——理论下界已知，训练有没有真的在学一目了然：

    刚初始化     ≈ ln(vocab) = ln(32768) ≈ 10.40
    学会了规律   → ln(branch) = ln(4)   ≈ 1.386

用法:
    python scripts/make_synthetic_data.py --out data/synthetic
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from mytransformer.data.shards import write_shard

EOT = 0  # 文档分隔符，与真实数据的约定一致


def successors(states: int, branch: int, rng: np.random.Generator) -> np.ndarray:
    """每个状态（1..states）恰好 branch 个**互不相同**的后继。有重复的话实际分支更少，下界就不是 ln(branch) 了。"""
    table = np.argsort(rng.random((states + 1, states)), axis=1)[:, :branch] + 1
    table[0] = 0  # 行 0 是 EOT，不会被当作当前状态使用
    return table


def generate(n_tokens: int, succ: np.ndarray, chain_len: int, rng: np.random.Generator) -> np.ndarray:
    """并行走很多条链（按时间步向量化），每条链前面放一个 EOT。"""
    states, branch = succ.shape[0] - 1, succ.shape[1]
    n_chains = math.ceil(n_tokens / (chain_len + 1))
    out = np.empty((n_chains, chain_len + 1), dtype=np.uint16)
    out[:, 0] = EOT
    cur = rng.integers(1, states + 1, n_chains)
    for t in range(chain_len):
        out[:, t + 1] = cur
        cur = succ[cur, rng.integers(0, branch, n_chains)]
    return out.reshape(-1)[:n_tokens]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--train-tokens", type=int, default=20_000_000)
    ap.add_argument("--val-tokens", type=int, default=1_000_000)
    ap.add_argument("--states", type=int, default=2048, help="用到的 token 种类数（必须 < 词表大小）")
    ap.add_argument("--branch", type=int, default=4)
    ap.add_argument("--chain-len", type=int, default=2048)
    ap.add_argument("--shard-tokens", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    succ = successors(args.states, args.branch, rng)
    out = Path(args.out)

    train = generate(args.train_tokens, succ, args.chain_len, rng)
    for i, start in enumerate(range(0, len(train), args.shard_tokens)):
        write_shard(out / f"train_{i:03d}.bin", train[start : start + args.shard_tokens])
    # 验证集用同一张转移表、不同的随机游走
    write_shard(out / "val_000.bin", generate(args.val_tokens, succ, args.chain_len, np.random.default_rng(args.seed + 1)))

    print(f"已写入 {out}：训练 {len(train):,} tokens（{math.ceil(len(train) / args.shard_tokens)} 个分片），验证 {args.val_tokens:,} tokens")
    print(f"loss 理论下界 ≈ ln({args.branch}) = {math.log(args.branch):.3f}"
          f"（EOT 之后的第一个 token 无法预测，实际会略高约 {math.log(args.states) / (args.chain_len + 1):.3f}）")


if __name__ == "__main__":
    main()
