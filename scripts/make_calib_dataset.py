"""用校准文档（data/filter_calib，每个来源约 2000 篇）做一份迷你数据集，端到端检验 P2-4（分词 → 打包 → 训练）。

    python scripts/make_calib_dataset.py
    python -m mytransformer.train.pretrain --config configs/train/smoke_realtext.yaml

这份数据只有约 1000 万 token，文档没有经过过滤、去重、去污染（那是 P2-5 管线的事），
目的只是确认各环节接得上：分片能被训练读取、分词器指纹核对通过、每个子集的验证 loss 都在降。
配比取 docs/05-data.md 的 v1（指令类 4% 暂缺，其余按比例放大）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from mytransformer.data.pack import pack, tokenize_docs
from mytransformer.tokenizer import Tokenizer

SRC = Path("data/filter_calib")
OUT = Path("data/tokens_calib")
WEIGHTS = {"en_web": 40, "en_web_dclm": 12, "zh_web": 20, "code": 12, "math": 6, "en_wiki": 3, "zh_wiki": 3}


def main() -> None:
    tok = Tokenizer.from_file("tokenizer/mt32k.json")
    dirs = {}
    for name in WEIGHTS:
        texts = [json.loads(line)["text"] for line in (SRC / f"{name}.jsonl").open(encoding="utf-8")]
        t0 = time.time()
        info = tokenize_docs(tok, texts, OUT / "docs" / name)
        mb = sum(len(t.encode("utf-8")) for t in texts) / 1e6
        print(f"[{name}] {info['docs']} 篇 → {info['tokens']:,} tokens，{mb / (time.time() - t0):.1f} MB/s", flush=True)
        dirs[name] = OUT / "docs" / name
    meta = pack(dirs, WEIGHTS, OUT / "mix", total_tokens=10_000_000, val_tokens=100_000, shard_tokens=2_000_000)
    print(f"训练 {meta['train']['tokens']:,} tokens，{len(meta['train']['shards'])} 个分片")
    for name, v in meta["train"]["by_subset"].items():
        print(f"  {name:12s} 占比 {v['share']:.3f}（目标 {v['target_share']:.3f}）  用了 {v['epochs']:.2f} 轮")


if __name__ == "__main__":
    main()
