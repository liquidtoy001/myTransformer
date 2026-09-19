"""校准去污染（P2-3）。前提：已运行 scripts/fetch_eval_sets.py 和 scripts/calibrate_filters.py --fetch。

    python scripts/calibrate_decontam.py --scan-reference   # 在 1 GB 普通语料上统计通用片段，写排除列表
    python scripts/calibrate_decontam.py                    # 各来源有多少文档被判为污染，打印命中样本

**通用片段怎么来的**：用 P1 的分词器训练语料（data/tokenizer_sample/train，1 GB）当"普通语料"，
数每个评测 n-gram 在里面出现几次。出现 ≥ MIN_COUNT 次的，是名言、法律条文、固定表述这类到处都有的文字，
从索引里排除。结果写进 configs/data/decontam_exclude.json 并提交到仓库，之后跑管线不需要这 1 GB 语料。

**为什么不用校准文档本身来统计**：校准文档（data/filter_calib）是用来检验效果的，
用它来定规则再用它来检验，等于拿考题复习再考同一张卷子。两份数据来自不同的文件，互不重叠。
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mytransformer.data.decontam import EvalIndex

EVAL_DIR = "data/eval_sets"
EXCLUDE = Path("configs/data/decontam_exclude.json")
REFERENCE = "data/tokenizer_sample/train/*.txt"
MIN_COUNT = 3
_idx: EvalIndex | None = None


def _init() -> None:
    global _idx
    _idx = EvalIndex.from_dir(EVAL_DIR)


def _scan(path: str) -> collections.Counter:
    """按段落（空行分隔）扫一个文件，数每个评测 n-gram 命中几次。"""
    counts: collections.Counter = collections.Counter()
    buf: list[str] = []
    with open(path, encoding="utf-8") as f:  # 逐行读：一次读进几百 MB 的文件，7 个进程一起就会爆内存
        for line in f:
            if line.strip():
                buf.append(line)
            elif buf:
                counts.update(_idx.hits("".join(buf)))
                buf = []
    if buf:
        counts.update(_idx.hits("".join(buf)))
    return counts


def scan_reference(workers: int) -> None:
    t0 = time.time()
    total: collections.Counter = collections.Counter()
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        for c in ex.map(_scan, sorted(glob.glob(REFERENCE))):
            total.update(c)
    _init()
    common = [(g, k) for g, k in total.most_common() if k >= MIN_COUNT]
    EXCLUDE.parent.mkdir(parents=True, exist_ok=True)
    EXCLUDE.write_text(json.dumps({
        "about": f"评测 13-gram 中在 {REFERENCE}（P1 分词器语料，1 GB）里出现 ≥ {MIN_COUNT} 次的通用片段，"
                 "去污染时不使用。由 scripts/calibrate_decontam.py --scan-reference 生成",
        "min_count": MIN_COUNT,
        "ngrams": [g for g, _ in common],
        "counts": [k for _, k in common],
        "bench": [_idx.bench_of[g] for g, _ in common],
    }, ensure_ascii=False, indent=1), "utf-8")
    print(f"普通语料里命中过的评测 n-gram {len(total)} 个，其中 ≥ {MIN_COUNT} 次的 {len(common)} 个 → {EXCLUDE}"
          f"（{time.time() - t0:.0f} 秒）")
    for g, k in common[:10]:
        print(f"  {k:3d}× [{_idx.bench_of[g]}] {g[:80]}")


def measure() -> None:
    idx = EvalIndex.from_dir(EVAL_DIR, exclude_file=EXCLUDE if EXCLUDE.exists() else None)
    print(f"索引：{len(idx):,} 个 n-gram；太短没进索引的题：{dict(idx.too_short)}")
    for f in sorted(Path("data/filter_calib").glob("*.jsonl")):
        docs = [json.loads(line)["text"] for line in f.open(encoding="utf-8")]
        res = [idx.contaminated(d) for d in docs]
        c = collections.Counter(r for r in res if r)
        print(f"{f.stem:12s} 剔除 {sum(c.values()):3d}/{len(docs)}  {dict(c)}")
        for d, r in zip(docs, res):
            if r:
                h = idx.hits(d)
                print(f"      [{r}] 命中 {len(h)} 个，例：{h[0].replace(' ', '')[:40] if '一' <= h[0][0] <= '鿿' else h[0][:70]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-reference", action="store_true")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    if args.scan_reference:
        scan_reference(args.workers)
    measure()


if __name__ == "__main__":
    main()
