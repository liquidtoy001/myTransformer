"""在真实文档上检验去重（P2-2）。用的是 scripts/calibrate_filters.py --fetch 取下来的文档。

    python scripts/calibrate_dedup.py                       # 各来源：近似重复数、段落去重删了什么
    python scripts/calibrate_dedup.py --pr zh_wiki          # 另外和暴力枚举对比，算精确率和召回率

**精确率/召回率怎么算**：2000 篇文档两两算真实 Jaccard（约 200 万对，几分钟），
J ≥ 0.8 的连成簇，得到"理想情况下应该删掉哪些文档"，再和 MinHash-LSH 的结果比。
这就是"答案已知"的检验：暴力枚举在几百万篇上做不到，但在 2000 篇上可以。
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
from pathlib import Path

import numpy as np

from mytransformer.data.dedup import (
    PARAGRAPH_DEDUP_SKIP,
    MinHasher,
    jaccard,
    near_duplicates,
    paragraph_dedup,
    shingles,
)

DOCS = Path("data/filter_calib")


def load(sub: str) -> list[str]:
    return [json.loads(line)["text"] for line in (DOCS / f"{sub}.jsonl").open(encoding="utf-8")]


def overview() -> None:
    mh = MinHasher()
    for path in sorted(DOCS.glob("*.jsonl")):
        sub = path.stem
        docs = load(sub)
        by_char = sub.startswith("zh")
        rep = near_duplicates(np.stack([mh.signature(d, by_char=by_char) for d in docs]))
        n_dup = int(np.sum(rep != np.arange(len(docs))))
        line = f"{sub:12s} 近似重复 {n_dup:4d}/{len(docs)}"
        if sub in PARAGRAPH_DEDUP_SKIP:
            print(line + "  段落去重：不做")
            continue
        _, removed = paragraph_dedup(docs)
        counts = collections.Counter()
        for d in docs:
            counts.update({p.strip() for p in re.split(r"\n{2,}", d) if len(p.strip()) >= 30})
        top = [f"{c}× {p[:40]!r}" for p, c in counts.most_common(2) if c >= 3]
        print(f"{line}  段落去重删 {removed:4d} 段  最常见：{'; '.join(top)}")


def precision_recall(sub: str) -> None:
    docs = load(sub)
    by_char = sub.startswith("zh")
    sh = [shingles(d, by_char=by_char) for d in docs]
    size = [len(s) for s in sh]
    parent = list(range(len(docs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in itertools.combinations(range(len(docs)), 2):
        # 集合大小差 20% 以上时 J 不可能 ≥ 0.8，跳过省时间
        if min(size[i], size[j]) >= 0.8 * max(size[i], size[j]) and jaccard(sh[i], sh[j]) >= 0.8:
            a, b = find(i), find(j)
            parent[max(a, b)] = min(a, b)
    ideal = {i for i in range(len(docs)) if find(i) != i}
    print(f"[{sub}] 暴力枚举：理想情况下应删 {len(ideal)} 篇")
    for num_perm, bands in [(128, 16), (256, 32)]:
        mh = MinHasher(num_perm=num_perm)
        rep = near_duplicates(np.stack([mh.signature(d, by_char=by_char) for d in docs]), bands=bands)
        got = {i for i, r in enumerate(rep) if r != i}
        hit = len(got & ideal)
        print(f"  {num_perm} 个哈希 / {bands} 段：删 {len(got)}，精确率 {hit / max(1, len(got)):.0%}，"
              f"召回率 {hit / max(1, len(ideal)):.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", nargs="*", default=[], help="对这些来源做暴力枚举对比")
    args = ap.parse_args()
    overview()
    for sub in args.pr:
        precision_recall(sub)


if __name__ == "__main__":
    main()
