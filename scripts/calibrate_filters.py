"""在真实文档上测质量过滤规则：每个来源丢了多少、各条规则丢了多少、被丢的长什么样（P2-1）。

    python scripts/calibrate_filters.py --fetch          # 第一次：从 HF 取每个来源约 2000 篇文档
    python scripts/calibrate_filters.py                  # 之后改了规则只需重跑这一句
    python scripts/calibrate_filters.py --show zh_web    # 另外打印 zh_web 里被丢文档的开头

文档取自 P1-2 抽样时留作评测的文件（见 data/tokenizer_sample/manifest.json），和分词器训练语料不重叠。
存到 data/filter_calib/<来源>.jsonl，每行 {"text": ...}，保留真实的文档边界。

**怎么读结果**：这些来源大多已经被原作者过滤过（fineweb-edu 用的就是 Gopher 规则），
所以正常情况下丢弃率应该很低。某条规则丢得特别多，先看被丢的样本是不是真的差——
多半是规则或阈值不适合这个来源，而不是数据真有那么差。
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import random
import time
from pathlib import Path

from mytransformer.data import filters

OUT = Path("data/filter_calib")
MANIFEST = Path("data/tokenizer_sample/manifest.json")
N_DOCS = 2000
# 和 scripts/sample_tokenizer_corpus.py 的 CODE_LANGS 一致
CODE_LANGS = {"Python", "JavaScript", "TypeScript", "Java", "C", "C++", "GO", "Rust", "Markdown",
              "Shell", "SQL", "HTML", "CSS"}


def fetch() -> None:
    import pyarrow.parquet as pq
    import zstandard
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    manifest = json.loads(MANIFEST.read_text("utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for sub, v in manifest.items():
        if sub == "ref":
            continue
        docs: list[str] = []
        if sub == "en_web_dclm":
            # 整个文件已经在 data/raw/dclm 里；跳过前 6 万行（抽样时用于训练和评测的部分）
            with open(Path("data/raw/dclm") / v["file"], "rb") as fh:
                lines = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
                for i, line in enumerate(lines):
                    if i >= 60_000:
                        docs.append(json.loads(line)["text"])
                    if len(docs) >= N_DOCS:
                        break
        else:
            path, group = v["eval"]["row_groups"][0]
            f = pq.ParquetFile(fs.open(f"datasets/{v['repo']}/{path}"))
            col = "code" if sub == "code" else "text"
            cols = [col, "language"] if sub == "code" else [col]
            while len(docs) < N_DOCS and group < f.metadata.num_row_groups:
                for row in f.read_row_group(group, columns=cols).to_pylist():
                    if sub != "code" or row["language"] in CODE_LANGS:
                        docs.append(row[col])
                group += 1
        with open(OUT / f"{sub}.jsonl", "w", encoding="utf-8") as fo:
            for d in docs[:N_DOCS]:
                fo.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
        print(f"[{sub}] {min(len(docs), N_DOCS)} 篇", flush=True)


def measure(show: list[str]) -> None:
    for path in sorted(OUT.glob("*.jsonl")):
        sub = path.stem
        docs = [json.loads(line)["text"] for line in path.open(encoding="utf-8")]
        t0 = time.time()
        reasons = collections.Counter()
        dropped = collections.defaultdict(list)
        for d in docs:
            r = filters.apply(sub, d)
            reasons[r] += 1
            if r:
                dropped[r].append(d)
        speed = sum(len(d.encode()) for d in docs) / 1e6 / (time.time() - t0)
        kept = reasons.pop(None, 0)
        print(f"{sub:12s} 保留 {kept / len(docs):6.1%}  单核 {speed:5.1f} MB/s  " +
              "  ".join(f"{k} {v / len(docs):.1%}" for k, v in reasons.most_common(6)))
        if sub in show:
            for r, ds in dropped.items():
                for d in random.Random(0).sample(ds, min(2, len(ds))):
                    print(f"    [{r}] {d[:200]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="先从 HF 取文档（约 1 分钟，下载几十 MB）")
    ap.add_argument("--show", nargs="*", default=[], help="打印这些来源里被丢文档的开头")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    measure(args.show)


if __name__ == "__main__":
    main()
