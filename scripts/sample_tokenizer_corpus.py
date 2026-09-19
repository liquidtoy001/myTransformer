"""从 HuggingFace 上抽取分词器训练语料和压缩率评测文本（P1-2）。

    python scripts/sample_tokenizer_corpus.py                 # 全部来源
    python scripts/sample_tokenizer_corpus.py --only zh_web   # 只重抽某一个

输出（都在 data/ 下，不进仓库）：

    data/tokenizer_sample/train/<来源>.txt    训练语料，约 1.0 GB
    data/tokenizer_sample/eval/<来源>.txt     评测文本，每个来源约 2 MB，和训练语料不重叠
    data/tokenizer_sample/manifest.json      每个来源用了哪些文件、哪些行组、多少字节、sha256
    data/tokenizers_ref/<名字>.json           对比用的别家分词器

**不下载整个文件**：parquet 按"行组"存储（这些数据集每组约 1000 行），可以只通过 HTTP
范围请求读需要的那几组的 text 列。fineweb-edu 一个文件 2.15 GB，这里只读其中几百 MB 的文本。

**可复现**：行组的选取顺序由固定种子决定，同样的参数永远抽出同样的文本；manifest 里的 sha256 可以核对。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

OUT = Path("data/tokenizer_sample")
REF_DIR = Path("data/tokenizers_ref")
MB = 1_000_000
EVAL_BYTES = 2 * MB
SEED = 0

# 注意这个数据集里 Go 的标签是 "GO"
CODE_LANGS = {"Python", "JavaScript", "TypeScript", "Java", "C", "C++", "GO", "Rust", "Markdown",
              "Shell", "SQL", "HTML", "CSS"}
CODE_MAX_FILE_BYTES = 100_000      # 更大的多半是压缩过的 JS、自动生成的代码或数据文件
CODE_MAX_SHARE_PER_LANG = 0.25     # 单一语言最多占代码部分的 25%，避免被 Java/JS 淹没


@dataclass
class Source:
    name: str
    repo: str
    subdir: str                 # 仓库里放 parquet 的目录
    train_bytes: int
    text_col: str = "text"
    n_train_files: int = 3      # 训练语料分散在几个文件里（同一文件的内容可能按来源扎堆）
    extra_cols: list[str] = field(default_factory=list)


PARQUET_SOURCES = [
    Source("en_web", "HuggingFaceFW/fineweb-edu", "sample/10BT", 350 * MB),
    Source("zh_web", "opencsg/chinese-fineweb-edu-v2", "data", 180 * MB),
    Source("zh_wiki", "wikimedia/wikipedia", "20231101.zh", 70 * MB),
    Source("en_wiki", "wikimedia/wikipedia", "20231101.en", 70 * MB),
    Source("code", "codeparrot/github-code-clean", "data", 150 * MB, text_col="code",
           extra_cols=["language", "size"]),
    Source("math", "HuggingFaceTB/finemath", "finemath-4plus", 80 * MB),
]
DCLM = ("mlfoundations/dclm-baseline-1.0",
        "global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst", 100 * MB)

REF_TOKENIZERS = {
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "llama3": "NousResearch/Meta-Llama-3-8B",
    "gpt-4o": "Xenova/gpt-4o",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
    "gpt2": "openai-community/gpt2",
}


class Writer:
    """把文档追加到文本文件，数字节、算 sha256。文档之间空一行。"""

    def __init__(self, path: Path, limit: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.limit = path, limit
        self.f = open(path, "w", encoding="utf-8", newline="")
        self.h = hashlib.sha256()
        self.bytes = self.docs = 0
        self.stats: dict = {}

    @property
    def full(self) -> bool:
        return self.bytes >= self.limit

    def add(self, text: str) -> None:
        text = text.rstrip("\n") + "\n\n"
        b = text.encode("utf-8")
        self.f.write(text)
        self.h.update(b)
        self.bytes += len(b)
        self.docs += 1

    def close(self) -> dict:
        self.f.close()
        return {"path": str(self.path).replace("\\", "/"), "bytes": self.bytes, "docs": self.docs,
                "sha256": self.h.hexdigest(), **self.stats}


def pick_files(fs: HfFileSystem, src: Source) -> tuple[list[str], str]:
    """训练用 n 个均匀分布的文件，评测用最后一个文件（训练永远不碰它）。"""
    files = sorted(f for f in fs.ls(f"datasets/{src.repo}/{src.subdir}", detail=False)
                   if f.endswith(".parquet"))
    pool, eval_file = files[:-1], files[-1]
    n = min(src.n_train_files, len(pool))
    return [pool[i * len(pool) // n] for i in range(n)], eval_file


def keep_code(row: dict, lang_bytes: dict[str, int], budget: int) -> bool:
    lang = row["language"]
    if lang not in CODE_LANGS or row["size"] > CODE_MAX_FILE_BYTES:
        return False
    if lang_bytes.get(lang, 0) >= budget * CODE_MAX_SHARE_PER_LANG:
        return False
    lang_bytes[lang] = lang_bytes.get(lang, 0) + row["size"]
    return True


def fill_from_parquet(fs: HfFileSystem, src: Source, files: list[str], w: Writer, seed: int) -> list:
    """在几个文件之间轮流、每个文件内按打乱的顺序读行组，直到写满。返回用过的 (文件, 行组) 列表。"""
    rng = random.Random(seed)
    handles = [pq.ParquetFile(fs.open(f)) for f in files]
    orders = []
    for h in handles:
        order = list(range(h.metadata.num_row_groups))
        rng.shuffle(order)
        orders.append(order)
    cols = [src.text_col, *src.extra_cols]
    lang_bytes: dict[str, int] = {}
    used = []
    step = 0
    while not w.full and any(orders):
        k = step % len(files)
        step += 1
        if not orders[k]:
            continue
        g = orders[k].pop(0)
        used.append([files[k].split("/", 3)[-1], g])
        for row in handles[k].read_row_group(g, columns=cols).to_pylist():
            text = row[src.text_col]
            if not text or not text.strip():
                continue
            if src.name == "code" and not keep_code(row, lang_bytes, w.limit):
                continue
            w.add(text)
            if w.full:
                break
    if lang_bytes:
        w.stats = {"lang_bytes": dict(sorted(lang_bytes.items(), key=lambda kv: -kv[1]))}
    return used


def sample_parquet(src: Source) -> dict:
    fs = HfFileSystem()
    train_files, eval_file = pick_files(fs, src)
    t0 = time.time()
    w = Writer(OUT / "train" / f"{src.name}.txt", src.train_bytes)
    used = fill_from_parquet(fs, src, train_files, w, SEED)
    train = w.close() | {"row_groups": used}
    w = Writer(OUT / "eval" / f"{src.name}.txt", EVAL_BYTES)
    used = fill_from_parquet(fs, src, [eval_file], w, SEED + 1)
    ev = w.close() | {"row_groups": used}
    print(f"[{src.name}] {train['bytes'] / MB:.0f} MB 训练 + {ev['bytes'] / MB:.1f} MB 评测，"
          f"{time.time() - t0:.0f} 秒", flush=True)
    return {"repo": src.repo, "train": train, "eval": ev}


def sample_dclm() -> dict:
    """DCLM 是 .jsonl.zst，不能只读一部分：整个下载一个文件，前 100 MB 训练，接下来 2 MB 评测。"""
    import zstandard

    repo, path, limit = DCLM
    t0 = time.time()
    local = hf_hub_download(repo, path, repo_type="dataset", local_dir="data/raw/dclm")
    train = Writer(OUT / "train" / "en_web_dclm.txt", limit)
    ev = Writer(OUT / "eval" / "en_web_dclm.txt", EVAL_BYTES)
    with open(local, "rb") as fh:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for line in reader:
            w = train if not train.full else ev
            if ev.full:
                break
            text = json.loads(line).get("text", "")
            if text.strip():
                w.add(text)
    out = {"repo": repo, "file": path, "train": train.close(), "eval": ev.close()}
    print(f"[en_web_dclm] {out['train']['bytes'] / MB:.0f} MB 训练 + {out['eval']['bytes'] / MB:.1f} MB 评测，"
          f"{time.time() - t0:.0f} 秒", flush=True)
    return out


def download_ref_tokenizers() -> dict:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, repo in REF_TOKENIZERS.items():
        p = Path(hf_hub_download(repo, "tokenizer.json", local_dir=REF_DIR / "_hf" / name))
        dst = REF_DIR / f"{name}.json"
        dst.write_bytes(p.read_bytes())
        out[name] = {"repo": repo, "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}
    print(f"[ref] {len(out)} 个对比分词器", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="只抽这些来源（名字见 PARQUET_SOURCES，另有 en_web_dclm、ref）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0, help="所有目标字节数乘以它；0.01 用来快速试跑")
    args = ap.parse_args()
    global EVAL_BYTES, DCLM
    EVAL_BYTES = int(EVAL_BYTES * args.scale)
    DCLM = (*DCLM[:2], int(DCLM[2] * args.scale))
    for s in PARQUET_SOURCES:
        s.train_bytes = int(s.train_bytes * args.scale)

    jobs = {s.name: (lambda s=s: sample_parquet(s)) for s in PARQUET_SOURCES}
    jobs["en_web_dclm"] = sample_dclm
    jobs["ref"] = download_ref_tokenizers
    if args.only:
        jobs = {k: v for k, v in jobs.items() if k in args.only}

    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {k: pool.submit(f) for k, f in jobs.items()}
        for k, fut in futures.items():
            manifest[k] = fut.result()
            OUT.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8")
    total = sum(v["train"]["bytes"] for k, v in manifest.items() if k != "ref")
    print(f"完成：训练语料共 {total / MB:.0f} MB，用时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
