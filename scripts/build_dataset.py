"""完整数据管线（P2-5）：下载 → 过滤 → 去重 → 去污染 → 段落去重 → 分词 → 打包。

    python scripts/build_dataset.py --config configs/data/pipeline_2b.yaml            # 全部阶段
    python scripts/build_dataset.py --config ... --stages fetch                        # 只跑某几个阶段
    python scripts/build_dataset.py --config ... --scale 0.01                          # 先用 1% 试跑

**可以中断重跑**：每个阶段按"分片"落盘，已经做完的分片（有对应的 .json 统计文件）直接跳过。
下载几 GB 的过程中断网、关机都不要紧，重跑只补没做完的。

**为什么分这几个阶段**：
- `fetch`（可并行）：读原始数据 + 质量过滤 + 算 MinHash 签名 + 查评测污染。
  这三件事都只看单篇文档，放在一起做，一篇文档只解析一次
- `clean`（**必须串行**）：近似去重要把所有签名放在一起看；段落去重要"只保留第一次出现"，
  顺序一变结果就变。所以这一步在主进程里按固定顺序走一遍
- `tokenize`、`pack`：见 docs/learn/p2-4-pack.md

**段落统计的近似**：段落出现在几篇文档里，是在 fetch 阶段对**过滤后、去重前**的文档统计的
（这样才能并行）。去重会再删掉一些文档，所以计数略偏高，对"≥3 篇才算常见段落"的判断影响很小。
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
import zstandard

from mytransformer.data import filters
from mytransformer.data.dedup import (
    PARAGRAPH_DEDUP_SKIP,
    MinHasher,
    near_duplicates,
    paragraph_keys,
    strip_repeated,
)
from mytransformer.data.decontam import EvalIndex
from mytransformer.data.pack import pack, tokenize_docs
from mytransformer.data.sources import Task, plan_tasks, read_task
from mytransformer.tokenizer import Tokenizer

MIN_CHARS_AFTER_PARAGRAPH_DEDUP = 200  # 段落删完只剩这么点，就整篇丢掉
STAGES = ("fetch", "clean", "tokenize", "pack", "report")
HAND_MARKER = "<!-- 以下为手写，重新生成报告时保留 -->"  # 报告里这一行之后的内容不会被覆盖


# ---------------------------------------------------------------- 小工具


def write_jsonl_zst(path: Path, docs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f, zstandard.ZstdCompressor(level=3).stream_writer(f) as z:
        w = io.TextIOWrapper(z, encoding="utf-8")
        for d in docs:
            w.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
        w.flush()


def read_jsonl_zst(path: Path):
    with open(path, "rb") as f:
        r = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8")
        for line in r:
            yield json.loads(line)["text"]


def load_cfg(path: str, scale: float) -> dict:
    cfg = yaml.safe_load(Path(path).read_text("utf-8"))
    cfg["total_tokens"] = int(cfg["total_tokens"] * scale)
    cfg["val_tokens"] = max(50_000, int(cfg["val_tokens"] * scale))
    cfg["shard_tokens"] = max(1_000_000, int(cfg["shard_tokens"] * scale))
    total_w = sum(cfg["weights"].values())
    cfg["target_tokens"] = {k: cfg["total_tokens"] * v / total_w for k, v in cfg["weights"].items()}
    yield_factor = cfg.get("yield_factor", {})
    cfg["target_bytes"] = {
        k: int(v * cfg["bytes_per_token"][k] * cfg["loss_factor"] / yield_factor.get(k, 1.0))
        for k, v in cfg["target_tokens"].items()
    }
    cfg["dir"] = Path(cfg["out"])
    return cfg


# ---------------------------------------------------------------- 阶段 1：fetch（并行）

_state: dict = {}


def _init(cfg: dict) -> None:
    _state["cfg"] = cfg
    _state["hasher"] = MinHasher()
    _state["index"] = EvalIndex.from_dir(cfg["eval_sets"], exclude_file=cfg["decontam_exclude"])


def _fetch_one(task: Task) -> dict:
    """读一个任务：质量过滤 → 签名 → 查污染 → 统计段落。输出 raw/part_*.zst + 三个 npy。"""
    cfg = _state["cfg"]
    out = cfg["dir"] / task.subset
    stats_path = out / "raw" / f"part_{task.part:05d}.json"
    if stats_path.exists():
        return json.loads(stats_path.read_text("utf-8"))

    t0 = time.time()
    by_char = task.subset.startswith("zh")
    docs, sigs, contam, paras = [], [], [], []
    dropped: collections.Counter = collections.Counter()
    n_in = 0
    for text in read_task(task):
        n_in += 1
        reason = filters.apply(task.subset, text)
        if reason:
            dropped[reason] += 1
            continue
        docs.append(text)
        sigs.append(_state["hasher"].signature(text, by_char=by_char))
        contam.append(_state["index"].contaminated(text) or "")
        if task.subset not in PARAGRAPH_DEDUP_SKIP:
            paras.extend(paragraph_keys(text))

    write_jsonl_zst(out / "raw" / f"part_{task.part:05d}.jsonl.zst", docs)
    sig_arr = np.stack(sigs).astype(np.uint32) if sigs else np.empty((0, 256), dtype=np.uint32)
    sig_arr.tofile(out / "raw" / f"part_{task.part:05d}.sig")
    (out / "raw" / f"part_{task.part:05d}.contam.json").write_text(json.dumps(contam), "utf-8")
    np.asarray(paras, dtype=np.uint32).tofile(out / "raw" / f"part_{task.part:05d}.para")
    stats = {
        "subset": task.subset, "part": task.part, "docs_in": n_in, "docs_kept": len(docs),
        "bytes_kept": sum(len(d.encode("utf-8")) for d in docs),
        "dropped": dict(dropped), "seconds": round(time.time() - t0, 1),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False), "utf-8")
    return stats


def stage_fetch(cfg: dict, workers: int) -> None:
    tasks: list[Task] = []
    for subset, target in cfg["target_bytes"].items():
        plan = plan_tasks(subset, target)
        print(f"[plan] {subset}: {len(plan)} 个分片，目标 {target / 1e9:.2f} GB 文本", flush=True)
        tasks.extend(plan)
    done = sum((cfg["dir"] / t.subset / "raw" / f"part_{t.part:05d}.json").exists() for t in tasks)
    print(f"[fetch] 共 {len(tasks)} 个分片，已完成 {done}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(cfg,)) as ex:
        for i, s in enumerate(ex.map(_fetch_one, tasks), 1):
            print(f"[fetch {i}/{len(tasks)}] {s['subset']} part {s['part']}："
                  f"{s['docs_kept']}/{s['docs_in']} 篇，{s['bytes_kept'] / 1e6:.0f} MB，{s['seconds']}s", flush=True)
    print(f"[fetch] 用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)


# ---------------------------------------------------------------- 阶段 2：clean（串行）


def _parts(cfg: dict, subset: str) -> list[int]:
    d = cfg["dir"] / subset / "raw"
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("part_*.json") if "." not in p.stem.split("_")[1])


def stage_clean(cfg: dict) -> dict:
    order = [(s, p) for s in cfg["priority"] for p in _parts(cfg, s)]  # 优先级：靠前的子集优先保留
    sigs, contam, sizes = [], [], []
    para_counts: list[np.ndarray] = []
    for subset, part in order:
        base = cfg["dir"] / subset / "raw" / f"part_{part:05d}"
        sig = np.fromfile(f"{base}.sig", dtype=np.uint32).reshape(-1, 256)
        sigs.append(sig)
        sizes.append(len(sig))
        contam.extend(json.loads(Path(f"{base}.contam.json").read_text("utf-8")))
        para_counts.append(np.fromfile(f"{base}.para", dtype=np.uint32))

    all_sigs = np.concatenate(sigs)
    print(f"[clean] 共 {len(all_sigs):,} 篇，签名 {all_sigs.nbytes / 1e9:.2f} GB，开始近似去重", flush=True)
    t0 = time.time()
    rep = near_duplicates(all_sigs)
    is_dup = rep != np.arange(len(rep))
    print(f"[clean] 近似重复 {int(is_dup.sum()):,} 篇（{is_dup.mean():.1%}），{time.time() - t0:.0f}s", flush=True)

    keys, counts = np.unique(np.concatenate(para_counts), return_counts=True)
    common = set(keys[counts >= 3].tolist())
    print(f"[clean] 段落 {len(keys):,} 种，出现 ≥3 篇的 {len(common):,} 种", flush=True)
    del para_counts, keys, counts

    seen: set[int] = set()
    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    cursor = 0
    for (subset, part), n_docs in zip(order, sizes):
        base = cfg["dir"] / subset / "raw" / f"part_{part:05d}"
        kept: list[str] = []
        for i, text in enumerate(read_jsonl_zst(Path(f"{base}.jsonl.zst"))):
            g = cursor + i
            st = stats[subset]
            st["in"] += 1
            if is_dup[g]:
                st["近似重复"] += 1
                continue
            if contam[g]:
                st[f"评测污染:{contam[g]}"] += 1
                continue
            if subset not in PARAGRAPH_DEDUP_SKIP:
                text, removed = strip_repeated(text, common, seen)
                st["删掉的段落"] += removed
                if len(text.strip()) < MIN_CHARS_AFTER_PARAGRAPH_DEDUP:
                    st["段落删完太短"] += 1
                    continue
            kept.append(text)
            st["kept"] += 1
            st["bytes"] += len(text.encode("utf-8"))
        cursor += n_docs
        write_jsonl_zst(cfg["dir"] / subset / "clean" / f"part_{part:05d}.jsonl.zst", kept)
    out = {s: dict(c) for s, c in stats.items()}
    (cfg["dir"] / "clean_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    for s, c in out.items():
        print(f"[clean] {s:12s} 保留 {c['kept']}/{c['in']}（{c['kept'] / max(1, c['in']):.1%}）", flush=True)
    return out


# ---------------------------------------------------------------- 阶段 3、4、5


def stage_tokenize(cfg: dict) -> None:
    tok = Tokenizer.from_file(cfg["tokenizer"])
    for subset in cfg["weights"]:
        out = cfg["dir"] / "docs" / subset
        if (out / "docs.json").exists():
            print(f"[tokenize] {subset} 已完成，跳过", flush=True)
            continue
        t0 = time.time()
        files = sorted((cfg["dir"] / subset / "clean").glob("part_*.jsonl.zst"))
        texts = (t for f in files for t in read_jsonl_zst(f))
        info = tokenize_docs(tok, texts, out)
        print(f"[tokenize] {subset}: {info['docs']:,} 篇 → {info['tokens'] / 1e6:.1f}M tokens，"
              f"{time.time() - t0:.0f}s", flush=True)


def stage_pack(cfg: dict) -> dict:
    dirs = {s: cfg["dir"] / "docs" / s for s in cfg["weights"]}
    meta = pack(dirs, cfg["weights"], cfg["dir"] / "mix", total_tokens=cfg["total_tokens"],
                val_tokens=cfg["val_tokens"], shard_tokens=cfg["shard_tokens"], seed=cfg["seed"])
    print(f"[pack] 训练 {meta['train']['tokens'] / 1e9:.3f}B tokens，{len(meta['train']['shards'])} 个分片", flush=True)
    for name, v in meta["train"]["by_subset"].items():
        print(f"  {name:12s} 占比 {v['share']:.3f}（目标 {v['target_share']:.3f}）  用了 {v['epochs']:.2f} 轮", flush=True)
    return meta


def stage_report(cfg: dict) -> None:
    """写 reports/data.md（漏斗表）和 data/review_sample.txt（随机 200 篇，人工过目用）。"""
    raw = collections.defaultdict(collections.Counter)
    for subset in cfg["weights"]:
        for f in (cfg["dir"] / subset / "raw").glob("part_*.json"):
            if f.stem.count(".") or not f.stem.split("_")[1].isdigit():
                continue
            s = json.loads(f.read_text("utf-8"))
            raw[subset]["docs_in"] += s["docs_in"]
            raw[subset]["docs_kept"] += s["docs_kept"]
            raw[subset]["bytes_kept"] += s["bytes_kept"]
            for k, v in s["dropped"].items():
                raw[subset][f"drop:{k}"] += v
    clean = json.loads((cfg["dir"] / "clean_stats.json").read_text("utf-8"))
    meta = json.loads((cfg["dir"] / "mix" / "meta.json").read_text("utf-8"))

    lines = [f"# 数据管线报告：{cfg['name']}", "",
             f"配置：`{cfg['config_path']}`　分词器指纹：`{meta['tokenizer']}`", "",
             "## 一、过滤漏斗（篇数）", "",
             "| 子集 | 读入 | 质量过滤后 | 去重去污染后 | 最终 tokens | 占比（目标） | 用了几轮 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for subset in cfg["weights"]:
        r, c = raw[subset], clean.get(subset, {})
        m = meta["train"]["by_subset"][subset]
        lines.append(f"| {subset} | {r['docs_in']:,} | {r['docs_kept']:,}（{r['docs_kept'] / max(1, r['docs_in']):.1%}）"
                     f" | {c.get('kept', 0):,}（{c.get('kept', 0) / max(1, r['docs_kept']):.1%}） | "
                     f"{m['tokens'] / 1e6:.1f}M | {m['share']:.3f}（{m['target_share']:.3f}） | {m['epochs']:.2f} |")
    lines += ["", f"**合计**：训练 {meta['train']['tokens'] / 1e9:.3f}B tokens，"
                  f"{len(meta['train']['shards'])} 个分片；验证集每个子集 "
                  f"{list(meta['val'].values())[0]['tokens'] / 1000:.0f}k tokens 左右", ""]

    lines += ["## 二、每条规则丢了多少", "", "| 子集 | 丢弃原因 | 篇数 | 占读入 |", "|---|---|---:|---:|"]
    for subset in cfg["weights"]:
        r = raw[subset]
        for k, v in sorted(((k[5:], v) for k, v in r.items() if k.startswith("drop:")), key=lambda x: -x[1])[:6]:
            lines.append(f"| {subset} | {k} | {v:,} | {v / max(1, r['docs_in']):.2%} |")
        for k, v in sorted(clean.get(subset, {}).items(), key=lambda x: -x[1] if isinstance(x[1], int) else 0):
            if k in ("in", "kept", "bytes", "删掉的段落"):
                continue
            lines.append(f"| {subset} | {k} | {v:,} | {v / max(1, r['docs_in']):.2%} |")
    lines += ["", "## 三、人工抽查", "",
              "`data/review_sample.txt` 是从最终数据里随机抽的 200 篇（每篇前 800 字），按 05-data.md 的要求人工过目。", "",
              HAND_MARKER]
    out = Path("reports/data.md")
    old = out.read_text("utf-8") if out.exists() else ""
    if HAND_MARKER in old:
        lines.append(old.split(HAND_MARKER, 1)[1].lstrip("\n"))  # 手写的部分保留，重新生成不覆盖
    out.write_text("\n".join(lines), "utf-8")
    print("[report] 已写 reports/data.md", flush=True)

    rng = random.Random(0)
    sample = []
    for subset in cfg["weights"]:
        files = sorted((cfg["dir"] / subset / "clean").glob("part_*.jsonl.zst"))
        if not files:
            continue
        texts = list(read_jsonl_zst(rng.choice(files)))
        for t in rng.sample(texts, min(200 // len(cfg["weights"]) + 1, len(texts))):
            sample.append(f"===== [{subset}] =====\n{t[:800]}\n")
    rng.shuffle(sample)
    Path("data/review_sample.txt").write_text("\n".join(sample[:200]), "utf-8")
    print("[report] 已写 data/review_sample.txt（200 篇）", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    ap.add_argument("--scale", type=float, default=1.0, help="所有目标量乘以它，先小规模试跑")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.scale)
    cfg["config_path"] = args.config
    if args.scale != 1.0:
        cfg["dir"] = cfg["dir"].with_name(cfg["dir"].name + f"_scale{args.scale:g}")
        cfg["name"] += f"（scale={args.scale:g}）"
    print(f"输出目录 {cfg['dir']}，目标 {cfg['total_tokens'] / 1e9:.3f}B tokens", flush=True)

    t0 = time.time()
    if "fetch" in args.stages:
        stage_fetch(cfg, args.workers)
    if "clean" in args.stages:
        stage_clean(cfg)
    if "tokenize" in args.stages:
        stage_tokenize(cfg)
    if "pack" in args.stages:
        stage_pack(cfg)
    if "report" in args.stages:
        stage_report(cfg)
    print(f"完成，总用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
