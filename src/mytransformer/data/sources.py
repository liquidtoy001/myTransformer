"""数据来源：每个子集从哪里读、怎么读（P2-5）。

和 P1 的 `scripts/sample_tokenizer_corpus.py` 读的是同一批数据集，但那边只抽 1 GB 训分词器，
这边要按配比取几 GB，所以把"读"这件事单独写在这里：

- `plan_tasks`：先探几个分散的行组（并套用这个子集的过滤规则），估出"每个行组能提供多少可用文本"，
  再决定从哪几个文件、哪些行组读。**任务切分是确定的**：文件顺序固定，每个任务固定取满 8 个行组，
  所以改了目标量也只是在末尾追加任务，已经做完的分片可以直接跳过
- `read_task`：真正读一个任务的文本

为什么要跨文件取：同一个文件里的文档常常按来源或时间扎堆，只读一个文件会偏。
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from dataclasses import dataclass, field

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

# 代码子集保留的语言（注意这个数据集里 Go 的标签是 "GO"）
CODE_LANGS = {"Python", "JavaScript", "TypeScript", "Java", "C", "C++", "GO", "Rust", "Markdown",
              "Shell", "SQL", "HTML", "CSS"}
CODE_MAX_FILE_BYTES = 100_000


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    kind: str                       # parquet | jsonl_zst
    subdir: str = ""
    text_col: str = "text"
    n_files: int = 4                # 从几个文件里取（跨文件取，避免只读到一个文件里扎堆的内容）
    extra_cols: tuple[str, ...] = ()
    files: tuple[str, ...] = ()     # jsonl_zst 直接列文件


SOURCES: dict[str, Source] = {
    "en_web": Source("en_web", "HuggingFaceFW/fineweb-edu", "parquet", "sample/10BT"),
    "en_web_dclm": Source(
        "en_web_dclm", "mlfoundations/dclm-baseline-1.0", "jsonl_zst",
        files=tuple(f"global-shard_01_of_10/local-shard_{i}_of_10/shard_0000000{i}_processed.jsonl.zst"
                    for i in range(4)),
    ),
    "zh_web": Source("zh_web", "opencsg/chinese-fineweb-edu-v2", "parquet", "data"),
    "code": Source("code", "codeparrot/github-code-clean", "parquet", "data",
                   text_col="code", extra_cols=("language", "size")),
    "math": Source("math", "HuggingFaceTB/finemath", "parquet", "finemath-4plus"),
    "en_wiki": Source("en_wiki", "wikimedia/wikipedia", "parquet", "20231101.en"),
    "zh_wiki": Source("zh_wiki", "wikimedia/wikipedia", "parquet", "20231101.zh"),
}


@dataclass
class Task:
    """一个可以独立完成、可以跳过重做的读取单位。"""
    subset: str
    part: int
    path: str                      # 仓库内的文件路径
    row_groups: tuple[int, ...] = field(default=())   # parquet 用
    max_bytes: int = 0             # jsonl_zst 用：读够这么多字节就停


def _fs() -> HfFileSystem:
    return HfFileSystem()


def _parquet_files(src: Source) -> list[str]:
    fs = _fs()
    files = sorted(f for f in fs.ls(f"datasets/{src.repo}/{src.subdir}", detail=False) if f.endswith(".parquet"))
    n = min(src.n_files, len(files))
    return [files[i * len(files) // n] for i in range(n)]  # 均匀取 n 个文件


def _probe_bytes_per_group(f: pq.ParquetFile, src: Source) -> int:
    """估计一个行组能提供多少**能用的**文本字节。

    两个容易低估工作量的坑：
    1. 只探第一个行组。维基按条目 id 排序，开头几千条和中间的长度差很多（实测偏差 3 倍）
    2. 不套过滤规则。代码只保留 13 种语言、单文件 ≤100 KB，实际可用只有原始的约 40%

    所以探 3 个分散的行组，并且用这个子集真正的规则过一遍。
    """
    from . import filters  # 延迟导入，避免模块之间循环依赖

    n = f.metadata.num_row_groups
    total = 0
    probes = sorted({0, n // 2, n - 1})
    for g in probes:
        cols = [src.text_col, *src.extra_cols]
        for row in f.read_row_group(g, columns=cols).to_pylist():
            text = row[src.text_col]
            if not text:
                continue
            if src.name == "code" and (row["language"] not in CODE_LANGS or row["size"] > CODE_MAX_FILE_BYTES):
                continue
            if filters.apply(src.name, text):
                continue
            total += len(text.encode("utf-8"))
    return max(1, total // len(probes))


def plan_tasks(subset: str, target_bytes: int, groups_per_task: int = 8) -> list[Task]:
    """规划任务：凑够 target_bytes 的原始文本需要读哪些行组。"""
    src = SOURCES[subset]
    if src.kind == "jsonl_zst":
        per_file = -(-target_bytes // len(src.files))
        return [Task(subset, i, f, max_bytes=per_file) for i, f in enumerate(src.files)]

    fs = _fs()
    files = _parquet_files(src)
    handles = [pq.ParquetFile(fs.open(f)) for f in files]
    per_group = _probe_bytes_per_group(handles[0], src)
    need = -(-target_bytes // per_group)  # 需要多少个行组

    tasks, part = [], 0
    cursor = [0] * len(files)
    planned = 0
    while planned < need:
        for k, h in enumerate(handles):
            if planned >= need:
                break
            avail = h.metadata.num_row_groups - cursor[k]
            if avail <= 0:
                continue
            # 每个任务固定取满 groups_per_task 个行组（宁可多取一点）：这样任务列表只和
            # 文件顺序有关，改了目标量也只是在末尾追加任务，已经下好的分片编号和内容都不变
            take = min(groups_per_task, avail)
            groups = tuple(range(cursor[k], cursor[k] + take))
            cursor[k] += take
            planned += take
            tasks.append(Task(subset, part, files[k], row_groups=groups))
            part += 1
        if all(c >= h.metadata.num_row_groups for c, h in zip(cursor, handles)):
            break  # 这些文件读完了也不够，就这样
    return tasks


def read_task(task: Task) -> Iterator[str]:
    """读一个任务的文本。代码子集在这里就按语言和大小过滤（和 P1 抽样一致）。"""
    src = SOURCES[task.subset]
    if src.kind == "jsonl_zst":
        import zstandard

        local = hf_hub_download(src.repo, task.path, repo_type="dataset", local_dir="data/raw/dclm")
        kept = 0
        with open(local, "rb") as fh:
            lines = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
            for line in lines:
                text = json.loads(line).get("text", "")
                if not text:
                    continue
                yield text
                kept += len(text.encode("utf-8"))
                if kept >= task.max_bytes:
                    return
        return

    f = pq.ParquetFile(_fs().open(task.path))
    cols = [src.text_col, *src.extra_cols]
    for g in task.row_groups:
        for row in f.read_row_group(g, columns=cols).to_pylist():
            text = row[src.text_col]
            if not text:
                continue
            if task.subset == "code" and (row["language"] not in CODE_LANGS or row["size"] > CODE_MAX_FILE_BYTES):
                continue
            yield text
