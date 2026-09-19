"""分词与打包（P2-4）：文档 → token → 按配比混合的训练分片 + 每个子集单独的验证集。

两步，中间结果落盘：

1. **分词**（`tokenize_docs`）：每个子集写成 docs.bin（所有文档的 token 首尾相接，每篇前面放一个
   <|endoftext|>）+ docs.idx（每篇的起止位置）。有了 idx，可以按编号随机读任意一篇，不用整个读进内存。
2. **打包**（`pack`）：
   - 每个子集先用固定种子打乱文档顺序，**排在最前面的文档划给验证集**，直到约 val_tokens 个 token。
     这些文档从此不会进入训练分片（物理上分开，而不是训练时跳过，见 docs/05-data.md §三）
   - 剩下的文档按配比混合：每次从"实际占比比目标落后最多"的子集取下一篇。
     ShardLoader 是按顺序读的，**混合必须在这里做好**：如果一个分片全是中文、下一个全是代码，
     训练时模型会在几百步里只看到一种数据
   - 某个子集的文档不够它的配额时，打乱后再用一轮（多 epoch），meta.json 里记下用了几轮
   - 分片在文档边界切开，每个约 shard_tokens 个 token

meta.json 记录分词器指纹、每个分片的 token 数、每个子集实际贡献了多少 token。
训练开始时 `check_meta` 核对：换了分词器、分片拷贝不完整，都会立刻报错。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from .shards import DTYPE, write_shard

META_FORMAT = 1


# ---------------------------------------------------------------- 第 1 步：分词


def tokenize_docs(tokenizer, texts: Iterable[str], out_dir: str | Path, batch_size: int = 1000) -> dict:
    """把一个子集的文档分词，写成 docs.bin + docs.idx。每篇文档前面放一个 EOT。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eot = tokenizer.eot_id
    offsets = [0]
    with open(out / "docs.bin", "wb") as f:
        batch: list[str] = []

        def flush() -> None:
            for ids in tokenizer.encode_batch(batch):
                arr = np.asarray([eot, *ids], dtype=DTYPE)
                f.write(arr.tobytes())
                offsets.append(offsets[-1] + arr.size)
            batch.clear()

        for t in texts:
            batch.append(t)
            if len(batch) == batch_size:
                flush()
        if batch:
            flush()
    np.asarray(offsets, dtype=np.int64).tofile(out / "docs.idx")
    info = {"docs": len(offsets) - 1, "tokens": offsets[-1], "tokenizer": tokenizer.fingerprint}
    (out / "docs.json").write_text(json.dumps(info, indent=1), "utf-8")
    return info


class DocReader:
    """按编号读 docs.bin 里的第 i 篇文档（含开头的 EOT）。"""

    def __init__(self, path: str | Path) -> None:
        self.dir = Path(path)
        self.idx = np.fromfile(self.dir / "docs.idx", dtype=np.int64)
        self.data = np.memmap(self.dir / "docs.bin", dtype=DTYPE, mode="r")
        self.info = json.loads((self.dir / "docs.json").read_text("utf-8"))
        if self.idx[-1] != self.data.size:
            raise ValueError(f"{self.dir}：docs.idx 记录 {self.idx[-1]} 个 token，docs.bin 实际 {self.data.size}")

    def __len__(self) -> int:
        return len(self.idx) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return self.data[self.idx[i] : self.idx[i + 1]]

    def n_tokens(self, i: int) -> int:
        return int(self.idx[i + 1] - self.idx[i])


# ---------------------------------------------------------------- 第 2 步：划验证集、混合、写分片


def split_val(reader: DocReader, val_tokens: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """打乱后，前面的文档给验证集直到 ≥ val_tokens，其余给训练。返回 (验证文档编号, 训练文档编号)。"""
    order = np.random.default_rng(seed).permutation(len(reader))
    total, cut = 0, 0
    while cut < len(order) and total < val_tokens:
        total += reader.n_tokens(int(order[cut]))
        cut += 1
    if cut == len(order):
        raise ValueError(f"{reader.dir}：全部文档加起来不够 {val_tokens} 个验证 token")
    return order[:cut], order[cut:]


class Mixer:
    """按配比产出 (子集, 文档编号)。

    每次选"已产出 token 占比比目标落后最多"的子集（按比例的轮询），所以任意一段连续的输出里，
    各子集的比例都接近目标。子集内部按打乱的顺序取；取完一轮，用新的种子再打乱一轮。
    """

    def __init__(self, pools: dict[str, tuple[DocReader, np.ndarray]], weights: dict[str, float], seed: int) -> None:
        self.pools = pools
        self.names = sorted(weights)
        w = np.array([weights[n] for n in self.names], dtype=np.float64)
        self.target = w / w.sum()
        self.seed = seed
        self.emitted = np.zeros(len(self.names))
        self.order = {n: pools[n][1] for n in self.names}
        self.cursor = {n: 0 for n in self.names}
        self.rounds = {n: 0 for n in self.names}  # 已经完整用完了几轮

    def run(self, total_tokens: int) -> Iterator[tuple[str, int]]:
        produced = 0
        while produced < total_tokens:
            k = int(np.argmax(self.target * max(produced, 1) - self.emitted))
            name = self.names[k]
            reader, ids = self.pools[name]
            if self.cursor[name] == len(self.order[name]):  # 这个子集用完一轮，重新打乱
                self.rounds[name] += 1
                rng = np.random.default_rng([self.seed, k, self.rounds[name]])
                self.order[name] = ids[rng.permutation(len(ids))]
                self.cursor[name] = 0
            doc = int(self.order[name][self.cursor[name]])
            self.cursor[name] += 1
            n = reader.n_tokens(doc)
            self.emitted[k] += n
            produced += n
            yield name, doc

    def epochs(self, name: str) -> float:
        """这个子集的训练文档被用了几轮（1.5 = 全部用过一遍，又用了一半）。"""
        return self.rounds[name] + self.cursor[name] / max(1, len(self.pools[name][1]))


def pack(subset_dirs: dict[str, str | Path], weights: dict[str, float], out_dir: str | Path, *,
         total_tokens: int, val_tokens: int = 1_000_000, shard_tokens: int = 100_000_000,
         seed: int = 0) -> dict:
    """完整的第 2 步。返回写进 meta.json 的内容。"""
    if set(subset_dirs) != set(weights):
        raise ValueError(f"子集和配比对不上：{sorted(subset_dirs)} vs {sorted(weights)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    readers = {n: DocReader(p) for n, p in subset_dirs.items()}
    fps = {r.info["tokenizer"] for r in readers.values()}
    if len(fps) != 1:
        raise ValueError(f"各子集用的分词器不一样：{fps}")

    pools, val_meta = {}, {}
    for i, (name, r) in enumerate(sorted(readers.items())):
        val_ids, train_ids = split_val(r, val_tokens, seed + i)
        val = np.concatenate([r[int(d)] for d in np.sort(val_ids)])
        write_shard(out / f"val_{name}.bin", val)
        val_meta[name] = {"file": f"val_{name}.bin", "tokens": int(val.size), "docs": len(val_ids)}
        pools[name] = (r, train_ids)

    shards, buf, buf_n = [], [], 0
    by_subset = {n: {"tokens": 0, "docs": 0} for n in weights}

    def close_shard() -> None:
        nonlocal buf, buf_n
        name = f"train_{len(shards):05d}.bin"
        write_shard(out / name, np.concatenate(buf))
        shards.append({"file": name, "tokens": buf_n})
        buf, buf_n = [], 0

    mixer = Mixer(pools, weights, seed)
    for name, doc in mixer.run(total_tokens):
        arr = pools[name][0][doc]
        buf.append(arr)
        buf_n += arr.size
        by_subset[name]["tokens"] += int(arr.size)
        by_subset[name]["docs"] += 1
        if buf_n >= shard_tokens:
            close_shard()
    if buf:
        close_shard()

    total = sum(s["tokens"] for s in shards)
    for name, v in by_subset.items():
        v["share"] = v["tokens"] / total
        v["epochs"] = round(mixer.epochs(name), 4)
        v["target_share"] = weights[name] / sum(weights.values())
    meta = {
        "format": META_FORMAT,
        "tokenizer": fps.pop(),
        "seed": seed,
        "train": {"tokens": total, "shards": shards, "by_subset": by_subset},
        "val": val_meta,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    return meta


# ---------------------------------------------------------------- 训练前核对


def check_meta(paths: list[str | Path], tokenizer_fingerprint: str | None) -> None:
    """训练开始前核对分片：同目录下有 meta.json 的话，分词器指纹要一致、每个分片的 token 数要对得上。

    分片大小对不上，最常见的原因是从对象存储拷贝时中断了——不查的话，训练会在缺了一截的数据上
    照常跑下去，不会报错。
    """
    by_dir: dict[Path, list[Path]] = {}
    for p in map(Path, paths):
        by_dir.setdefault(p.parent, []).append(p)
    for d, files in by_dir.items():
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue  # 合成数据等没有 meta 的情况
        meta = json.loads(meta_path.read_text("utf-8"))
        if tokenizer_fingerprint is not None and meta["tokenizer"] != tokenizer_fingerprint:
            raise ValueError(f"{d}：分片是用分词器 {meta['tokenizer']} 切的，当前配置的是 {tokenizer_fingerprint}")
        expected = {s["file"]: s["tokens"] for s in meta["train"]["shards"]}
        expected |= {v["file"]: v["tokens"] for v in meta["val"].values()}
        for f in files:
            want = expected.get(f.name)
            got = f.stat().st_size // np.dtype(DTYPE).itemsize
            if want is not None and got != want:
                raise ValueError(f"{f}：meta.json 记录 {want} 个 token，文件里只有 {got} 个（拷贝不完整？）")
