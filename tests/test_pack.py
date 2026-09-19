"""分词打包：验证集和训练集不重叠、混合均匀、分片在文档边界切开、meta.json 能拦住错误的数据。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from mytransformer.data.pack import DocReader, Mixer, check_meta, pack, split_val, tokenize_docs
from mytransformer.data.shards import DTYPE
from mytransformer.tokenizer import Tokenizer

MT32K = Path(__file__).resolve().parents[1] / "tokenizer" / "mt32k.json"


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_file(MT32K)


def docs(subset: str, n: int, words: int = 30) -> list[str]:
    """每篇文档带一个独一无二的标记 <subset-i>，解码训练分片后能认出它是哪一篇。"""
    return [f"<{subset}-{i}> " + " ".join(f"{subset} text {i} word {j}." for j in range(words)) for i in range(n)]


def build(tmp: Path, tok: Tokenizer, sizes: dict[str, int], weights: dict[str, float], **kw) -> dict:
    dirs = {}
    for name, n in sizes.items():
        tokenize_docs(tok, docs(name, n), tmp / "docs" / name, batch_size=64)
        dirs[name] = tmp / "docs" / name
    return pack(dirs, weights, tmp / "out", **kw)


def markers(tok: Tokenizer, path: Path) -> list[str]:
    return re.findall(r"<(\w+-\d+)>", tok.decode(np.fromfile(path, dtype=DTYPE).tolist()))


def test_tokenize_docs_roundtrip(tok, tmp_path):
    texts = ["第一篇文档。", "second document", "  三  "]
    info = tokenize_docs(tok, texts, tmp_path)
    r = DocReader(tmp_path)
    assert len(r) == 3 and info["tokens"] == r.data.size and info["tokenizer"] == tok.fingerprint
    for i, t in enumerate(texts):
        ids = r[i].tolist()
        assert ids[0] == tok.eot_id and tok.decode(ids[1:]) == t


def test_split_val_is_disjoint_and_deterministic(tok, tmp_path):
    tokenize_docs(tok, docs("a", 200), tmp_path)
    r = DocReader(tmp_path)
    val, train = split_val(r, 2000, seed=0)
    assert set(val).isdisjoint(train) and len(val) + len(train) == len(r)
    assert sum(r.n_tokens(int(d)) for d in val) >= 2000
    v2, _ = split_val(r, 2000, seed=0)
    assert np.array_equal(val, v2)


def test_val_documents_never_appear_in_train(tok, tmp_path):
    """验证集的文档必须从训练分片里物理删除。子集要用好几轮（epochs > 1）时也不能混进来。"""
    meta = build(tmp_path, tok, {"en": 150, "zh": 60}, {"en": 0.5, "zh": 0.5},
                 total_tokens=60_000, val_tokens=1500, shard_tokens=8000)
    out = tmp_path / "out"
    val = {m for name in meta["val"] for m in markers(tok, out / f"val_{name}.bin")}
    train = {m for s in meta["train"]["shards"] for m in markers(tok, out / s["file"])}
    assert val and train and val.isdisjoint(train)
    assert meta["train"]["by_subset"]["zh"]["epochs"] > 1   # zh 文档少，被用了不止一轮


def test_shards_start_at_document_boundaries_and_match_meta(tok, tmp_path):
    meta = build(tmp_path, tok, {"en": 100, "zh": 100}, {"en": 1, "zh": 1},
                 total_tokens=30_000, val_tokens=1000, shard_tokens=5000)
    out = tmp_path / "out"
    assert len(meta["train"]["shards"]) > 3
    for s in meta["train"]["shards"]:
        arr = np.fromfile(out / s["file"], dtype=DTYPE)
        assert arr[0] == tok.eot_id and arr.size == s["tokens"]
    assert meta["tokenizer"] == tok.fingerprint
    assert json.loads((out / "meta.json").read_text("utf-8")) == meta


def test_mixing_is_smooth():
    """7:3 的配比：不只是总量对，任意连续 100 篇里的比例也要接近 7:3（顺序读取时模型看到的就是这个）。"""

    class FakeReader:
        def __len__(self):
            return 1000

        def n_tokens(self, i):
            return 100

    pools = {"a": (FakeReader(), np.arange(1000)), "b": (FakeReader(), np.arange(1000))}
    out = [name for name, _ in Mixer(pools, {"a": 0.7, "b": 0.3}, seed=0).run(100_000)]
    for start in range(0, len(out) - 100, 50):
        share = out[start : start + 100].count("a") / 100
        assert abs(share - 0.7) <= 0.02, (start, share)


def test_repeated_subset_uses_every_document_evenly():
    """文档不够时多用几轮，但每篇被用的次数最多差 1：先把一轮用完，才开始下一轮。"""

    class FakeReader:
        def n_tokens(self, i):
            return 10

    pools = {"small": (FakeReader(), np.arange(7))}
    m = Mixer(pools, {"small": 1.0}, seed=0)
    counts = np.bincount([d for _, d in m.run(10 * 23)], minlength=7)
    assert counts.max() - counts.min() <= 1
    assert m.epochs("small") == pytest.approx(23 / 7)


def test_check_meta_catches_wrong_tokenizer_and_truncated_shard(tok, tmp_path):
    meta = build(tmp_path, tok, {"en": 80}, {"en": 1}, total_tokens=10_000, val_tokens=1000, shard_tokens=4000)
    out = tmp_path / "out"
    shards = [out / s["file"] for s in meta["train"]["shards"]]
    check_meta(shards, tok.fingerprint)  # 正常情况不报错
    with pytest.raises(ValueError, match="分词器"):
        check_meta(shards, "0000000000000000")
    data = shards[0].read_bytes()
    shards[0].write_bytes(data[: len(data) // 2])  # 模拟拷贝中断
    with pytest.raises(ValueError, match="拷贝不完整"):
        check_meta(shards, tok.fingerprint)


def test_check_meta_ignores_dirs_without_meta(tmp_path):
    """合成数据没有 meta.json，不做检查。"""
    p = tmp_path / "train_000.bin"
    np.zeros(10, dtype=DTYPE).tofile(p)
    check_meta([p], "anything")
