"""去重（P2-2）：文档级 MinHash-LSH 近似去重 + 段落级精确去重。

**为什么要去重**：同一篇文章被转载几百次，模型就会背下它；重复数据还白白占用训练预算。

**文档级：MinHash-LSH**。两篇文档有多像，用 Jaccard 相似度衡量：

    J(A, B) = |A ∩ B| / |A ∪ B|        A、B 是两篇文档的"shingle"（连续 5 个词）集合

两两比较要 N² 次，几百万篇文档做不到。MinHash 把每篇文档压成 256 个整数的"签名"，
两个签名里相等的位置所占比例 ≈ J（这是 MinHash 的核心性质，见 test_signature_estimates_jaccard）。
LSH 再把签名切成 32 段、每段 8 个数，任何一段完全相同的两篇文档才拿出来比较。
这样只比较"很可能相似"的少数几对，而不是全部 N² 对。

**为什么是 256 而不是常见的 128**：在中文维基 2000 篇上和暴力枚举对比（6 个随机种子，
scripts/calibrate_dedup.py）：128 个哈希时精确率 51–84%、召回率 77–95%；256 个时 86–96%、89–98%。
中文维基有大量模板条目，彼此的 J 在 0.75 左右，128 个哈希的估计噪声（±0.04）会把其中一部分推过阈值。
代价是签名慢约 1.7 倍。

**段落级：精确去重**。整篇不重复、但包含相同段落的情况很多：cookie 提示、版权声明、"本文转载自……"。
在很多篇文档里出现过的段落，只在第一次出现的地方保留。**代码不做**：代码里重复的块（许可证头、
常见的样板函数）删掉可能让文件不完整，见 PARAGRAPH_DEDUP_SKIP。
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter
from collections.abc import Iterable

import numpy as np

MERSENNE = np.uint64((1 << 61) - 1)
MAX32 = np.uint64(0xFFFFFFFF)
_PUNCT_RE = re.compile(r"[\W_]+")


def normalize(text: str) -> str:
    """去重前的归一化：NFKC、小写、标点和空白都换成一个空格。
    "Hello,  World!" 和 "hello world" 应当被当成同一段文字。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return _PUNCT_RE.sub(" ", text).strip()


def shingles(text: str, n: int = 5, by_char: bool = False) -> set[str]:
    """文档的 shingle 集合：连续 n 个词（中文按连续 n 个字）。文档太短时整篇算一个。"""
    norm = normalize(text)
    units = list(norm.replace(" ", "")) if by_char else norm.split()
    if len(units) < n:
        return {"".join(units) if by_char else " ".join(units)}
    sep = "" if by_char else " "
    return {sep.join(units[i : i + n]) for i in range(len(units) - n + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


class MinHasher:
    """num_perm 个哈希函数 h_i(x) = (a_i·x + b_i) mod (2^61 − 1)，取低 32 位。

    x 是 shingle 的 crc32（32 位）。a、b 也限制在 32 位以内，a·x + b 就不会超过 uint64，
    numpy 里不会悄悄溢出。种子固定，同一篇文档永远得到同一个签名。
    """

    def __init__(self, num_perm: int = 256, ngram: int = 5, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 1 << 32, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 32, num_perm, dtype=np.uint64)
        self.num_perm, self.ngram = num_perm, ngram

    def signature(self, text: str, by_char: bool = False) -> np.ndarray:
        sh = shingles(text, self.ngram, by_char)
        x = np.fromiter((zlib.crc32(s.encode("utf-8")) for s in sh), dtype=np.uint64, count=len(sh))
        sig = np.full(self.num_perm, MAX32, dtype=np.uint64)
        for start in range(0, len(x), 4096):  # 分块：长文档一次算完会占几百 MB
            chunk = x[start : start + 4096]
            h = (np.outer(self.a, chunk) + self.b[:, None]) % MERSENNE & MAX32
            np.minimum(sig, h.min(axis=1), out=sig)
        return sig.astype(np.uint32)


def estimated_jaccard(s1: np.ndarray, s2: np.ndarray) -> float:
    return float(np.mean(s1 == s2))


def lsh_candidate_prob(j: float, bands: int, rows: int) -> float:
    """相似度为 j 的两篇文档，至少有一段完全相同（成为候选）的概率：1 − (1 − j^r)^b。"""
    return 1 - (1 - j**rows) ** bands


def _find(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def near_duplicates(sigs: np.ndarray, bands: int = 32, threshold: float = 0.8) -> np.ndarray:
    """返回每篇文档所属簇的"代表"（簇里下标最小的文档）。代表是自己的文档保留，其余是重复。

    sigs：(N, num_perm) 的签名矩阵，行的顺序就是优先级（靠前的优先保留）。
    1. LSH：按段分桶，同一个桶里的文档成为候选对
    2. 用签名估计的 Jaccard 复核候选对，≥ threshold 才算重复（LSH 只保证"可能相似"）
    3. 并查集把重复关系连成簇
    """
    n, num_perm = sigs.shape
    rows = num_perm // bands
    parent = np.arange(n)
    for b in range(bands):
        band = np.ascontiguousarray(sigs[:, b * rows : (b + 1) * rows])
        keys = band.view(np.dtype((np.void, band.dtype.itemsize * rows))).ravel()
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        # 桶的边界：排序后键值发生变化的位置
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        ends = np.r_[starts[1:], n]
        # 绝大多数桶只有一篇文档。几百万篇时，逐个桶用 Python 循环要跑几千万次，
        # 所以先用 numpy 挑出至少两篇的桶，只遍历它们
        multi = np.flatnonzero(ends - starts >= 2)
        for s, e in zip(starts[multi], ends[multi]):
            members = order[s:e]
            # 小桶两两复核；大桶（常见于大量模板页）只和桶里第一篇比，避免平方级开销
            pairs = ([(members[x], members[y]) for x in range(len(members)) for y in range(x + 1, len(members))]
                     if len(members) <= 32 else [(members[0], m) for m in members[1:]])
            for i, j in pairs:
                if estimated_jaccard(sigs[i], sigs[j]) >= threshold:
                    ri, rj = _find(parent, int(i)), _find(parent, int(j))
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)  # 下标小的做代表
    return np.array([_find(parent, i) for i in range(n)])


# ---------------------------------------------------------------- 段落级

PARAGRAPH_DEDUP_SKIP = {"code"}
_PARA_SPLIT = re.compile(r"(\n{2,})")


def _para_key(p: str) -> int:
    return zlib.crc32(normalize(p).encode("utf-8"))


def paragraph_keys(doc: str, min_chars: int = 30) -> set[int]:
    """这篇文档里参与去重的段落（的哈希），同一段在一篇里出现多次只算一次。"""
    return {_para_key(p) for p in _PARA_SPLIT.split(doc)[::2] if len(p.strip()) >= min_chars}


def common_paragraphs(docs: Iterable[str], min_docs: int = 3, min_chars: int = 30) -> set[int]:
    """第一遍：统计每个段落出现在几篇文档里，返回出现在 ≥ min_docs 篇里的段落（的哈希）。

    段落 = 按空行切开的块。短于 min_chars 的段落不参与（"参考文献"、"Introduction"
    这类小标题到处都有，删了反而破坏文章结构）。只存哈希，几百万篇文档也放得进内存。
    """
    counts: Counter[int] = Counter()
    for d in docs:
        counts.update(paragraph_keys(d, min_chars))
    return {k for k, c in counts.items() if c >= min_docs}


def strip_repeated(doc: str, common: set[int], seen: set[int], min_chars: int = 30) -> tuple[str, int]:
    """第二遍：删掉这篇文档里已经在前面出现过的常见段落。seen 在调用之间共享，按文档顺序更新。
    返回 (新文本, 删掉的段落数)。分隔的空行原样保留，删完的文本结构不变。"""
    parts = _PARA_SPLIT.split(doc)
    kept: list[str] = []
    removed = 0
    for idx in range(0, len(parts), 2):
        p = parts[idx]
        key = _para_key(p) if len(p.strip()) >= min_chars else None
        if key is not None and key in common:
            if key in seen:
                removed += 1
                continue
            seen.add(key)
        kept.append(p)
        if idx + 1 < len(parts):
            kept.append(parts[idx + 1])
    return "".join(kept).strip("\n"), removed


def paragraph_dedup(docs: list[str], min_docs: int = 3, min_chars: int = 30) -> tuple[list[str], int]:
    """两遍合在一起的便捷版本，文档都在内存里时用。"""
    common = common_paragraphs(docs, min_docs, min_chars)
    seen: set[int] = set()
    out, total = [], 0
    for d in docs:
        new, n = strip_repeated(d, common, seen, min_chars)
        out.append(new)
        total += n
    return out, total
