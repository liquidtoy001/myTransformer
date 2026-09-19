"""去重：MinHash 签名能估计 Jaccard、LSH 找得到近似重复、段落去重只删该删的。"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from mytransformer.data.dedup import (
    MinHasher,
    estimated_jaccard,
    jaccard,
    lsh_candidate_prob,
    near_duplicates,
    normalize,
    paragraph_dedup,
    shingles,
)

WORDS = [f"w{i}" for i in range(3000)]


def doc(rng: random.Random, n: int = 400) -> list[str]:
    return [rng.choice(WORDS) for _ in range(n)]


def edit(words: list[str], frac: float, rng: random.Random) -> list[str]:
    """随机替换一部分词，得到一篇"改了几处"的近似副本。"""
    out = list(words)
    for i in rng.sample(range(len(out)), int(len(out) * frac)):
        out[i] = rng.choice(WORDS)
    return out


def test_normalize_ignores_case_punctuation_and_spacing():
    assert normalize("Hello,  World!\n") == normalize("hello world") == "hello world"
    assert normalize("ＡＢＣ　１２３") == "abc 123"  # 全角转半角（NFKC）


def test_signature_estimates_jaccard():
    """MinHash 的核心性质：两个签名相等位置的比例 ≈ 真实 Jaccard。
    256 个哈希时标准差约 √(J(1−J)/256) ≤ 0.032，这里允许 0.1（3 倍多）。"""
    rng = random.Random(0)
    mh = MinHasher()
    base = doc(rng)
    for frac in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        other = edit(base, frac, rng)
        a, b = " ".join(base), " ".join(other)
        true_j = jaccard(shingles(a), shingles(b))
        est = estimated_jaccard(mh.signature(a), mh.signature(b))
        assert abs(est - true_j) < 0.1, (frac, true_j, est)


def test_signature_is_deterministic_across_processes():
    """签名只取决于文本和种子。Python 自带的 hash() 对字符串每次启动都不同（PYTHONHASHSEED），
    同一个进程里看不出来，所以这里另起一个进程再算一次。去重分多个进程、多个作业跑，签名必须一致。"""
    import os
    import subprocess
    import sys

    text = "the same document appears here again and again " * 5
    code = ("from mytransformer.data.dedup import MinHasher;"
            f"print(MinHasher().signature({text!r}).tolist())")
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert json.loads(out.stdout) == MinHasher().signature(text).tolist()


def test_lsh_curve():
    """32 段 × 8 行：相似度 0.8 的一对几乎一定成为候选，0.3 的几乎一定不会。
    中间（0.5 左右约 12%）成为候选的，由签名估计的 Jaccard 再复核一次。"""
    assert lsh_candidate_prob(0.8, 32, 8) > 0.99
    assert lsh_candidate_prob(0.3, 32, 8) < 0.01


def test_near_duplicates_found_and_lowest_index_kept():
    rng = random.Random(1)
    mh = MinHasher()
    a = doc(rng)
    texts = [
        " ".join(a),                                  # 0 原文
        " ".join(doc(rng)),                           # 1 无关
        " ".join(edit(a, 0.02, rng)),                 # 2 原文改了 2% 的词 → 重复
        " ".join(doc(rng)),                           # 3 无关
        " ".join(a).upper().replace(" ", ",  "),      # 4 只改大小写和标点 → 重复
        " ".join(edit(a, 0.5, rng)),                  # 5 改了一半 → 不算重复
    ]
    rep = near_duplicates(np.stack([mh.signature(t) for t in texts]))
    assert rep.tolist() == [0, 1, 0, 3, 0, 5]


def test_chinese_near_duplicate_by_char():
    base = ("水循环是指水在陆地、海洋和大气之间不断运动的过程。太阳的热量使湖泊和海洋中的水蒸发，"
            "水蒸气上升到高空后遇冷凝结成云。云中的小水滴不断合并变大，最终以雨或雪的形式落回地面。")
    repost = "【转载】" + base + "（来源：网络）"
    other = "城市的排水系统也和水循环密切相关。暴雨来临时，地面硬化面积越大，雨水越难渗入土壤，容易形成内涝。"
    mh = MinHasher()
    sigs = np.stack([mh.signature(t, by_char=True) for t in (base, repost, other)])
    assert near_duplicates(sigs).tolist() == [0, 0, 2]


@pytest.mark.parametrize("n_docs, removed", [(3, 2), (2, 0)])
def test_paragraph_dedup_threshold(n_docs, removed):
    """同一段版权声明出现在 3 篇里：只保留第一篇里的。只出现在 2 篇里：不动。"""
    notice = "All content on this site is protected by copyright and may not be reproduced."
    docs = [f"Article {i} has its own opening paragraph number {i}.\n\n{notice}\n\nClosing words {i}."
            for i in range(n_docs)]
    out, n = paragraph_dedup(docs)
    assert n == removed
    assert notice in out[0]
    if removed:
        assert all(notice not in d for d in out[1:])
        assert out[1] == "Article 1 has its own opening paragraph number 1.\n\nClosing words 1."


def test_paragraph_dedup_keeps_short_headings():
    """"References" 这种短段落到处都有，不参与去重。"""
    docs = [f"Body text of article {i} goes here and is long enough.\n\nReferences\n\n[{i}] A paper."
            for i in range(5)]
    out, n = paragraph_dedup(docs)
    assert n == 0 and out == docs


def test_bucket_collision_is_not_a_duplicate():
    """LSH 只保证"可能相似"：两篇文档碰巧有一段（8 个数）完全相同，但其余 248 个都不同，
    估计的 J = 8/256 ≈ 0.03，必须经过复核被排除。不复核的话，真实数据上精确率会掉到 10% 以下。"""
    rng = np.random.default_rng(0)
    sigs = rng.integers(0, 1 << 32, (2, 256), dtype=np.uint64).astype(np.uint32)
    sigs[1, :8] = sigs[0, :8]
    assert near_duplicates(sigs).tolist() == [0, 1]
