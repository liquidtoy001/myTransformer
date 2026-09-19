"""去污染（P2-3）：训练语料里出现了评测题，整篇剔除。

**为什么必须做**：网页上到处是题库。HellaSwag 的上下文来自 WikiHow，MMLU、C-Eval 的题目
在各种刷题网站上都有。模型在预训练时见过题目和答案，评测分数就是在考"背没背过"，不是能力。

**怎么判断**：把每道评测题（题面 + 正确答案）切成连续 13 个单位的片段（13-gram，GPT-3 论文的做法），
训练文档里只要出现其中任何一个，就算污染。13 足够长，普通文章碰巧撞上的可能性很小；
又不会太长，题目被改写几个字、换了标点的转载也能抓到（先做和去重一样的归一化）。

**单位怎么切**：每个汉字单独算一个单位，其余按空格切词。这样中文、英文、中英混排用同一套逻辑，
不用先判断语言。窗口长度按"词"计：汉字算半个，所以中文窗口约 26 个字（见 ngrams）。

**两类不参与索引的片段**（都在真实数据上校准过，见 docs/learn/p2-3-decontam.md）：
- 整道题不够一个窗口长（PIQA 测试集只有一句目标、没有公开答案，66% 属于这种）
- 在普通语料里反复出现的"通用片段"：名人演讲、法律条文、固定的政治表述、"0123456789abcdef"。
  题目引用了它们，但语料里出现它们不代表见过这道题。列表由 scripts/calibrate_decontam.py
  在 1 GB 普通语料上统计得到，存在 configs/data/decontam_exclude.json
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .dedup import normalize

N = 13
_HAN = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_HAN_RE = re.compile(f"[{_HAN}]")
_UNIT_RE = re.compile(rf"[{_HAN}]|[^\s{_HAN}]+")


def units(text: str) -> list[str]:
    """归一化后切成单位：一个汉字一个单位，其余按空白切词。"""
    return _UNIT_RE.findall(normalize(text))


HAN_WEIGHT = 0.5


def ngrams(us: list[str], n: int = N) -> list[str]:
    """从每个位置起，取最短的一段连续单位，使"长度" ≥ n：英文词算 1，汉字算 HAN_WEIGHT。

    汉字按 1 算的话，13 个字只相当于 7 个词左右。C-Eval 的题目大量引用"全面建设社会主义现代化国家"
    （正好 13 个字）这类固定说法和名句，中文网页 9% 的文档会被误判（校准记录见 docs/learn）。
    汉字按 0.5 算，窗口约 26 个字，和英文的 13 个词大致相当。
    评测题和训练文档用同一个函数切，同一段文字从同一个位置起切出的窗口完全相同，所以仍然是精确匹配。
    """
    w = [HAN_WEIGHT if len(u) == 1 and _HAN_RE.match(u) else 1.0 for u in us]
    out: list[str] = []
    j, total = 0, 0.0
    for i in range(len(us)):
        while j < len(us) and total < n:
            total += w[j]
            j += 1
        if total < n:
            break
        out.append(" ".join(us[i:j]))
        total -= w[i]
    return out


class EvalIndex:
    """评测题 13-gram 的索引。存 n-gram 字符串本身而不是哈希：几百万个 n-gram 用 32 位哈希时，
    一篇几千词的训练文档撞上假阳性的概率不可忽略；用字符串就是精确匹配。"""

    def __init__(self, n: int = N) -> None:
        self.n = n
        self.bench_of: dict[str, str] = {}   # n-gram → 第一个包含它的评测集
        self.items: Counter[str] = Counter()  # 每个评测集有多少题
        self.too_short: Counter[str] = Counter()

    def add(self, bench: str, text: str) -> None:
        self.items[bench] += 1
        grams = ngrams(units(text), self.n)
        if not grams:  # 整道题都不够一个窗口长
            self.too_short[bench] += 1
            return
        for g in grams:
            self.bench_of.setdefault(g, bench)

    @classmethod
    def from_dir(cls, path: str | Path, n: int = N, exclude_file: str | Path | None = None) -> EvalIndex:
        """读 scripts/fetch_eval_sets.py 生成的 data/eval_sets/*.jsonl，再去掉通用片段。"""
        idx = cls(n)
        for f in sorted(Path(path).glob("*.jsonl")):
            for line in f.open(encoding="utf-8"):
                r = json.loads(line)
                idx.add(r["bench"], r["text"])
        if exclude_file is not None:
            idx.exclude(json.loads(Path(exclude_file).read_text("utf-8"))["ngrams"])
        return idx

    def __len__(self) -> int:
        return len(self.bench_of)

    def hits(self, text: str) -> list[str]:
        """文档里命中了哪些评测 n-gram（去重后按出现顺序）。"""
        found = dict.fromkeys(g for g in ngrams(units(text), self.n) if g in self.bench_of)
        return list(found)

    def contaminated(self, text: str) -> str | None:
        """命中则返回评测集名字，否则 None。和 filters 的接口一致：返回原因，None 表示保留。"""
        for g in ngrams(units(text), self.n):
            b = self.bench_of.get(g)
            if b is not None:
                return b
        return None

    def exclude(self, grams: Iterable[str]) -> int:
        """把通用片段移出索引，返回实际移除了几个。"""
        removed = 0
        for g in grams:
            if self.bench_of.pop(g, None) is not None:
                removed += 1
        return removed
