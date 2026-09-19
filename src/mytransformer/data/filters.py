"""文档级质量过滤（P2-1）。

每条规则是一个函数：输入文本，返回**丢弃原因**（字符串），保留则返回 None。
返回原因而不是 True/False，是为了统计"每条规则各丢了多少"——报告里的过滤漏斗表就从这里来，
调阈值时也能一眼看出是哪条规则在"误杀"。

三套规则，按子集选用（`FILTERS`）：

- **英文网页**：Gopher 规则（DeepMind, Rae et al. 2021, 附录 A），数值照原文
- **中文网页**：Gopher 的重复度规则照用；按"词"算的规则换成按字算，另加汉字占比
- **代码**：StarCoder 的规则（行太长、字母数字太少、自动生成的文件）

这些来源（fineweb-edu、chinese-fineweb-edu、finemath）已经用质量分类器筛过，
所以这里不再做困惑度过滤，主要拦截分类器不管的东西：重复、模板页、乱码。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

Filter = Callable[[str], "str | None"]

WORD_RE = re.compile(r"\S+")
HAN_RE = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")  # CJK 扩展 A、基本区、兼容区
# Gopher 用的停用词：一篇正常英文里至少出现 2 个。表格、代码、关键词堆砌通常一个都没有
STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
BULLETS = ("•", "·", "●", "*", "-", "‣", "▪")


# ---------------------------------------------------------------- 重复度（中英通用）
#
# 实现照 datatrove 的 GopherRepetitionFilter（FineWeb 用的就是它），两个容易写错的细节：
# 1. 分母是**整篇文本的字符数**（含空格和换行），不是词的字符数之和
# 2. 统计重复 n-gram 时，找到一个重复就**跳过这 n 个词**，重叠的部分不重复计数
# 细节不一样，同样的阈值就会多杀很多正常文档：fineweb-edu 已经用这套规则过滤过，
# 实现对了的话，在它上面再跑一遍应该几乎什么都不丢（见 tests/test_filters.py）


def _dup_fraction(items: list[str], text_len: int) -> tuple[float, float]:
    """重复出现的条目：(占条数的比例, 占全文字符的比例)。第一次出现不算重复。"""
    seen: set[str] = set()
    dup_n = dup_chars = 0
    for x in items:
        if x in seen:
            dup_n += 1
            dup_chars += len(x)
        else:
            seen.add(x)
    return dup_n / max(1, len(items)), dup_chars / max(1, text_len)


def _top_ngram_chars(tokens: list[str], n: int, sep: str) -> int:
    """最常见的那个 n-gram 一共占了多少字符。"Click here click here click here" 这类会很高。"""
    grams = [sep.join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0
    gram, count = Counter(grams).most_common(1)[0]
    return len(gram) * count


def _dup_ngram_chars(tokens: list[str], n: int, sep: str) -> int:
    """从左往右扫，遇到见过的 n-gram 就记下它的字符数并跳过这 n 个 token。"""
    seen: set[str] = set()
    chars = i = 0
    while i < len(tokens) - n + 1:
        gram = sep.join(tokens[i : i + n])
        if gram in seen:
            chars += len(gram)
            i += n
        else:
            seen.add(gram)
            i += 1
    return chars


# Gopher 附录 A 表 A1 的阈值（按"词"）
TOP_NGRAM_MAX = {2: 0.20, 3: 0.18, 4: 0.16}
DUP_NGRAM_MAX = {5: 0.15, 6: 0.14, 7: 0.13, 8: 0.12, 9: 0.11, 10: 0.10}


def repetition(text: str, tokens: list[str] | None = None, *, sep: str = " ",
               n_scale: int = 1, by_count: bool = True) -> str | None:
    """tokens 为 None 时只查重复行和重复段落。

    n_scale：n-gram 的 n 乘以它（中文按字切时用）。
    by_count=False：不按"重复行/段落的条数"判断，只按它们占的字符。论坛页面里 "Reply"、"Posts: 2"
    这类很短的界面文字会重复几十次，按条数算超标，但它们只占全文很小一部分。
    """
    text_len = len(text)
    paras = [p for p in re.split(r"\n{2,}", text.strip()) if p]
    para_n, para_c = _dup_fraction(paras, text_len)
    if by_count and para_n > 0.30:
        return "重复段落（按段数）"
    if para_c > 0.20:
        return "重复段落（按字符）"
    lines = [ln for ln in re.split(r"\n+", text) if ln]
    line_n, line_c = _dup_fraction(lines, text_len)
    if by_count and line_n > 0.30:
        return "重复行（按行数）"
    if line_c > 0.20:
        return "重复行（按字符）"
    if tokens is None:
        return None
    for n, limit in TOP_NGRAM_MAX.items():
        if _top_ngram_chars(tokens, n * n_scale, sep) / text_len > limit:
            return f"最高频 {n}-gram 占比过高"
    for n, limit in DUP_NGRAM_MAX.items():
        if _dup_ngram_chars(tokens, n * n_scale, sep) / text_len > limit:
            return f"重复 {n}-gram 占比过高"
    return None


# ---------------------------------------------------------------- 英文


def gopher_quality(text: str, words: list[str]) -> str | None:
    n = len(words)
    if n < 50:
        return "太短（< 50 词）"
    if n > 100_000:
        return "太长（> 10 万词）"
    mean_len = sum(len(w) for w in words) / n
    if not 3 <= mean_len <= 10:
        return "平均词长异常"
    if (text.count("#") + text.count("...") + text.count("…")) / n > 0.1:
        return "# 或省略号过多"
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        if sum(ln.lstrip().startswith(BULLETS) for ln in lines) / len(lines) > 0.9:
            return "几乎全是列表项"
        if sum(ln.rstrip().endswith(("...", "…")) for ln in lines) / len(lines) > 0.3:
            return "省略号结尾的行过多"
    if sum(bool(re.search(r"[^\W\d_]", w)) for w in words) / n < 0.8:
        return "含字母的词不到 80%"
    if len(STOP_WORDS & {w.lower() for w in words}) < 2:
        return "停用词不足"
    return None


def english(text: str) -> str | None:
    words = WORD_RE.findall(text)
    return gopher_quality(text, words) or repetition(text, words)


# ---------------------------------------------------------------- 中文


def chinese(text: str) -> str | None:
    chars = [c for c in text if not c.isspace()]
    n = len(chars)
    if n < 100:
        return "太短（< 100 字）"
    if n > 300_000:
        return "太长（> 30 万字）"
    if len(HAN_RE.findall(text)) / n < 0.4:
        return "汉字不到 40%"
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines and sum(ln.rstrip().endswith(("...", "…", "……")) for ln in lines) / len(lines) > 0.3:
        return "省略号结尾的行过多"
    # 中文没有空格，n-gram 按"字"算。Gopher 的 n 是按"词"定的，一个中文词常常就有 2–4 个字，
    # n 不放大的话，常用词组、反复出现的术语这些自然的短重复累计起来就会超标。实测 chinese-fineweb-edu
    # 上，只算重复度规则：n×1 丢 22%、n×2 丢 6.0%、n×3 丢 3.8%，n×3 时被丢的基本都是真正整段重复的文档
    # （校准过程见 docs/learn/p2-1-filters.md）
    return repetition(text, chars, sep="", n_scale=3)


# ---------------------------------------------------------------- 代码

AUTOGEN_RE = re.compile(r"auto-?generated|do not edit|generated by", re.I)


def code(text: str) -> str | None:
    lines = text.split("\n")
    if len(text) < 50:
        return "太短"
    if max(len(ln) for ln in lines) > 1000:
        return "有超过 1000 字符的行（多半是压缩过的代码或数据）"
    if sum(len(ln) for ln in lines) / len(lines) > 100:
        return "平均行长 > 100"
    if sum(c.isalnum() for c in text) / len(text) < 0.25:
        return "字母数字不到 25%"
    if AUTOGEN_RE.search("\n".join(lines[:5])):
        return "自动生成的文件"
    return None


# ---------------------------------------------------------------- 百科、数学：已经很干净，只拦截极端情况


def light(text: str, min_chars: int = 200, by_count: bool = True) -> str | None:
    n = len(text.strip())
    if n < min_chars:
        return f"太短（< {min_chars} 字符）"
    if n > 1_000_000:
        return "太长（> 100 万字符）"
    return repetition(text, by_count=by_count)


FILTERS: dict[str, Filter] = {
    "en_web": english,
    "en_web_dclm": english,
    "zh_web": chinese,
    "code": code,
    # finemath 里有很多论坛页（见 repetition 的 by_count）；很短的解题过程也有价值，下限放到 100
    "math": lambda t: light(t, min_chars=100, by_count=False),
    "en_wiki": light,
    "zh_wiki": lambda t: light(t, min_chars=100),  # 一个汉字的信息量约等于英文 2–3 个字母
}


def apply(subset: str, text: str) -> str | None:
    """按子集选规则。返回丢弃原因，保留返回 None。"""
    return FILTERS[subset](text)
