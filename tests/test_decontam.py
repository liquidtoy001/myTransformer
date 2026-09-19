"""去污染：抓得住题目的转载，放得过只是引用了常见说法的正常文章。"""

from __future__ import annotations

from mytransformer.data.decontam import EvalIndex, ngrams, units

QUESTION = ("A man is sitting on a roof. He starts pulling up roofing on a roof, "
            "then carefully throws the old shingles into a large dumpster below.")
ZH_QUESTION = "下列关于资本结构理论的说法中，不正确的是：按照优序融资理论的观点，管理者偏好首选留存收益筹资"


def make_index() -> EvalIndex:
    idx = EvalIndex()
    idx.add("hellaswag", QUESTION)
    idx.add("ceval", ZH_QUESTION)
    return idx


def test_units_split_han_chars_and_words():
    assert units("Hello, 世界 ABC-def") == ["hello", "世", "界", "abc", "def"]


def test_window_length_counts_han_as_half():
    """13 个英文词正好一个窗口；汉字按半个算，26 个字才一个窗口，25 个字一个都没有。"""
    assert len(ngrams([f"w{i}" for i in range(13)])) == 1
    assert len(ngrams(list("字" * 26))) == 1
    assert ngrams(list("字" * 25)) == []
    assert len(ngrams(["a"] * 7 + list("字" * 12))) == 1   # 混排：7 + 12×0.5 = 13


def test_repost_with_different_case_and_punctuation_is_caught():
    idx = make_index()
    doc = "Today's quiz!!!\n\na MAN is sitting on a roof -- he starts pulling up roofing on a roof; THEN carefully..."
    assert idx.contaminated(doc) == "hellaswag"


def test_chinese_question_in_a_quiz_page_is_caught():
    idx = make_index()
    doc = "财务管理每日一练\n\n1、" + ZH_QUESTION + "。\nA. 正确 B. 错误"
    assert idx.contaminated(doc) == "ceval"


def test_shared_short_phrase_is_not_contamination():
    """只和题目共享 12 个词（不够一个窗口），正常文章不能被误伤。"""
    idx = make_index()
    doc = "Yesterday a man is sitting on a roof. He starts pulling up roofing and then goes home."
    assert idx.contaminated(doc) is None


def test_chinese_fixed_phrase_alone_is_not_contamination():
    """校准时的主要误判来源：题目里引用的 13 个字的固定说法。窗口约 26 个字后，只出现这一句不算污染。"""
    idx = EvalIndex()
    idx.add("ceval", "党的二十大报告指出，全面建设社会主义现代化国家，最艰巨最繁重的任务仍然在农村。这体现了")
    doc = "我们要团结一心，为全面建设社会主义现代化国家而努力奋斗。"
    assert idx.contaminated(doc) is None


def test_too_short_items_are_counted_not_indexed():
    idx = EvalIndex()
    idx.add("piqa", "How do I ready a guinea pig cage?")
    assert len(idx) == 0 and idx.too_short["piqa"] == 1


def test_excluded_ngrams_no_longer_match():
    idx = make_index()
    doc = "a man is sitting on a roof he starts pulling up roofing on a roof"
    assert idx.contaminated(doc) == "hellaswag"
    assert idx.exclude(idx.hits(doc)) >= 1
    assert idx.contaminated(doc) is None
