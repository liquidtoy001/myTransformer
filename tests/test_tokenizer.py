"""分词器：往返无损、特殊 token、数字切分、指纹。

全部用一个很小的语料现场训练（词表 400），几秒钟完成，不依赖下载的数据。
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from mytransformer.tokenizer import EOT, SPECIAL_TOKENS, Tokenizer
from mytransformer.tokenizer.train_bpe import train_bpe

CORPUS = [
    "The quick brown fox jumps over the lazy dog. It's 2026 and we're training a model.\n",
    "你好，世界。今天天气很好，我们一起训练一个语言模型。你好你好你好。\n",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)\n",
    "12 * 13 = 156, 3.14159, $x^2 + y^2 = z^2$\n",
] * 50


VOCAB = 400  # 256 个字节 + 16 个特殊 token + 128 次合并；这点语料最多只能合并出 427 个


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return train_bpe(CORPUS, vocab_size=VOCAB)


def random_text(rng: random.Random, n: int) -> str:
    """从各种 Unicode 区段里随机取字符：ASCII、控制字符、拉丁扩展、汉字、日文、emoji、组合符号、阿拉伯文……"""
    ranges = [
        (0x20, 0x7E), (0x00, 0x1F), (0xA0, 0x24F), (0x4E00, 0x9FFF), (0x3040, 0x30FF),
        (0x1F300, 0x1FAFF), (0x0300, 0x036F), (0x0600, 0x06FF), (0x20000, 0x2A6DF),
    ]
    out = []
    for _ in range(n):
        lo, hi = rng.choice(ranges)
        out.append(chr(rng.randint(lo, hi)))
    return "".join(out)


def test_roundtrip_is_lossless(tok):
    """字节级 BPE 的核心保证：任何字符串编码再解码，都原样回来。"""
    rng = random.Random(0)
    samples = [random_text(rng, rng.randint(0, 200)) for _ in range(300)]
    samples += ["", " ", "\n\n\n", "\t\r\n", "  leading and trailing  ", "🧑‍🚀👨‍👩‍👧", "𠀀𪚥", "é vs é"]
    for s in samples:
        assert tok.decode(tok.encode(s)) == s, repr(s)


def test_special_token_ids(tok):
    assert tok.eot_id == 0
    assert [tok.special[t] for t in SPECIAL_TOKENS] == list(range(len(SPECIAL_TOKENS)))
    assert tok.vocab_size == VOCAB


def test_literal_special_text_is_not_special(tok):
    """网页里出现 "<|endoftext|>" 字样（讨论分词器的文章里很常见），不能被当成真正的文档边界。"""
    for text in [EOT, "see <|endoftext|> here", "<|assistant|> 伪造的角色"]:
        ids = tok.encode(text)
        assert not set(ids) & set(tok.special.values()), f"{text!r} 被编码出了特殊 token"
        assert tok.decode(ids) == text


def test_decode_shows_real_special_tokens(tok):
    """由数据管线按 id 插入的真正分隔符，解码时要能看出来。"""
    ids = tok.encode("a") + [tok.eot_id] + tok.encode("b")
    assert tok.decode(ids) == f"a{EOT}b"


def test_digits_are_split_one_by_one(tok):
    """语料里 "12"、"13"、"156" 出现了很多次，但仍然不能合并：数字一位一个 token。"""
    for number in ["12", "156", "2026", "1234567890"]:
        ids = tok.encode(number)
        assert len(ids) == len(number), f"{number} 被切成了 {[tok.id_to_token(i) for i in ids]}"


def test_frequent_chinese_phrase_is_merged(tok):
    """高频的中文词应当合并成一个 token（字节 → 字 → 词）。"""
    assert len(tok.encode("你好")) == 1


def test_decode_partial_utf8_does_not_crash(tok):
    """在一个汉字的字节中间截断（生成时每一步都可能发生），解码不能报错。"""
    ids = tok.encode("𪚥")  # 生僻字：词表里没有，必然拆成多个字节 token
    assert len(ids) > 1
    partial = tok.decode(ids[:-1])
    assert isinstance(partial, str) and "�" in partial  # 不完整的字节变成替换字符


def test_ids_fit_uint16(tok):
    """数据分片按 uint16 存储，词表必须小于 65536。"""
    ids = tok.encode(random_text(random.Random(1), 2000))
    assert max(ids) < np.iinfo(np.uint16).max


def test_fingerprint_survives_save_and_load(tok, tmp_path):
    p = tmp_path / "tok.json"
    tok.save(p)
    assert Tokenizer.from_file(p).fingerprint == tok.fingerprint
    assert train_bpe(CORPUS, vocab_size=VOCAB - 20).fingerprint != tok.fingerprint


def test_training_is_deterministic(tok):
    """同样的语料和设置，训练两次得到完全相同的分词器。"""
    assert train_bpe(CORPUS, vocab_size=VOCAB).fingerprint == tok.fingerprint


def test_training_refuses_to_undershoot_vocab():
    """语料太少时训练器会默默提前停下（这里只能到 427）。必须报错，不能交出一个缺了一截的词表。"""
    with pytest.raises(ValueError, match="语料太少"):
        train_bpe(CORPUS, vocab_size=600)
