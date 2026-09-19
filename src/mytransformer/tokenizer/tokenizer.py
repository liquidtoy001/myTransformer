"""分词器：字节级 BPE，词表 32768。

三个设计决定（理由见 docs/learn/p1-1-tokenizer.md）：

1. **字节级**（byte-level）：先把文本变成 UTF-8 字节，再在字节上做 BPE。
   任何字符串都能被编码，不存在"未知字符"；生僻字、emoji 最坏也就是拆成几个字节。
2. **预切分正则**：BPE 只在切出来的片段内部合并，不会跨过片段边界。
   用的是 GPT-4 / Llama-3 系列的写法，只改了一处：**数字一位一切**，
   让 "1234" 和 "12" 里的 "1" 永远是同一个 token，对小模型学算术更友好。
3. **特殊 token 不从文本里识别**：数据里出现的 "<|endoftext|>" 字样按普通文本编码。
   真正的文档分隔符只由数据管线按 id 插入，对话角色只由 SFT 代码按 id 插入。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from tokenizers import Regex, decoders, models, pre_tokenizers
from tokenizers import Tokenizer as HFTokenizer

EOT = "<|endoftext|>"
ROLE_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|tool|>"]
PAD = "<|pad|>"
N_RESERVED = 10
# 顺序即 id：<|endoftext|> 必须是 0（数据分片和训练代码都依赖这一点）
SPECIAL_TOKENS = [EOT, *ROLE_TOKENS, PAD, *[f"<|reserved_{i}|>" for i in range(N_RESERVED)]]

# GPT-4（cl100k）风格的预切分，唯一改动是 \p{N} 不再是 \p{N}{1,3}：数字一位一切
SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"   # 英文缩写：it's → it + 's
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"      # 一串字母（含汉字），可带一个前导符号或空格
    r"|\p{N}"                         # 单个数字
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"     # 一串标点符号
    r"|\s*[\r\n]+"                    # 换行
    r"|\s+(?!\S)"                     # 行尾空白
    r"|\s+"                           # 其余空白
)


def build_untrained() -> HFTokenizer:
    """还没训练的空分词器：只定好切分规则和解码方式。"""
    tok = HFTokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


class Tokenizer:
    """对 HuggingFace tokenizers 的薄封装，只暴露本项目需要的接口。"""

    def __init__(self, tok: HFTokenizer) -> None:
        tok.encode_special_tokens = True  # 文本里的 "<|endoftext|>" 字样按普通文本编码
        self._tok = tok
        self.special = {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}
        missing = [t for t, i in self.special.items() if i is None]
        if missing:
            raise ValueError(f"分词器缺少特殊 token：{missing}")
        if self.special[EOT] != 0:
            raise ValueError(f"{EOT} 的 id 必须是 0，实际是 {self.special[EOT]}")

    # ---- 读写 ----

    @classmethod
    def from_file(cls, path: str | Path) -> Tokenizer:
        return cls(HFTokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    # ---- 编码 / 解码 ----

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: list[int]) -> str:
        """特殊 token 会以 "<|endoftext|>" 这样的字面形式出现，方便调试时看清文档边界。"""
        return self._tok.decode(ids, skip_special_tokens=False)

    def id_to_token(self, i: int) -> str:
        return self._tok.id_to_token(i)

    # ---- 属性 ----

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def eot_id(self) -> int:
        return self.special[EOT]

    @property
    def fingerprint(self) -> str:
        """分词器的指纹。写进每个数据分片的 meta.json：换了分词器，旧分片立刻对不上。"""
        return hashlib.sha256(self._tok.to_str().encode("utf-8")).hexdigest()[:16]
