"""训练 BPE 分词器。

    python -m mytransformer.tokenizer.train_bpe --input "data/tokenizer_sample/*.txt" --out tokenizer/mt32k.json

BPE 的训练过程：从 256 个字节开始，反复找语料里出现次数最多的相邻一对 token，
把它们合并成一个新 token，直到词表达到 vocab_size。
"""

from __future__ import annotations

import argparse
import glob
from collections.abc import Iterable

from tokenizers import pre_tokenizers, trainers

from .tokenizer import SPECIAL_TOKENS, Tokenizer, build_untrained


def train_bpe(texts: Iterable[str], vocab_size: int = 32768, min_frequency: int = 2,
              show_progress: bool = False) -> Tokenizer:
    tok = build_untrained()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,            # 含特殊 token 在内的总数
        min_frequency=min_frequency,      # 只出现 1 次的组合不值得占一个词表位置
        special_tokens=SPECIAL_TOKENS,    # 按顺序占住 id 0..15
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 256 个字节全部先放进词表：任何输入都能编码
        show_progress=show_progress,
    )
    tok.train_from_iterator(texts, trainer)
    # 语料里可合并的组合用完时，训练器会默默提前停下，词表比要求的小，而且不报任何错。
    # 模型的 vocab_size 是按要求的大小配置的，词表缺一截就意味着一部分 id 永远用不到。
    if tok.get_vocab_size() != vocab_size:
        raise ValueError(
            f"只训练出 {tok.get_vocab_size()} 个 token，达不到要求的 {vocab_size}："
            f"语料太少，或 min_frequency={min_frequency} 太高"
        )
    return Tokenizer(tok)


def _iter_files(paths: list[str], chunk_chars: int = 1 << 20):
    """按块读文件，避免一次把几个 GB 的样本全读进内存。"""
    for path in paths:
        with open(path, encoding="utf-8") as f:
            while chunk := f.read(chunk_chars):
                yield chunk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="训练语料的 glob，例如 data/tokenizer_sample/*.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input))
    if not paths:
        raise SystemExit(f"没有匹配 {args.input} 的文件")
    tok = train_bpe(_iter_files(paths), args.vocab_size, args.min_frequency, show_progress=True)
    tok.save(args.out)
    print(f"已保存 {args.out}：词表 {tok.vocab_size}，指纹 {tok.fingerprint}")


if __name__ == "__main__":
    main()
