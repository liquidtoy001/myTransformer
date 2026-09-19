"""比较分词器的压缩率（P1-2）。

    python scripts/tokenizer_report.py tokenizer/mt32k.json data/tokenizers_ref/*.json

在 data/tokenizer_sample/eval/ 的每份评测文本上，对每个分词器算：

- **bytes/token**：每个 token 平均代表多少字节的 UTF-8 文本。越大越好，各种语言都能比
- **tokens/word**（英文、数学）：按空白切出的"词"平均被切成几个 token
- **tokens/字**（中文）：每个字符平均几个 token。汉字 UTF-8 占 3 字节，纯字节切分就是 3.0

评测文本在抽样时和训练语料分开（不同文件或不同位置），所以我们的分词器没有"见过答案"。
结果写到 reports/tokenizer_eval.json，并打印成 Markdown 表格。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer

EVAL_DIR = Path("data/tokenizer_sample/eval")
# 评测文本 → 按什么单位报告 fertility
UNIT = {"en_web": "word", "en_web_dclm": "word", "en_wiki": "word", "math": "word",
        "zh_web": "char", "zh_wiki": "char", "code": None}
WORD_RE = re.compile(r"\S+")


def load(path: str) -> tuple[str, HFTokenizer]:
    tok = HFTokenizer.from_file(path)
    tok.encode_special_tokens = True   # 所有分词器一视同仁：文本里的特殊 token 字样按普通文本算
    return Path(path).stem, tok


def count_tokens(tok: HFTokenizer, text: str) -> int:
    # 在空行之后切开、分批编码（空行本身留在前一段末尾，也计入 token）。
    # 整份几 MB 一次编码很慢；切开只影响跨段落边界的合并，对各分词器几乎一样
    docs = re.split(r"(?<=\n\n)", text)
    return sum(len(e.ids) for e in tok.encode_batch(docs, add_special_tokens=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizers", nargs="+")
    ap.add_argument("--out", default="reports/tokenizer_eval.json")
    args = ap.parse_args()

    toks = [load(p) for p in args.tokenizers]
    texts = {p.stem: p.read_text("utf-8") for p in sorted(EVAL_DIR.glob("*.txt"))}
    results: dict[str, dict] = {}
    for name, tok in toks:
        r = {"vocab_size": tok.get_vocab_size()}
        for src, text in texts.items():
            n_tok = count_tokens(tok, text)
            row = {"tokens": n_tok, "bytes_per_token": len(text.encode("utf-8")) / n_tok}
            if UNIT[src] == "word":
                row["tokens_per_word"] = n_tok / len(WORD_RE.findall(text))
            elif UNIT[src] == "char":
                row["tokens_per_char"] = n_tok / len(re.sub(r"\s", "", text))
            r[src] = row
        results[name] = r
        print(f"  {name} 完成", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1), "utf-8")

    srcs = list(texts)
    print("\n### bytes/token（越大越好）\n")
    print("| 分词器 | 词表 | " + " | ".join(srcs) + " |")
    print("|---|---:|" + "---:|" * len(srcs))
    for name, r in results.items():
        print(f"| {name} | {r['vocab_size']:,} | " +
              " | ".join(f"{r[s]['bytes_per_token']:.2f}" for s in srcs) + " |")
    print("\n### fertility（越小越好）：英文/数学 tokens/word，中文 tokens/字\n")
    fs = [s for s in srcs if UNIT[s]]
    print("| 分词器 | " + " | ".join(fs) + " |")
    print("|---|" + "---:|" * len(fs))
    for name, r in results.items():
        cells = [r[s].get("tokens_per_word", r[s].get("tokens_per_char")) for s in fs]
        print(f"| {name} | " + " | ".join(f"{c:.3f}" for c in cells) + " |")


if __name__ == "__main__":
    main()
