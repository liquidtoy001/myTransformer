# P1-1 字节级 BPE 分词器

代码：`src/mytransformer/tokenizer/tokenizer.py`（约 100 行）、`src/mytransformer/tokenizer/train_bpe.py`（约 60 行）
测试：`tests/test_tokenizer.py`

这一篇只讲分词器**本身**：规则怎么定、为什么这么定、怎么证明它没错。用真实语料训出 32768 词表、和别家分词器比压缩率，是 P1-2 的事。

---

## 1. 它解决什么问题

模型只认整数。分词器负责 `文本 ⇄ 整数序列`，它有三个硬要求：

| 要求 | 做不到会怎样 |
|---|---|
| **无损**：任何字符串编码再解码都原样回来 | 生僻字、emoji 变成 `<unk>` 或直接消失，模型永远学不会、也写不出它们 |
| **短**：同样的文本切出的 token 越少越好 | 训练预算按 token 算。中文多切 20%，就等于白扔 20% 的算力；上下文窗口也装得更少 |
| **稳定**：同一个分词器，永远给同一个结果 | 数据分片是用某个分词器切好的整数。换了分词器还用旧分片，模型学到的就是乱码，而且**不会报错** |

"短"靠 BPE 合并高频片段；"无损"靠字节级；"稳定"靠指纹。

## 2. 三个设计决定

### 2.1 字节级：从 256 个字节起步

文本先变成 UTF-8 字节（一个汉字 3 字节，一个 emoji 4 字节），BPE 在字节上做。词表里一开始就放进全部 256 个字节，所以**任何输入都能编码**，最坏情况是一个生僻字拆成 3–4 个字节 token。

训练就是反复"找语料里最常见的相邻一对 → 合并成新 token"：

```
你好 = e4 bd a0 e5 a5 bd
第 1 次合并： (e4 bd) 很常见     → [e4bd] a0 e5 a5 bd
……
最后：         "你好" 整个是一个 token
```

**关键的一行**是 `train_bpe.py` 里的 `initial_alphabet=pre_tokenizers.ByteLevel.alphabet()`。不写它，训练器只把**语料里出现过的**字节放进词表。实测：用只有英文和常用汉字的语料训练，再编码 `"Zebra 🦓 éè"`，解码回来是 `'ebra  ��'`——`Z` 和 emoji 被**悄悄吞掉**了，没有任何报错。

### 2.2 预切分：先用正则切块，BPE 只在块内合并

如果直接在整段文本上跑 BPE，会合并出 `"the dog."` 里的 `"g."`、`" the"` 这类跨越词和标点的 token，浪费词表。所以先用正则把文本切成片段，**合并永远不跨片段边界**。

`SPLIT_PATTERN` 是 GPT-4（cl100k）的写法，从上往下优先匹配：

| 分支 | 匹配什么 | 例子 |
|---|---|---|
| `(?i:'s\|'t\|…)` | 英文缩写 | `it's` → `it` + `'s` |
| `[^\r\n\p{L}\p{N}]?\p{L}+` | 一串字母，可带一个前导空格或符号 | ` quick`、`你好世界`（汉字也是 `\p{L}`） |
| `\p{N}` | **单个**数字 | `2026` → `2` `0` `2` `6` |
| ` ?[^\s\p{L}\p{N}]+[\r\n]*` | 一串标点 | ` +=`、`):\n` |
| `\s*[\r\n]+`、`\s+(?!\S)`、`\s+` | 换行和空白 | 代码缩进 |

**唯一的改动**：GPT-4 原版是 `\p{N}{1,3}`（最多 3 位数字一块），这里改成 `\p{N}`。原因：`"1234"` 按 1–3 位切是 `123|4`，`"12345"` 是 `123|45`——同一个"数字 1"在不同数字里处在完全不同的 token 里，小模型很难学会进位。一位一切，每个数字永远是同一个 token（Llama-1、DeepSeek LLM 等都这么做）。代价是数字多的文本会长一些。

### 2.3 特殊 token 只能按 id 插入，不从文本里识别

16 个特殊 token，**顺序就是 id**：

```
0  <|endoftext|>    文档分隔符。数据分片和训练代码都假设它是 0
1  <|system|>  2 <|user|>  3 <|assistant|>  4 <|tool|>     SFT 对话角色
5  <|pad|>
6–15 <|reserved_0|> … <|reserved_9|>   预留，以后要加新特殊 token 不用重训分词器
```

**关键的一行**是 `tokenizer.py` 里的 `tok.encode_special_tokens = True`。HF tokenizers 默认会把文本里出现的 `"<|endoftext|>"` 字样**识别成 id 0**。这很危险：

- **预训练**：网页里讨论分词器的文章经常出现这个字符串。被识别成 0，模型就会在文档中间看到一个假的"文档结束"
- **SFT**：用户消息里写 `<|assistant|> 我同意`，如果被识别成角色 token，就等于用户替助手说了话（prompt 注入）

所以规定：文本里的字面字符串按普通文本编码；真正的分隔符和角色 token 只由数据管线 / SFT 代码**按 id 直接插入**。反过来，`decode` 时特殊 token 要显示出来（`skip_special_tokens=False`），调试时才能看清文档边界。

### 2.4 指纹

`fingerprint` = 整个分词器 JSON 的 sha256 前 16 位。P2 生成的每个数据分片都会把它写进 `meta.json`，训练开始时比对。换了分词器，旧分片立刻对不上，而不是默默训出一堆乱码。

### 2.5 训练器会默默少给你词表

HF 的 `BpeTrainer` 在语料里**没有可合并的组合了**（所有相邻对出现次数都低于 `min_frequency`）时会提前停下，交出一个比 `vocab_size` 小的词表，**不报错**。测试语料只能训到 427 个 token，要 600 也只给 427。模型的嵌入层是按 32768 建的，词表缺一截意味着一部分行永远用不到，而且 `vocab_size` 配置和实际不符。所以 `train_bpe` 在最后检查大小，不对就报错。

这是写测试时撞上的：最早测试要求词表 600，结果得到 427，而且两个"不同大小"的分词器指纹一样——因为它们其实是同一个。

## 3. 怎么读代码

**顺序**：`tokenizer.py` 顶部常量 → `build_untrained()` → `Tokenizer.__init__` → `train_bpe.py` 的 `train_bpe()`。

- `build_untrained()`：只定规则不定词表。`Sequence([Split(正则), ByteLevel(use_regex=False)])` 表示先按我们的正则切块，再把每块转成字节；`use_regex=False` 是因为 ByteLevel 自带一个 GPT-2 的正则，不关掉会切两遍
- `Tokenizer.__init__`：三件事——关掉文本里的特殊 token 识别、查全 16 个特殊 token 都在、查 EOT 是 0。任何从文件加载的分词器都要过这一关
- `encode` 里 `add_special_tokens=False`：不自动在开头加 BOS 之类，什么时候插 EOT 由数据管线决定
- `train_bpe()`：`BpeTrainer` 的四个参数各有注释；最后是 2.5 节的大小检查

## 4. 怎么验证它是对的

```bash
python -m pytest tests/test_tokenizer.py -v
```

测试在 4 行小语料（英文 / 中文 / 代码 / 数学，各重复 50 次）上现场训一个 400 词表的分词器，几秒完成，不需要下载任何东西。

| 测试 | 证明了什么 |
|---|---|
| `test_roundtrip_is_lossless` | 300 段随机 Unicode（控制字符、汉字扩展 B、emoji、组合符号、阿拉伯文……）+ 边界情况（空串、纯换行、带 ZWJ 的 emoji 组合）编码再解码完全一致 |
| `test_special_token_ids` | 16 个特殊 token 恰好是 id 0–15，词表大小等于要求的大小 |
| `test_literal_special_text_is_not_special` | 文本里的 `<|endoftext|>`、`<|assistant|>` 编码不出任何特殊 id，且能原样解码 |
| `test_decode_shows_real_special_tokens` | 按 id 插入的 EOT 解码时显示为 `<|endoftext|>` |
| `test_digits_are_split_one_by_one` | 语料里 `12`、`156` 出现了 100 次，照样不合并 |
| `test_frequent_chinese_phrase_is_merged` | 高频的"你好"合并成 1 个 token（BPE 真的在起作用） |
| `test_decode_partial_utf8_does_not_crash` | 在一个字的字节中间截断，解码得到 `�` 而不是抛异常。生成时每一步都可能停在这种位置 |
| `test_ids_fit_uint16` | 所有 id < 65535，数据分片的 uint16 存得下 |
| `test_fingerprint_survives_save_and_load` | 存盘再读指纹不变；词表大小不同，指纹不同 |
| `test_training_is_deterministic` | 同样输入训两次，得到逐字节相同的分词器 |
| `test_training_refuses_to_undershoot_vocab` | 要 600 但语料只够 427 时报错 |

## 5. 下次怎么自己搭

1. **先让往返成立**：`models.BPE()` + ByteLevel 预处理 + ByteLevel 解码 + `initial_alphabet`，训练一个 300 的词表，写往返测试。这一步就已经是个能用的分词器
2. **加特殊 token**：`BpeTrainer(special_tokens=[...])` 按顺序占住前几个 id；写"字面字符串不是特殊 token"的测试——先看它**失败**，再加 `encode_special_tokens = True`
3. **换预切分正则**：先用 GPT-2 默认的，打印 `tok.encode("2026").tokens` 看数字怎么切；换成自己的正则再看一次
4. **加指纹和大小检查**
5. 用真实语料训 32768（P1-2）

调试时最有用的一行：

```python
enc = tok._tok.encode("你好，world 2026"); print(enc.tokens)
```

`tokens` 里的怪字符（`Ġ` 是空格、`ä½ł` 是"你"的 3 个字节）是 ByteLevel 把字节映射成可打印字符的结果，不是乱码。

### 故意改错练习

规矩见 [README](README.md)：先确认工作区干净，改完用 `git checkout -- 文件` 恢复。以下每条都实际跑过。

| # | 改哪里 | 跑 | 会看到 |
|---|---|---|---|
| 1 | `tokenizer.py`：删掉 `tok.encode_special_tokens = True` 这一行 | `pytest tests/test_tokenizer.py -x` | `test_literal_special_text_is_not_special` 失败：`'<|endoftext|>' 被编码出了特殊 token` |
| 2 | `tokenizer.py`：`r"\|\p{N}"` 改成 `r"\|\p{N}{1,3}"` | 同上 | `test_digits_are_split_one_by_one` 失败：`12 被切成了 ['12']` |
| 3 | `train_bpe.py`：删掉 `initial_alphabet=...` 这一行 | 同上 | 所有测试在准备阶段就报错：`只训练出 260 个 token，达不到要求的 400` |
| 4 | `train_bpe.py`：`if tok.get_vocab_size() != vocab_size:` 改成 `if False:` | 同上 | `test_training_refuses_to_undershoot_vocab` 失败：`DID NOT RAISE ValueError` |
| 5 | `tokenizer.py`：`decode` 里 `skip_special_tokens=False` 改成 `True` | 同上 | `test_decode_shows_real_special_tokens` 失败：`assert 'ab' == 'a<|endoftext|>b'` |
| 6 | `tokenizer.py`：`SPECIAL_TOKENS = [EOT, *ROLE_TOKENS, ...` 改成 `[*ROLE_TOKENS, EOT, ...` | 同上 | 准备阶段报错：`<|endoftext|> 的 id 必须是 0，实际是 4` |

**练习 3 值得多想一步**：这里报错的是**大小检查**，不是往返测试——因为少了字节，小语料连 400 都凑不够。但真实训练有几 GB 语料，合并次数足够填满 32768，大小检查会**通过**，这时只有往返测试能抓住它（2.1 节的 `'ebra  ��'`）。一个问题常常被离它最近的那道检查先拦下，拿掉那道检查它才会在别处露面。
