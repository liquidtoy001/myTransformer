# P1-2 训练真正的分词器：抽样、训练、评测

代码：`scripts/sample_tokenizer_corpus.py`（抽样）、`scripts/tokenizer_report.py`（评测）
产物：`tokenizer/mt32k.json`、[reports/tokenizer.md](../../reports/tokenizer.md)
测试：`tests/test_tokenizer.py::test_committed_tokenizer_is_the_reported_one`

P1-1 讲的是分词器的**规则**。这一篇讲怎么给它喂数据，以及怎么判断训出来的东西好不好。结论和全部数字在报告里，这里讲方法和踩过的坑。

---

## 1. 它解决什么问题

BPE 的词表完全由训练语料决定：语料里什么多，什么就被合并成长 token。所以：

- **语料配比决定了哪种语言便宜。** 只用英文训练，中文每个字要 2.76 个 token，几乎是逐字节切（纯字节是 3.0）。第 5 节的练习 1 可以亲手验证
- **评测决定了你知不知道它好不好。** 不和别家比，"英文 1.45 tokens/word"这个数字本身说明不了任何事

## 2. 抽样：怎么从几百 GB 里拿 1 GB

### 2.1 只读需要的部分

parquet 文件内部按"行组"存储（这些数据集每组 1000 或 10000 行），每组里每一列单独压缩。所以可以：

1. 读文件末尾的元数据（几 KB），知道有几个行组、每组在文件的什么位置
2. 用 HTTP 范围请求只下载某几个行组的 `text` 列

```python
f = pq.ParquetFile(HfFileSystem().open("datasets/HuggingFaceFW/fineweb-edu/sample/10BT/000_00000.parquet"))
f.metadata.num_row_groups                      # 726
f.read_row_group(17, columns=["text"])         # 只下载第 17 组的 text 列
```

fineweb-edu 一个文件 2.15 GB，这样只读了其中几百 MB。DCLM 是 `.jsonl.zst`，zstd 压缩流不能从中间开始读，只能整个下载一个文件（141 MB）。

### 2.2 抽得均匀

同一个文件里的数据常常按来源或时间扎堆。所以每个来源分散在 3 个文件里，每个文件内部按**固定种子打乱**的顺序读行组，几个文件轮流读。固定种子保证重跑得到逐字节相同的结果，`manifest.json` 里的 sha256 可以核对。

### 2.3 评测文本不能是训练文本

每个来源的评测文本取自**训练没用过的文件**。拿训练过的文本测压缩率，就像拿做过的题考试，分数会偏高。

### 2.4 这次踩的坑

| 坑 | 怎么发现的 | 教训 |
|---|---|---|
| CCI3-HQ、The Stack 需要登录并同意条款 | 下载前用 HF API 查了 `gated` 字段 | 先查可访问性，再定方案 |
| SmolLM 语料里的 `python-edu` 只有 `blob_id`，没有代码 | 看了第一行数据 | 不要只看数据集名字，要看列 |
| `github-code-clean` 里 Go 的标签是 `"GO"`，我写的是 `"Go"`，Go 被全部过滤掉 | 代码 150 MB 只用 18 秒就抽完，快得可疑，查了语言分布 | **过滤条件写错不会报错，只会让数据悄悄变少**。抽完一定要看分布 |

第三个坑最有代表性：程序跑得"太顺利"本身就是一个信号。

## 3. 训练

```bash
python -m mytransformer.tokenizer.train_bpe --input "data/tokenizer_sample/train/*.txt" --out tokenizer/mt32k.json
```

1 GB 用时 5.7 分钟、约 2 GB 内存。HF tokenizers 的训练分两步：先把所有文本预切分并统计每个片段出现几次（大部分时间和内存花在这里），再在这张计数表上反复合并。

原计划抽 5 GB。1 GB 已经给每个类别几十到几百 MB，再多对 32k 词表的影响很小，就没有加。

## 4. 评测：怎么判断分词器好不好

### 4.1 用哪个指标

| 指标 | 优点 | 缺点 |
|---|---|---|
| **bytes/token** | 任何语言、任何分词器都能直接比 | 不直观 |
| tokens/word（英文） | 直观 | 中文没有空格，没法切词 |
| tokens/字（中文） | 直观 | 只适用于中文 |

报告两张表都给了。**真正做决定用的是 bytes/token**，因为训练预算按 token 算，而我们关心的是花同样的钱能学多少文本。

### 4.2 为什么是 32768：算力/字节

词表越大，压缩率越好（每个 token 装更多文本），但 lm_head 越大（每个 token 更贵）。两者相除：

```
FLOPs/字节 = FLOPs/token ÷ bytes/token
```

对 250M 模型，16k / 32k / 48k / 64k 分别是 1.035 / **1.000** / 1.011 / 1.035，32k 是谷底。这个结果**依赖模型大小**：模型越大，lm_head 在总计算量里占比越小，谷底就往大词表移。

`FLOPs/token` 直接调用 `ModelConfig.flops_per_token()`，和训练时算 MFU 用的是同一个函数。

### 4.3 读对比表的方法

不要只看 mt32k 一行，要**看它和谁接近、和谁差得远，再找原因**：

- 中文接近 Qwen2.5，比 GPT-4o 好 → 词表里 27% 是中文 token。用第 5 节末尾"看词表里有什么"的代码能翻到
- 英文比所有大词表都差 → 只有 2 万个英文位置
- zh_wiki 所有分词器都比 zh_web 差 → 猜测是繁体，数了 9 个常用字验证：繁体占 53%

每一个"为什么"都应该能用一小段代码验证，而不是停在猜测上。

## 5. 下次怎么自己搭 / 练习

**从零的步骤**：

1. 先从一个来源读 10 MB，训练一个 8k 分词器，写压缩率评测。整个流程跑通只要一分钟
2. 加第二个语言，看两种语言的压缩率怎么互相挤占
3. 加抽样的可复现性（固定种子、manifest、sha256）
4. 最后才放大到 1 GB

**练习**（1、2 实际跑过）：

1. **只用英文训练，看中文变成什么样。** 取 `en_web.txt` 前 30 MB，训练 8192 词表（约 4 秒），在评测集上测：en_web 1.62 tokens/word，**zh_web 2.76 tokens/字**，几乎是逐字节切
2. **复现 Go 被漏掉的 bug。** 把 `CODE_LANGS` 里的 `"GO"` 改成 `"Go"`，运行 `python scripts/sample_tokenizer_corpus.py --only code --scale 0.01`，再看 `manifest.json` 里 `code.train.lang_bytes`：没有 Go，也没有任何报错。**注意**：这会覆盖 `data/tokenizer_sample/train/code.txt`，做完要改回 `"GO"`，再不带 `--scale` 重跑一次 `--only code`（约 2 分钟）
3. **自己加一个对比分词器。** 从 HF 下载任意模型的 `tokenizer.json`，加到 `tokenizer_report.py` 的命令行参数里

### 看词表里有什么

```python
from tokenizers import Tokenizer, decoders
tok = Tokenizer.from_file("tokenizer/mt32k.json")
dec = decoders.ByteLevel()
for i in range(20000, 20020):
    print(i, repr(dec.decode([tok.id_to_token(i)])))
```

id 越大的 token 合并得越晚，也就越少见。翻到 30000 以后，能看到很多只在特定领域出现的词片段。
