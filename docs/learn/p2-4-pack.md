# P2-4 分词与打包：从文档到训练分片

代码：`src/mytransformer/data/pack.py`（约 230 行）；训练端：`trainer.py` 的 `_check_data`、`_log_eval`
端到端检验：`scripts/make_calib_dataset.py` + `configs/train/smoke_realtext.yaml`
测试：`tests/test_pack.py`、`tests/test_trainer.py` 末尾两个

---

## 1. 它解决什么问题

训练代码（P4 的 `ShardLoader`）只认一种东西：**一串 uint16 的 token id**，按顺序读，切成 2048 长的序列。它不知道文档、子集、配比。所以中间这一步要把"几个子集的一堆文档"变成"训练直接能读的分片"，并且做对四件事：

| 要做对的事 | 做错了会怎样 |
|---|---|
| 文档之间放分隔符 `<|endoftext|>` | 模型分不清一篇结束、另一篇开始，会把不相干的内容当成上下文 |
| **按配比混合好**再写分片 | 加载器按顺序读：如果一个分片全是中文、下一个全是代码，模型会连续几百步只看一种数据，loss 曲线大起大落 |
| 验证集**物理上**和训练集分开 | 验证文档混进训练，val loss 就是在测记忆，不是泛化 |
| 记下"这些分片是用哪个分词器切的" | 换了分词器还用旧分片，模型学到的是错位的 token，而且**不会报错** |

## 2. 两步，中间结果落盘

```
每个子集的文档 ──分词──> docs/<子集>/docs.bin + docs.idx
                              │
                    划验证集、按配比混合
                              ▼
               mix/train_00000.bin ...  mix/val_<子集>.bin  mix/meta.json
```

**第 1 步：分词**（`tokenize_docs`）。每篇文档编码后，前面加一个 EOT（id 0），首尾相接写进 `docs.bin`；`docs.idx` 记下每篇的起止位置。有了 idx 就能按编号随机读任意一篇（memmap，不用整个读进内存），第 2 步打乱、混合都靠它。

**分开两步的好处**：改配比、改验证集大小、改种子，只需重跑第 2 步（几秒），不用重新分词。

## 3. 第 2 步的三件事

### 3.1 划验证集

每个子集用固定种子打乱文档顺序，**排在最前面的文档划给验证集**，直到约 `val_tokens` 个 token。剩下的编号才进入训练池。之后混合时只从训练池里取，所以验证文档**不可能**出现在训练分片里。

`test_val_documents_never_appear_in_train` 是这样证明的：每篇测试文档带一个独一无二的标记 `<zh-17>`，把所有训练分片解码成文本，找出所有标记，和验证集的标记做交集，结果必须为空。而且故意让一个子集的文档不够用（被用了不止一轮），确认多轮时也不会混进来。

### 3.2 按配比混合

`Mixer` 每次选"已产出的 token 占比比目标落后最多"的子集，从它那里取下一篇：

```python
k = argmax(目标占比 × 已产出总 token − 各子集已产出 token)
```

这是**按比例的轮询**：不是先把英文全放完再放中文，而是交错着放，所以**任意一小段**里的比例都接近目标。测试用 7:3 的配比检查每连续 100 篇的比例，偏差不超过 ±0.02。

注意按 **token** 而不是按篇数算：代码文件平均 3000 个 token，维基条目平均 1200 个，按篇数轮询会让代码占比远超目标（练习 2）。

某个子集的文档不够它的配额时，**先把这一轮用完，再用新种子打乱，开始下一轮**。这样每篇文档被用的次数最多差 1，不会有的用了 5 次、有的一次没用。`meta.json` 里记下每个子集用了几轮（`epochs`）：一个子集被用了 4 轮以上，就要考虑降低它的配比了（重复太多会过拟合）。

### 3.3 写分片

按顺序把文档拼起来，攒够约 `shard_tokens` 个就在**文档边界**切一刀，写一个分片。所以每个分片都以 EOT 开头。

## 4. meta.json 和训练前的核对

```json
{"tokenizer": "0a87b9848bf6d598",
 "train": {"tokens": 10000798, "shards": [{"file": "train_00000.bin", "tokens": 2000621}, ...],
           "by_subset": {"zh_web": {"share": 0.208, "target_share": 0.208, "epochs": 0.81}, ...}},
 "val": {"zh_web": {"file": "val_zh_web.bin", "tokens": 104832, "docs": 73}, ...}}
```

训练配置里写上 `tokenizer: tokenizer/mt32k.json`，训练开始前 `_check_data` 会：

1. 分片的分词器指纹必须和配置的一致
2. 每个分片的实际大小必须等于 meta 里记的 token 数。**这条是为 P6 准备的**：几十 GB 的分片从对象存储拷到租的机器上，中途断了，文件就只有一半。不查的话，训练会在缺了一截的数据上照常跑，不报错
3. 分词器的词表放得进模型

### 分子集的验证 loss

验证目录里每个 `val_<子集>.bin` 单独算 loss，日志变成：

```
step    150  val_loss 7.1565  code 7.712  en_web 6.811  ...  zh_web 7.796  zh_wiki 7.377
```

`val_loss` 是各子集的**等权**平均，不让 token 多的子集主导。只有一个验证文件时（比如合成数据），输出和以前一样。

## 5. 端到端检验

```bash
python scripts/make_calib_dataset.py                                         # 约 10 秒
python -m mytransformer.train.pretrain --config configs/train/smoke_realtext.yaml   # 本机约 3.5 分钟
```

用 P2-1 取下来的 1.4 万篇校准文档（没有过滤去重，只为检验链路），打包成 1000 万 token，按 v1 配比混合：

| 子集 | 实际占比 | 目标 | 用了几轮 |
|---|---:|---:|---:|
| en_web | 0.416 | 0.417 | 1.87 |
| zh_web | 0.208 | 0.208 | 0.81 |
| code | 0.127 | 0.125 | 0.21 |
| zh_wiki | 0.031 | 0.031 | 0.69 |

ladder_s1（22M）在本机训 150 步，每个子集的验证 loss 都在降：

| step | code | en_web | math | zh_web | zh_wiki |
|---:|---:|---:|---:|---:|---:|
| 25 | 8.73 | 8.11 | 7.81 | 9.24 | 9.03 |
| 150 | 7.71 | 6.81 | 6.71 | 7.80 | 7.38 |

中文比英文高约 1。可能的原因是中文训练数据只占 24%，而且总共才训了 1000 万 token，离收敛还很远；具体原因要到 P5 用更多数据才看得清。**这就是分子集报 loss 的意义**：混成一个数就看不出这种差别。

## 6. 怎么读代码

**顺序**：`tokenize_docs` → `DocReader` → `split_val` → `Mixer.run` → `pack` → `check_meta`，最后看 `trainer.py` 的 `_check_data` 和 `_log_eval`。

## 7. 下次怎么自己搭

1. 先写 `tokenize_docs` + `DocReader`，测"编码再按编号读出来能解码回原文"
2. 写 `split_val`，测"验证和训练不重叠"
3. 写最简单的混合（按顺序拼接），**看它的问题**：打印每个分片里各子集的比例
4. 换成按比例轮询，测任意窗口的比例
5. 加 meta.json 和核对，最后接到训练

### 故意改错练习

规矩见 [README](README.md)。以下都实际跑过。

| # | 改哪里（`pack.py`） | 失败的测试 |
|---|---|---|
| 1 | `split_val` 最后 `return order[:cut], order[cut:]` 改成 `return order[:cut], order` | `test_split_val_is_disjoint_...`、`test_val_documents_never_appear_in_train` |
| 2 | `Mixer.run` 里 `self.emitted[k] += n` 改成 `+= 1`（按篇数算配比） | `test_mixing_is_smooth` |
| 3 | `tokenize_docs` 里 `[eot, *ids]` 改成 `ids`（不放分隔符） | `test_tokenize_docs_roundtrip`、`test_shards_start_at_document_boundaries_and_match_meta` |
| 4 | `check_meta` 里 `if want is not None and got != want:` 改成 `if False:` | `test_check_meta_catches_wrong_tokenizer_and_truncated_shard` |
