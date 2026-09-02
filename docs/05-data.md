# 数据方案

**数据是小模型效果差异的最大来源。** 同样 500M 参数，好数据和差数据的下游分数能差 10 个点以上，远超架构调整的收益。这一章的工作量占整个项目的 1/3，值得。

---

## 一、目标配比（v1，9.44B tokens）

| 子集 | 占比 | tokens | 来源 |
|---|---|---|---|
| 英文网页（高质） | 40% | 3.78B | `HuggingFaceFW/fineweb-edu` |
| 英文网页（补充多样性） | 12% | 1.13B | `mlfoundations/dclm-baseline-1.0` |
| 中文网页 | 20% | 1.89B | `opencsg/chinese-fineweb-edu-v2` + `BAAI/CCI3-HQ` |
| 代码 | 12% | 1.13B | The Stack v2（Python/JS/Go/Rust/C++/SQL/Markdown） |
| 数学 | 6% | 0.57B | `HuggingFaceTB/finemath` + `open-web-math` |
| 百科/书籍 | 6% | 0.57B | Wikipedia zh+en、公版书 |
| 指令类（预训练期少量混入） | 4% | 0.38B | `smoltalk`、`Infinity-Instruct` 的纯文本化 |

**退火期（最后 0.94B tokens）改用**：数学 20% / 代码 20% / 教科书与高质长文 30% / 指令 20% / 通用网页 10%。这个阶段的数据质量对最终下游分数影响极大，值得单独准备。

配比不是拍脑袋——P5 的消融实验 A4 会对照 v1 与 v2（中文 35%），用结果决定最终版本。

> 数据量只有 9.44B tokens，**下载和处理的工作量比原计划小一个数量级**：FineWeb-Edu 和 CCI3-HQ 各取一小部分分片即可，本地就能跑完，不需要租 CPU 机器。

---

## 二、处理流水线

```
下载 → 语言识别 → 启发式质量过滤 → 质量分类器 → 去重 → 去污染 → 分词 → 打包分片
```

### 2.1 语言识别
`fastText` lid176，保留 zh/en 且置信度 > 0.65。中英混排文档单独归一类。

### 2.2 启发式过滤（Gopher 规则）
- 文档词数 50–100,000
- 平均词长 3–10 字符
- `#`、`...`、省略号结尾行占比 < 30%
- 停用词命中 ≥ 2 个（过滤非自然语言）
- 重复行/重复 n-gram 占比超阈值则丢弃
- 中文额外：非中文字符占比 > 60% 的"中文"文档丢弃；广告/导航模板匹配

### 2.3 质量分类器
FineWeb-Edu / CCI3-HQ 已经筛过，直接用。DCLM 部分自己跑一个 fastText 分类器（正例=Wikipedia+教科书，负例=随机 CC）。

### 2.4 去重
- **文档级**：MinHash-LSH，5-gram，128 permutation，Jaccard 阈值 0.8。用 `datatrove` 或 `text-dedup`
- **段落级**：对留下的文档做精确子串去重（suffix array 或简单的 hash 段落集合）
- 跨子集也要去重（Wikipedia 在网页语料里出现无数次）

> 去重是**性价比最高**的一步。重复数据不仅浪费算力，还会导致记忆化和 loss 曲线异常。

### 2.5 去污染（decontamination）
**必做，否则评测分数是假的。** 把所有评测集（HellaSwag / ARC / PIQA / WinoGrande / C-Eval / CMMLU / GSM8K / MMLU）的题面做 13-gram 索引，训练语料中命中的文档整篇剔除。记录剔除了多少条，写进报告。

### 2.6 分词与打包
```
输出：data/tokens/<subset>/shard_NNNNN.bin   # 纯 uint16 数组，无 header
      data/tokens/<subset>/meta.json          # n_tokens / tokenizer_hash / shard 列表
```
- 每 shard 100M tokens（≈200MB）
- 文档间插入 `<|endoftext|>`(id=0)
- 训练时跨文档拼接成定长 2048 序列，**不 padding**
- `tokenizer_hash` 必须写进 meta，换分词器时能立刻发现不匹配

---

## 三、验证集

每个子集留 1M tokens 作 val，**必须从训练分片中物理删除**（不是靠 index 跳过——那种做法在断点续训时容易出错）。

分子集报 val loss 而不是只报一个总数：中文 loss 和英文 loss 的变化趋势往往不同，混在一起看不出问题。

---

## 四、工程注意事项

| 坑 | 后果 | 做法 |
|---|---|---|
| 数据处理没有断点 | 机器挂了从头来（几十小时） | 每个 shard 独立处理，产出 `.done` 标记 |
| 单进程处理 | 慢 100 倍 | `multiprocessing` 按 shard 并行，或用 Modal/Ray 横向扩 |
| 中间产物不落盘 | 想改一个过滤规则要重跑全部 | 每个阶段的输出都落盘 |
| 没统计每步过滤率 | 报告里说不清数据从哪来 | 每阶段记录输入/输出文档数与 token 数 |
| 采样权重在训练时算 | 数据顺序不可复现 | 提前生成确定性的 shard 访问序列，存进 checkpoint |
| 只看统计不看数据 | 垃圾数据混进去没人知道 | **每个子集随机抽 200 条人工过目**，这一步不能省 |

---

## 五、产出

`reports/data.md` 至少包含：

1. 各子集的原始规模 → 每步过滤后规模 → 最终 token 数（一张漏斗表）
2. 去重去掉了多少（按子集分）
3. 去污染剔除了多少条、命中哪些评测集
4. 最终配比表 + 采样权重
5. 每子集随机 5 条样本原文（放附录，证明你真的看过）
6. 分词器在各子集上的 fertility
