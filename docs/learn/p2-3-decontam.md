# P2-3 去污染：把评测题从训练语料里剔除

代码：`src/mytransformer/data/decontam.py`（约 120 行）
评测集：`scripts/fetch_eval_sets.py`（11 个评测集，8.1 万道题，下载 25 MB）
校准：`scripts/calibrate_decontam.py`；通用片段列表：`configs/data/decontam_exclude.json`
测试：`tests/test_decontam.py`

---

## 1. 它解决什么问题

网页上到处是题库。HellaSwag 的上下文直接取自 WikiHow，MMLU、C-Eval 的题目在刷题网站上随处可见。模型在预训练时见过题目和答案，评测分数考的就是"背没背过"，不是能力。

[07-evaluation.md](../07-evaluation.md) 里写过："250M 模型在 MMLU 上超过 28%，先怀疑数据污染"。去污染就是让这句话有底气：我们确实把题目剔除了。

## 2. 基本做法：13-gram

GPT-3 论文的做法：每道题（题面 + 正确答案）切成连续 13 个词的片段，放进一个集合。训练文档只要包含其中任何一个，整篇剔除。

- **13 足够长**：两篇无关的文章碰巧有 13 个词完全相同，概率很小
- **又不会太长**：题目被转载时加了标点、改了大小写，归一化之后仍然能匹配（和去重用同一个 `normalize`）
- **存字符串，不存哈希**：索引里有 240 万个片段。用 32 位哈希的话，一篇几千词的文档里总有一个片段会碰巧撞上，误判率不可忽略

## 3. 中文：13 个字太短

第一版每个汉字算一个单位，中文网页 **9%**（2000 篇里 181 篇）被判为污染。看样本：几乎全是 C-Eval 题目里引用的**固定说法**：

- "全面建设社会主义现代化国家"：正好 13 个字
- "春蚕到死丝方尽，蜡炬成灰泪始干"、"山重水复疑无路"：名句
- 法律条文里的固定措辞

13 个汉字只相当于 7 个左右的英文词，比英文窗口短得多。改法：**汉字按半个单位算**，窗口约 26 个字，和英文的 13 个词相当。评测题和训练文档用同一个函数切，同一段文字从同一个位置切出的窗口完全相同，所以仍然是精确匹配。

中文网页的误判从 181 篇降到 15 篇。

## 4. 通用片段：用另一份语料来识别

剩下的命中里还有一类：**评测题引用了到处都有的文字**。

| 出现次数 | 片段 | 来自 |
|---:|---|---|
| 24 | "人民日益增长的美好生活需要和不平衡不充分的发展之间的矛盾" | C-Eval |
| 16 | "we hold these truths to be self evident that all men are created equal" | MMLU（美国历史题引用《独立宣言》） |
| 15 | "0 1 2 3 4 5 6 7 8 9 a b c" | HumanEval（十六进制字符表） |
| 6 | "be it enacted by the senate and house of representatives of the united states" | MMLU |

（"出现次数"是在 1 GB 普通语料里命中的次数）

一篇文章引用了《独立宣言》，不代表它见过这道 MMLU 题。做法：用 P1 的 1 GB 分词器语料当"普通语料"，数每个评测片段出现几次，**≥ 3 次**的当作通用片段，从索引里排除，共 220 个。

阈值的影响不大：

| 排除出现 ≥ k 次的 | 不排除 | ≥ 10 | ≥ 5 | **≥ 3** | ≥ 2 |
|---|---:|---:|---:|---:|---:|
| zh_web 剔除（/2000） | 15 | 13 | 12 | **11** | 7 |
| en_web 剔除（/2000） | 3 | 1 | 1 | **1** | 1 |

选 3：≥ 3 次的基本都是一眼能认出的名言、法条、固定表述。只出现 1–2 次的里面也有名言，但也可能是真的泄露到网上的题目，靠出现次数分不开，**宁可多剔除几篇**。多剔除的代价只是千分之几的文档。

**为什么用另一份语料来定规则**：校准文档（`data/filter_calib`）是用来检验效果的。用它来定规则、再用它来检验，等于拿考题复习再考同一张卷子。两份数据来自不同的文件。

## 5. 最终结果

| 来源 | 剔除 | 命中的是什么 |
|---|---:|---|
| en_web | 1/2000 | 一篇 WikiHow 文章，正是 HellaSwag 的出处（命中 23 个片段） |
| math | 2/2000 | 一道 MMLU 微积分题、一段 MBPP 的循环代码 |
| zh_web | 11/2000 | 消防考试资料（命中 49 个片段）、税法和会计题、时政题 |
| zh_wiki | 1/2000 | 最高法、最高检的司法解释 |
| 其余 | 0 | |

**已知的局限**：窗口约 26 个字之后，**C-Eval 27%、CMMLU 38% 的题比一个窗口还短**，没进索引，所以这些题查不出来；PIQA 测试集没有公开答案，只有一句目标，66% 太短。要抓短题，需要换一种机制（比如在文档里直接搜整道题的原文），不是把窗口调短：调短了误判就回到 9%。这一点要写进最终报告。

## 6. 怎么读代码

**顺序**：`units` → `ngrams` → `EvalIndex.add` → `contaminated` → `from_dir` 里的 `exclude_file`。

- `units`：先 `normalize`（小写、去标点、全角转半角），再用一个正则切：汉字一个一个切，其余按空白切
- `ngrams`：双指针。从每个位置 i 起，把 j 往后推，直到长度 ≥ 13（汉字算 0.5），取出 `us[i:j]`
- `contaminated` 和 P2-1 的过滤规则接口一致：返回原因（评测集名字），`None` 表示保留。P2-5 的管线可以把它和过滤规则串在一起

## 7. 怎么验证它是对的

```bash
pytest tests/test_decontam.py
python scripts/calibrate_decontam.py                   # 各来源剔除多少，打印命中的片段
python scripts/calibrate_decontam.py --scan-reference  # 重新生成通用片段列表（约 80 秒）
```

| 测试 | 证明了什么 |
|---|---|
| `test_units_split_han_chars_and_words` | 汉字逐个切，英文按词切，先归一化 |
| `test_window_length_counts_han_as_half` | 13 个词或 26 个字才构成一个窗口 |
| `test_repost_with_different_case_and_punctuation_is_caught` | 大小写、标点不同的转载也能抓到 |
| `test_chinese_question_in_a_quiz_page_is_caught` | 刷题页里的中文题 |
| `test_shared_short_phrase_is_not_contamination` | 只共享 12 个词不算 |
| `test_chinese_fixed_phrase_alone_is_not_contamination` | 只出现一句 13 个字的固定说法不算（第 3 节的主要误判） |
| `test_too_short_items_are_counted_not_indexed` | 短题记数但不进索引 |
| `test_excluded_ngrams_no_longer_match` | 通用片段排除后不再命中 |

## 8. 下次怎么自己搭

1. 先下载评测集，统一成"一行一道题"的格式。**看几条**，确认答案真的拼上了（CMMLU 第一次抽出 0 道题，就是因为 zip 里的路径和我以为的不一样）
2. 写最简单的版本：英文按词切、13-gram、集合查找
3. 在真实文档上跑，**看命中的片段**。中文的问题一看就知道
4. 修中文，再跑、再看
5. 用另一份语料统计通用片段，定阈值时列出不同阈值的结果

### 故意改错练习

规矩见 [README](README.md)。以下都实际跑过，练习 1、3 另外跑了校准脚本。

| # | 改哪里（`decontam.py`） | 单元测试 | 校准脚本 |
|---|---|---|---|
| 1 | `HAN_WEIGHT = 0.5` 改成 `1.0` | `test_window_length_...`、`test_chinese_fixed_phrase_...` 失败 | zh_web 剔除 11 → 181 篇 |
| 2 | `units` 里 `normalize(text)` 改成 `text.lower()` | `test_units_...`、`test_repost_...`、`test_excluded_...` 失败 | — |
| 3 | `add` 里 `grams = ngrams(units(text), self.n)` 改成 `grams = ngrams(units(text), self.n) or [" ".join(units(text))]`（把太短的题整道放进索引） | `test_too_short_...` 失败 | **没有任何变化** |

**练习 3 想说明什么**：看起来"把短题也放进索引"就能解决第 5 节的局限，其实完全没用。训练文档切出来的每个窗口都是固定长度（13 个词），而短题的整道题比这短，两者**永远不可能相等**。校准脚本的结果一个数都没变。改代码之前，先想清楚这个改动在机制上能不能起作用。
