# P5-1 实验设计：先量噪声，再谈结论

代码：`src/mytransformer/train/muon.py`、模型的 `activation` / `qk_norm` 开关
配置：`configs/train/p5/`（12 个实验）、`configs/model/s2_*.yaml`
脚本：`scripts/rangpur/submit_p5.sh`（排队提交）、`scripts/analyze_p5.py`（出表）
测试：`tests/test_p5_configs.py`、`tests/test_muon.py`、`tests/test_analyze_p5.py`

这一篇是 P5 的准备工作：把"要做哪些对照、怎么判断结果是不是真的"定下来，代码和配置都写好，
真正的训练在 Rangpur 上跑（12 个实验、约 14 小时 A100）。

---

## 1. 这一步真正解决的问题

消融实验很容易做成自欺欺人：改一处、跑一次、看到 loss 低了 0.003，写进报告说"有效"。
问题是——**同样的配置换个随机种子，loss 本来就会差多少？** 不知道这个数，上面那句话没有任何意义。

所以 P5 的第一件事不是消融，是**用完全相同的配置跑 3 次，只换随机种子**，得到 val loss 的标准差 σ。
之后所有消融结果都拿 2σ 当尺子：差异小于 2σ 的，报告里只能写"看不出差别"。

这一步几乎没人做（跑 3 次的算力"浪费"在没有新结论的地方），但它决定了消融表里哪几行能下结论。

## 2. 对照实验的纪律：一次只改一处

12 个实验共用一个底座 `configs/train/p5/_base_s2.yaml`（ladder_s2，39M 参数，0.79B tokens），
每个消融只改一行。这条纪律由测试守着：

```python
diff = {k for k in base | other if base.get(k) != other.get(k)} - {"run_dir"}
assert diff == ALLOWED[name]     # a5_cosine 只允许 schedule 不同
```

配置写错是最贵的错误：跑完一小时才发现两组差了两个变量，结论直接作废。

### 公平对比要对齐参数量

| 消融 | 陷阱 | 处理 |
|---|---|---|
| A2 SwiGLU vs GeLU | SwiGLU 有 3 个矩阵、GeLU 只有 2 个。同样的 `ffn_hidden` 下 GeLU 少 1/3 参数，比的就成了参数量 | GeLU 的 `ffn_hidden` 放大到 1.5 倍（1408 → 2112），**非嵌入参数完全相等**（测试守着） |
| A3 GQA vs MHA | MHA 的 K、V 投影更大，必然多 13.9% 参数 | 对不齐，也不该对齐：这 13.9% 正是 GQA 省下的。所以这一组问的是"省下 14% 参数和 KV cache，loss 亏多少" |
| A6 QK-Norm | 多出的缩放参数只有几百个 | 可忽略（测试限定差异 < 0.1%） |

## 3. 三个新开关

### Muon（A1）

一句话：**先按动量算出更新量，再把这个更新矩阵"正交化"，然后才加到权重上。**

一个矩阵的梯度往往被少数几个方向主导（奇异值能差几个数量级），沿它更新等于只在那几个方向上走。
正交化把奇异值都拉到 1 附近，各方向走得一样多。

正交化用 Newton-Schulz 迭代做，**不求 SVD**：SVD 在 GPU 上慢，而且我们不需要精确的正交矩阵。
实测 5 步之后，原本相差 1000 倍的奇异值被压到 5 倍以内（落在 0.34–1.20），迭代 10 步会更接近 1。
**这是"够用就行"的典型例子**——测试里写的就是"差距缩到 5 倍以内"，而不是"等于 1"。

两个细节：

- **只有二维权重用 Muon**，嵌入和一维参数仍用 AdamW。嵌入的每一行是一个独立的词向量，对整个矩阵做正交化没有意义
- **两个优化器的学习率不是一回事**（Muon 通常大一个数量级：0.02 vs 2e-3），调度按同一个比例同时缩放两者

### GeLU（A2）、QK-Norm（A6）

QK-Norm 是对每个头的 q、k 做 RMSNorm，防住注意力打分变得过大（大模型 loss spike 的常见来源）。

**放在 RoPE 之前还是之后？** 我在代码注释里一度写成"两者等价"，动手一测才发现只对了一半：

- RoPE 是旋转，不改变模长 → 归一化里"除以 rms"那部分换序确实没区别
- 但 RMSNorm 还有一组**可学习的逐维权重**，它和旋转**不可交换**

初始化时权重全是 1，两种顺序输出完全一样（差 3.6e-07）；把权重换成随机数（模拟训练过），差异是 **3.0**。

这件事还有一个更重要的含义：**把实现里的顺序改掉，模型测试全绿**——因为测试用的是未训练的模型，
权重恰好全是 1。换序这类错误，未训练的模型测不出来。`test_qk_norm_before_rope_is_not_interchangeable`
就是把这个坑固定下来。

## 4. 缩放律怎么拟合

三个点（9.4M / 22.6M / 74.3M 非嵌入参数，各按 Chinchilla 20 tokens/参数训练）拟合：

```
L(N) = L∞ + A · N^(−α)
```

拟合方法故意用最笨的：**α 在网格上扫，每个 α 下 (L∞, A) 用线性最小二乘解**。三个点、两个线性参数，
不需要 scipy，也不会陷进局部最优。

验证方式仍然是"答案已知"：按 `L = 1.9 + 12·N^(−0.28)` 造三个点，拟合必须把这三个数找回来
（实测 1.8999 + 11.965·N^(−0.2798)）。

**三个点拟三个参数，残差必然接近 0，这不代表外推准。** 报告里会写明：P6 训完 250M 之后回来对照，
预测值和实际差多少——这一条比拟合本身更有价值。

## 5. 怎么跑

```bash
# 本机：检查配置
pytest tests/test_p5_configs.py tests/test_muon.py tests/test_analyze_p5.py

# Rangpur：数据上传见 docs/09-rangpur.md，然后排队提交
bash scripts/rangpur/submit_p5.sh            # 全部 12 个，约 14 小时
bash scripts/rangpur/submit_p5.sh ablation   # 只跑种子方差 + 消融，约 9 小时

# 结果拉回本机后出表
python scripts/analyze_p5.py --runs runs/p5
```

提交顺序是**先短后长**：ladder_s1（8 分钟）最先，能尽早发现配置错误；ladder_s3（4.4 小时）放最后。

## 6. 故意改错练习

规矩见 [README](README.md)。以下都实际跑过。

| # | 改哪里 | 结果 |
|---|---|---|
| 1 | `configs/model/s2_gelu.yaml` 的 `ffn_hidden` 改回 1408 | `test_gelu_and_qknorm_variants_keep_parameter_count` 失败：比的变成了参数量 |
| 2 | `configs/train/p5/a1_muon.yaml` 里再改一个字段（比如 `lr`） | `test_ablation_changes_exactly_one_thing[a1_muon.yaml]` 失败 |
| 3 | `muon.py` 的 `split_params` 里把 `is_embedding` 改成 `False`（嵌入也交给 Muon） | `test_split_params_keeps_embeddings_in_adamw` 失败 |
| 4 | `orthogonalize` 里去掉 `x = x / (x.norm() + eps)` | 三个测试失败：迭代不再收敛，正交化结果完全跑偏 |
| 5 | 把 `attention.py` 里的 QK-Norm 挪到 `apply_rope` **之后** | **全部测试通过**（见 §3）——直到你把权重改成非 1 |
