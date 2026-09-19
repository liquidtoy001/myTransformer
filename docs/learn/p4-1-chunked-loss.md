# P4-1 分块交叉熵

代码：`src/mytransformer/model/transformer.py`，`forward` 从 `h = self.norm(x)` 往下，以及 `_loss_sum`、`_loss_from_logits`
测试：`tests/test_loss.py`

---

## 1. 它解决什么问题

模型最后一步 `lm_head` 把每个位置的隐藏向量（1024 维）映射成词表上的分数（32768 维）。这个输出叫 logits，形状是 `(batch, seq, vocab)`，**非常大**：

```
micro_batch=16, seq=2048, vocab=32768
bf16 logits    16×2048×32768×2 字节 = 2 GiB
fp32 拷贝（算 softmax 前要升精度）       = 4 GiB
反向时的梯度、softmax 中间量              = 再几个 GiB
```

而 250M 模型的全部训练状态（权重 + 梯度 + Adam）才 3.5 GB。**loss 这一步的显存是模型本身的好几倍。**

**分块**：把 `batch×seq` 个位置切成若干块，一块一块地算 logits 和 loss，算完一块就扔掉这一块的 logits。

## 2. 怎么读代码

按执行顺序看 `forward` 的后半段：

**① 对齐预测位置和答案**

```python
h_pred, y = (h[:, :-1], targets[:, 1:]) if shift else (h, targets)
```

语言模型用位置 t 的隐藏状态预测位置 t+1 的 token。两种约定：

- `shift=True`（HF 约定）：`targets` 就是输入本身，这里错开一位。位置 T−1 没有答案，所以只有 T−1 个位置算 loss。
- `shift=False`（训练用）：加载器已经给好了 `x` 和 `y = x 右移一位`，T 个位置全部参与，一个都不浪费。

**② 数有效位置**

```python
n_valid = (y != -100).sum().clamp(min=1)
```

`-100` 是约定俗成的"这个位置不算 loss"（比如 SFT 时屏蔽掉问题部分，只对回答算 loss）。

**③ 分块 + 激活重算**（核心）

```python
total = sum(
    checkpoint(self._loss_sum, h_pred[i : i + ce_chunk], y[i : i + ce_chunk], z_loss, use_reentrant=False)
    for i in range(0, h_pred.size(0), ce_chunk)
)
```

关键是 `checkpoint`。PyTorch 做反向传播时，需要用到前向时的一些中间结果，所以前向时会把它们**存起来**。如果只是切块、不用 `checkpoint`：

| 方式 | 峰值显存（实测，vocab 32768，4×2048） | 原因 |
|---|---|---|
| 不分块 | 3.32 GiB | 完整的 bf16 logits、fp32 拷贝、softmax 输出、梯度同时存在 |
| 只切块 | 1.57 GiB | 省掉了同时存在的大张量，但**每块的 `log_softmax` 输出都要留到反向**，加起来还是一整份 fp32 logits（1.00 GiB） |
| **切块 + 重算** | **0.72 GiB** | 前向只存每块的输入 `h`（很小），反向时再把这一块的 logits 算一遍 |

**代价**：反向时多算一遍 lm_head 的前向。对 250M，每 token 多 `2·d·V = 67M` FLOPs，约占总计算量的 **3.6%**。用 3.6% 的时间换 4.6 倍的显存。

**④ 求和，最后统一除一次**

```python
total = F.cross_entropy(logits, y, ignore_index=-100, reduction="sum")   # _loss_from_logits 里
...
return {"logits": logits, "loss": total / n_valid, ...}                   # forward 最后
```

每块返回的是 **loss 之和**，不是均值。因为块的大小可能不一样（最后一块往往更小），块里被屏蔽的位置数也不一样。如果每块先求均值再平均，小块和大块的权重就变成一样了——结果是错的。

**⑤ 精度**

```python
if logits.dtype in (torch.float16, torch.bfloat16):
    logits = logits.float()
```

softmax 要先求指数再求和，bf16 精度不够，所以升到 fp32。但如果本来就是 fp64（测试里用 fp64 做精确比对），**不能降成 fp32**。这里原本写的是无条件 `.float()`，被测试抓出来改掉了。

**⑥ z-loss（可选）**

```python
lse = torch.logsumexp(logits, dim=-1)
total = total + z_loss * lse[y != -100].pow(2).sum()
```

惩罚 logits 整体漂得太大（`logsumexp` 就是 softmax 的分母取对数）。logits 越飘越大，bf16 就会溢出、训练就会崩。系数 1e-4，几乎不影响正常 loss。来自 PaLM。

## 3. 怎么验证它是对的

```bash
python -m pytest tests/test_loss.py -v
```

| 测试 | 证明了什么 |
|---|---|
| `test_chunked_ce_matches_plain` | 分块与不分块的 **loss 和每个参数的梯度** 在 fp64 下误差 < 1e-12。覆盖 12 种组合：shift 两种 × z_loss 两种 × 块大小三种（7 除不尽、3 刚好整除、1000 比总长还大）× 含一个 `-100` 位置。总位置数是 117（shift=True）或 120（shift=False） |
| `test_chunked_mode_does_not_return_logits` | 分块模式下不返回完整 logits（训练用不到，而它正是显存大头） |
| `test_shift_false_uses_every_position` | `shift=False` 与"手动算 logits 再交叉熵"一致 |
| `test_z_loss_penalizes_logit_drift` | z-loss 确实增大了 loss |
| `test_chunked_ce_reduces_peak_memory` | 峰值显存低于**一整份 fp32 logits**。这个判据能区分有没有做激活重算（只切块是 1.57 GiB，超过 1.00 GiB 这条线） |

**为什么"只比 loss"不够、还要比梯度**：训练靠的是梯度。如果分块写错了导致梯度没传回某个参数，loss 的数值可能照样对，但模型根本学不动。

**为什么用 fp64**：两种算法在数学上等价，但浮点加法的顺序不同，fp32 下会有 1e-6 左右的误差，分不清是 bug 还是舍入。用 fp64 可以把容差压到 1e-12，任何真 bug 都藏不住。

## 4. 下次怎么自己搭

**第 1 步：最朴素的版本**

```python
logits = lm_head(h)                                            # (B, T, V)
loss = F.cross_entropy(logits.reshape(-1, V).float(), y.reshape(-1))
```

用 `torch.cuda.max_memory_allocated()` 量一下峰值，记下来。

**第 2 步：改成"求和 ÷ 有效数"**

`reduction="sum"`，最后除以 `(y != -100).sum()`。和第 1 步的结果比一下，应该完全一样。

**第 3 步：切块，但先不加重算**

写个循环逐块算。再量一次峰值——**你会发现只降了一半**。停下来想想为什么（提示：反向需要什么？）。

**第 4 步：加上 `torch.utils.checkpoint`**

把"一块的 lm_head + loss"包成一个函数，用 `checkpoint(fn, h_chunk, y_chunk, use_reentrant=False)` 调用。再量一次，应该降到一整份 fp32 logits 以下。

**第 5 步：写等价测试**

fp64、随机数据、带 `-100`、块大小故意选一个除不尽的，比 loss 和所有参数的梯度。

### 练习：故意改错

| # | 改哪里 | 会失败的测试 | 说明 |
|---|---|---|---|
| L1 | 把 `checkpoint(self._loss_sum, h_pred[...], y[...], z_loss, use_reentrant=False)` 改成直接调用 `self._loss_sum(h_pred[...], y[...], z_loss)` | `test_chunked_ce_reduces_peak_memory` | `test_chunked_ce_matches_plain` **仍然通过**——结果完全正确，只是显存只降了一半（1.57 GiB），没到该有的水平。**测试全绿 ≠ 达到了目的** |
| L2 | `_loss_from_logits` 里 `reduction="sum"` 改成 `"mean"` | `test_chunked_ce_matches_plain`（块大小 7 和 3 的那几组）、`test_hf_parity.py::test_loss_matches_hf` | 块大小 1000 那组**仍然通过**：只有一块时，"每块求均值"和"整体求均值"没有区别。想想为什么 |
| L3 | 把 `if logits.dtype in (...)` 那两行换成无条件的 `logits = logits.float()` | `test_shift_false_uses_every_position` | 报错是 `Float did not match Double`——fp64 被悄悄降成了 fp32 |

改完用 `git checkout -- src/mytransformer/model/transformer.py` 恢复。
