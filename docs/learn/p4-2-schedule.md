# P4-2 WSD 学习率调度

代码：`src/mytransformer/train/schedule.py`（全文约 40 行）
测试：`tests/test_schedule.py`

---

## 1. 它解决什么问题

学习率（LR）决定每一步参数改多少。**整个训练过程中它不能是一个常数**：

- **开头要小**：刚初始化的模型梯度很乱，一上来就用大 LR 容易炸。所以先用几百步从很小线性升到峰值，叫 **warmup**。
- **结尾要降到很小**：LR 大的时候，参数在最优点附近来回跳，loss 降不到底。最后把 LR 降下来，参数才能"落"进去，loss 会明显再降一截。这叫**退火**（annealing）。

## 2. 为什么选 WSD 而不是 cosine

两种常见的形状：

```
cosine:  /‾‾‾‾\___           一直在降，形状由总步数决定
WSD:     /‾‾‾‾‾‾‾‾‾‾\        warmup → 稳定（恒定峰值）→ 最后 10% 线性降到 0
```

WSD 的好处：**稳定段 LR 恒定，所以可以在任意一步"分叉"出去单独做退火。**

本项目正好用到这一点：缩放律需要一个"250M 模型只训 20 tokens/param 就收尾"的数据点。有两种拿法：

- 专门再训一个 250M 模型到 4.73B tokens：约 $15
- 从主训练的 step 9021（正好 4.73B tokens）**分叉**，只跑 900 步的衰减：约 $1.5

用 cosine 就做不到——cosine 在 step 9021 时 LR 已经降了一截，分叉出来的模型和"训到 4.73B 就收尾"的模型不等价。

## 3. 怎么读代码

`__call__(step)` 分三段，从上往下读：

**warmup**

```python
if step < self.warmup_steps:
    return peak * (step + 1) / self.warmup_steps
```

为什么是 `step + 1`：如果写 `step`，第 0 步 LR 就是 0，这一步完全白跑。用 `step + 1`，第 0 步是 `peak/warmup`，最后一个 warmup 步正好到达峰值。

**越界保护**

```python
if step >= self.total_steps:
    return floor
```

看起来多余，其实很关键：没有这一行，衰减公式里的 `frac` 会超过 1，**LR 变成负数**——参数朝着让 loss 变大的方向走。

**稳定 + 衰减**

```python
if step < start:
    return peak
frac = (step - start) / max(1, self.total_steps - start)
return peak + (floor - peak) * frac
```

`frac` 从 0 走到 1，LR 从 `peak` 线性走到 `floor`。`max(1, ...)` 防止衰减段长度为 0 时除以零。

**分叉不需要特殊代码**：分叉就是另一个 `LRSchedule`，只是 `decay_start=9021, total_steps=9921`。在 9021 之前，两条调度在每一步都相同（测试保证了这一点），所以分叉出来的模型在分叉点之前和主训练完全一样。

## 4. 怎么验证它是对的

```bash
python -m pytest tests/test_schedule.py -v
```

| 测试 | 证明了什么 |
|---|---|
| `test_wsd_shape` | 四个阶段的关键点：第 0 步、warmup 最后一步、整个稳定段、衰减中点、终点、远远超过终点 |
| `test_wsd_is_monotone_in_decay` | 用主线的真实参数，衰减段 1800 步逐步不增 |
| `test_wsd_fork_matches_main_run_until_fork` | **分叉前 9021 步，分叉调度与主调度逐步相同**；分叉在 9921 降到 0，主调度此时仍在峰值 |
| `test_cosine_endpoints` | cosine 的起点、中点、终点 |
| `test_rejects_bad_config` | 未知类型、`decay_start` 早于 warmup 结束，都直接报错 |

**自己画出来看**，比看测试更直观：

```python
import matplotlib.pyplot as plt
from mytransformer.train.schedule import LRSchedule

main = LRSchedule(7e-4, warmup_steps=720, total_steps=18000, decay_start=16200)
fork = LRSchedule(7e-4, warmup_steps=720, total_steps=9921, decay_start=9021)
steps = range(18001)
plt.plot(steps, [main(s) for s in steps], label="主训练")
plt.plot(range(9922), [fork(s) for s in range(9922)], "--", label="缩放律分叉")
plt.legend(); plt.xlabel("step"); plt.ylabel("lr"); plt.show()
```

## 5. 下次怎么自己搭

1. 先写一个只有 warmup + 恒定的函数，打印前 10 步检查
2. 加衰减段；在衰减段取几个点手算对比
3. 加越界保护；试一下 `s(total_steps * 2)` 是不是 0 而不是负数
4. 包成 `dataclass(frozen=True)`：调度参数一旦定了就不该在训练中被改
5. 写分叉等价测试

### 练习：故意改错

| # | 改哪里 | 会失败的测试 | 说明 |
|---|---|---|---|
| S1 | 删掉 `if step >= self.total_steps: return floor` 这两行 | `test_wsd_shape` | 看报错里 `s(10000)` 的值——**是个负数** |
| S2 | warmup 里的 `(step + 1)` 改成 `step` | `test_wsd_shape` | 第 0 步 LR 变成 0 |

改完用 `git checkout -- src/mytransformer/train/schedule.py` 恢复。
