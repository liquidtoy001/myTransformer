# P4-3 可续跑的数据加载器

代码：`src/mytransformer/data/shards.py`
测试：`tests/test_shards.py`

---

## 1. 它解决什么问题

三个要求，一个比一个难：

1. **数据太大，不能全读进内存。** 9.44B 个 token 存成 uint16 是 19 GB。
2. **续跑后要接着读，而且一个 batch 都不能错。** Rangpur 每个作业最多几个小时，训练要拆成很多段接力完成。如果续跑后数据流错位——重复读了一段、或者跳过一段——loss 曲线看起来可能还正常，但你已经不知道模型实际看了什么数据。
3. **多卡时，每张卡读不同的数据，不重不漏，且永远同步。**

## 2. 怎么读代码

### 数据格式：最简单的那种

每个分片（`.bin` 文件）就是一长串 uint16 数字，**没有文件头，没有分隔**。文档之间用一个特殊 token `<|endoftext|>` 隔开。

- 为什么是 uint16：词表 32768 < 65536，每个 token 2 字节刚好够
- 为什么不做 padding：训练时直接把文档首尾相接切成定长序列，一个位置都不浪费
- `np.memmap`：把文件"映射"成数组，读哪段才从磁盘加载哪段，19 GB 的文件也不占内存

### 从一段连续 token 切出 x 和 y

`_read` 的核心就这几行：

```python
start = offset + self.rank * bt                      # bt = B × T
buf = torch.from_numpy(data[start : start + bt + 1].astype(np.int64))
x = buf[:-1].view(self.B, self.T)
y = buf[1:].view(self.B, self.T)
```

一次读 **B×T+1** 个 token。去掉最后一个就是 `x`，去掉第一个就是 `y`，所以 `y` 的每个位置恰好是 `x` 对应位置的下一个 token。多读的那 1 个，是最后一行最后一个位置的答案。

### 状态只有三个整数

```python
@dataclass
class LoaderState:
    epoch: int = 0   # 第几轮
    shard: int = 0   # 本轮第几个分片（按洗牌后的顺序）
    offset: int = 0  # 分片里读到第几个 token
```

续跑时只要把这三个数存进 checkpoint、恢复时读回来即可。

### 两个设计决定

**① `_read` 是纯函数**：输入一个状态，返回（新状态, x, y），**不修改 `self.state`**。

好处是 `peek()`（看下一个 batch 但不前进）不需要任何额外代码，调用 `_read(self.state)` 然后丢掉新状态就行。续跑自检要用它：存档时记下"下一个 batch 是什么"，恢复后比对。

**② 读完立刻推进指针**（`_normalize`）

```python
return LoaderState(*self._normalize(epoch, shard, offset + bt * self.world)), x, y
```

读完一个 batch 后，如果当前分片剩下的不够再读一步，**马上**跳到下一个分片（必要时进入下一轮）。这样 `state` 永远指向"下一个 batch 真正从哪里读"。

最初的版本是"下次读的时候再跳"，数据流本身没错，但存进 checkpoint 的状态会指向一个已经读空的分片，`epoch` 也会滞后一步。这个问题被测试抓出来后改成了现在的写法（见练习 D1）。

### 多卡：大家共享同一个 offset

```python
start = offset + self.rank * bt
...
need = self.B * self.T * self.world + 1     # _normalize 里：按最后一个 rank 判断
```

rank 0 读 `offset` 开始的一段，rank 1 读紧接着的一段，以此类推，每步整体前进 `bt × world`。**判断分片够不够用时按最后一个 rank 算**，所以所有 rank 在同一步换分片，永远不会错位。

### 多轮与洗牌

```python
perm = np.random.default_rng(self.seed + epoch).permutation(len(self.paths))
```

每轮用 `seed + epoch` 洗牌分片顺序。**是确定性的**：同一个 epoch 永远得到同一个顺序，所以续跑不需要保存随机数状态。

## 3. 怎么验证它是对的

```bash
python -m pytest tests/test_shards.py -v
```

测试数据是递增序列（0, 1, 2, …），一眼能看出读的是哪一段。

| 测试 | 证明了什么 |
|---|---|
| `test_x_y_are_next_tokens` | y 是 x 的下一个 token，相邻行首尾衔接 |
| `test_peek_does_not_advance` | peek 看到的就是下一次 next_batch 拿到的 |
| `test_resume_reproduces_stream` | **核心**：读 9 个 batch 后保存状态，新加载器恢复后接下来 40 个 batch 与原加载器逐个相同，跨越多个分片和 epoch |
| `test_epoch_reshuffles_shard_order` | 每轮顺序不同，但每轮都完整走一遍所有分片 |
| `test_ranks_partition_the_stream` | 双卡时 rank r 第 s 步读到的 = 单卡第 2s+r 步读到的：不重不漏，且两个 rank 的状态始终一致 |
| `test_short_shards_are_skipped` | 比一步还短的分片被跳过；全都太短时报错 |
| `test_eval_batches_are_fixed` | 验证集每次取到的 batch 相同，val loss 才可比 |
| `test_write_shard_rejects_out_of_range` | 超出 uint16 的 token id 直接报错，而不是悄悄截断 |

## 4. 下次怎么自己搭

1. `np.array(...).astype(np.uint16).tofile(path)` 写一个分片；`np.memmap(path, dtype=np.uint16, mode="r")` 读回来
2. 从 memmap 里切 B×T+1 个 token，做出 x、y；用递增序列检查
3. 加 `offset`，每次前进 B×T；加"不够就换下一个分片"
4. 把状态抽成 dataclass，写 `state_dict` / `load_state_dict`，写续跑测试
5. 把读取改成纯函数，顺手得到 `peek`
6. 加 epoch 和洗牌；加 rank

### 练习：故意改错

| # | 改哪里 | 会失败的测试 | 说明 |
|---|---|---|---|
| D1 | `_read` 最后一行改成 `return LoaderState(epoch, shard, offset + bt * self.world), x, y`（读完不立即推进） | `test_epoch_reshuffles_shard_order` | `test_resume_reproduces_stream` **仍然通过**——数据流本身没错，错的是状态报告的位置。这正是最初那个 bug：一个测试只守一个性质 |
| D2 | `start = offset + self.rank * bt` 改成 `start = offset` | `test_ranks_partition_the_stream` | 所有卡读到同一段数据：等于把 batch 缩小了 world 倍，但你以为是正常大小 |

改完用 `git checkout -- src/mytransformer/data/shards.py` 恢复。
