# P4-4 原子 checkpoint

代码：`src/mytransformer/train/checkpoint.py`（全文约 60 行）
测试：`tests/test_checkpoint.py`

---

## 1. 它解决什么问题

Rangpur 的作业到时间会被杀，云上的竞价实例随时会被收回。**被杀可能发生在写 checkpoint 的中途。**

最朴素的写法是直接 `torch.save(state, "ckpt_00002000.pt")`。如果写到一半被杀，磁盘上会留下一个**写了一半的文件**，文件名看起来完全正常。下次启动时，程序找"最新的 checkpoint"，找到的就是这个坏文件，加载时报错——**前面几个小时的训练等于白跑**，除非你手动去翻更早的 checkpoint。

## 2. 怎么读代码

### 原子写入：先写临时文件，再改名

```python
def save(run_dir, step, payload):
    final = path_for(run_dir, step)                  # ckpt_00002000.pt
    tmp = final.with_name(final.name + ".tmp")       # ckpt_00002000.pt.tmp
    torch.save(payload, tmp)
    os.replace(tmp, final)
```

关键在 `os.replace`：**在同一个文件系统里，改名是原子操作**——要么完全没发生，要么已经完成，不存在"改了一半"的状态。

- 在 `torch.save` 中途被杀：只留下一个 `.tmp`，正式文件名根本不存在，上一份 checkpoint 仍然是"最新"
- 在 `os.replace` 之后被杀：新 checkpoint 已经完整写好

所以任何时刻被杀，磁盘上带正式文件名的 checkpoint 都是完整的。

### 找最新的：只认正式文件名

```python
_NAME = re.compile(r"^ckpt_(\d{8})\.pt$")
```

`^` 和 `$` 表示整个文件名必须完全匹配。`ckpt_00003000.pt.tmp` 以 `ckpt_00003000.pt` 开头，但结尾多了 `.tmp`，所以**不匹配**。少了 `$`，这个半截文件就会被当成最新的 checkpoint（见练习 C2）。

step 用 8 位补零（`{step:08d}`），这样文件名按字符串排序和按数字排序结果一致。

### 清理：保留最近几份 + 里程碑

```python
def prune(run_dir, keep, keep_every=0):
```

- 只保留最近 `keep` 份（home 只有 16 GB，一份 250M 的 checkpoint 就 2.84 GB）
- step 是 `keep_every` 整数倍的永久保留（用来画"能力随训练的增长曲线"）
- 顺手删掉上次被杀留下的 `.tmp`

### 安全加载：`weights_only=True`

```python
torch.load(path, map_location=map_location, weights_only=True)
```

`torch.save` 底层用的是 Python 的 pickle，**加载任意 pickle 文件可以执行任意代码**。`weights_only=True`（PyTorch 2.6 起的默认值）只允许加载张量、dict、list、数字、字符串这些基本类型。

这反过来约束了 checkpoint 里能放什么：numpy 数组、自定义对象都不能放。所以数据加载器的状态设计成三个整数，洗牌用 `seed + epoch` 确定性生成，不需要保存 numpy 的随机数状态。

## 3. 怎么验证它是对的

```bash
python -m pytest tests/test_checkpoint.py -v
```

| 测试 | 证明了什么 |
|---|---|
| `test_atomic_save_leaves_no_tmp` | 正常保存后没有 `.tmp` 残留，能读回来 |
| `test_crash_during_save_keeps_previous` | **核心**：模拟写到一半被杀（把 `torch.save` 换成"写一半就抛异常"），上一份 checkpoint 仍然是最新且能加载 |
| `test_latest_ignores_partial_writes` | 有残留的 `.tmp` 时，`latest` 不会选中它 |
| `test_latest_on_missing_dir` | 目录不存在时返回 None（第一次启动的情况） |
| `test_prune_keeps_recent_and_milestones` | 保留最近 3 份 + 4000 的整数倍，删掉 `.tmp` |
| `test_payload_loads_with_weights_only` | RNG 状态、嵌套 dict、字符串在 `weights_only=True` 下都能加载 |

**关于 `test_crash_during_save_keeps_previous`**：这是最后补上的测试。最初没有它的时候，把 `save` 改回直接写正式文件名，**其余测试照样全部通过**。原子写入要防的是"中途被杀"，不主动模拟这个故障就测不出来。

它用了 pytest 的 `monkeypatch`：测试期间临时把 `torch.save` 换成一个假函数，测试结束后自动换回来。这是"注入故障"的标准做法。

## 4. 下次怎么自己搭

1. 先用最朴素的 `torch.save(state, final)`
2. **先写故障测试**：用 monkeypatch 模拟写一半被杀，确认朴素版本**会失败**
3. 改成 tmp + `os.replace`，确认测试通过
4. 写 `latest`：用正则只匹配正式文件名，按 step 排序
5. 写 `prune`
6. 用 `weights_only=True` 加载，把 payload 里所有东西都换成基本类型

第 2、3 步的顺序很重要：**先让测试在错误的实现上失败，再修到通过**。否则你没法确定这个测试真的在测你以为它在测的东西。

### 练习：故意改错

| # | 改哪里 | 会失败的测试 | 说明 |
|---|---|---|---|
| C1 | `save` 里的两行 `torch.save(payload, tmp)` + `os.replace(tmp, final)` 换成一行 `torch.save(payload, final)` | `test_crash_during_save_keeps_previous` | `test_atomic_save_leaves_no_tmp` **仍然通过**——正常情况下根本看不出区别 |
| C2 | 正则 `r"^ckpt_(\d{8})\.pt$"` 去掉结尾的 `$` | `test_latest_ignores_partial_writes` | 半截的 `.tmp` 被当成最新 checkpoint |

改完用 `git checkout -- src/mytransformer/train/checkpoint.py` 恢复。
