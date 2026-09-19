# P4-5 训练循环

代码：`src/mytransformer/train/trainer.py`（主体）、`config.py`（配置）、`pretrain.py`（命令行入口）
测试：`tests/test_trainer.py`
冒烟：`configs/train/smoke_s1.yaml` + `scripts/make_synthetic_data.py`

---

## 1. 它解决什么问题

把前四部分（分块交叉熵、学习率调度、数据加载器、checkpoint）组装成一个能**无人值守**跑完的训练过程。难点不在"训练一步"——那只有十几行——而在于：

- **会被打断**：Rangpur 的作业到时就被杀，云上的竞价实例随时被收回
- **打断后要能无缝接上**：接上之后的每一步，都要和没被打断时一模一样
- **接错了要能发现**：权重加载错了、数据分片被换了，都要在第一步就报错，而不是悄悄训出一个错的模型
- **出了数值问题不能一路错下去**：一步 NaN 就可能把所有权重毁掉

## 2. 怎么读代码

按这个顺序读 `trainer.py`：

### ① `_train_step`：训练一步

```python
for _ in range(self.accum):
    x, y = self.loader.next_batch()
    with self._autocast():
        loss = self.fwd(x, y, shift=False, ce_chunk=..., z_loss=...)["loss"]
    (loss / self.accum).backward()
```

**梯度累积**：显存放不下一个完整的 batch，就把它拆成 `accum` 个小 batch（micro batch），逐个做前向和反向。`backward()` 会把梯度**加**到已有的梯度上，所以累积 k 次后，得到的是 k 个梯度之和。除以 `accum`，就变成了平均值——和一次性算一个 k 倍大的 batch 完全等价。

这就是跨硬件纪律第 1 条的实现基础：本地 `micro=2 × 累积 128`、Rangpur `micro=16 × 累积 16`，只要乘起来一样，训练就是同一个训练。

```python
grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
```

**梯度裁剪**：所有梯度拼成一个向量，如果它的长度超过 1.0，就等比缩放到 1.0。防止某一步梯度特别大，一下子把参数推得太远。注意它**返回的是裁剪前的长度**——这个数是训练要出问题的最灵敏的信号，要记进日志。

```python
if torch.isfinite(grad_norm):
    self.opt.step()
else:
    self.nonfinite_streak += 1   # 跳过这一步
```

**NaN 保护**：梯度里只要有一个 NaN，用它更新一次，所有权重都会变成 NaN，之后永远救不回来。所以检测到就跳过这一步。连续 10 次就存档停下，不再硬撑。

### ② `_fit`：主循环

每做完一步，依次检查：要不要记日志、要不要评测、要不要存档、要不要停。

**停下有三种原因**，都是先存档再退出：

| 原因 | 谁触发 |
|---|---|
| 训完了 | 走到 `max_steps` |
| 时间预算快用完 | `--max-minutes`；按平均每步时间留两步余量 |
| 收到信号 | SLURM 在杀作业前发 `USR1`（我们在 sbatch 里要求的）或 `TERM` |

**信号处理函数只设一个标志**（`_on_signal` → `request_stop`），真正的存档在当前这一步做完之后。因为信号可能在任何时刻到来——比如反向传播到一半，那时的梯度是不完整的，不能在那里存档。

`pretrain.py` 无论哪种退出都返回 0。sbatch 接力用的是 `afterany`（前一个作业不管成功失败都接着跑），不看退出码。

### ③ `_save` 和 `_resume`：存档与续跑

存档里除了模型权重，还有：优化器状态（Adam 的一阶、二阶矩）、step、已见 token 数、**加载器状态**、随机数状态、自检信息、元数据（git commit、完整配置）。

**少存任何一样，续跑都会不一样**：

- 没存优化器状态：Adam 的动量从零开始，接下来几百步的更新和原来不同（练习 T3）
- 没存加载器状态：从数据开头重新读，模型会把前面的数据再看一遍（练习 T2）

### ④ `_verify_resume`：续跑自检

加载完之后、训练第一步之前，做两个检查：

**模型输出对不对**：存档时，在一个固定的小 batch 上算过 loss 和 logits 的均方根，并记在存档里；恢复后重算一次，两个数都必须一致（相对误差 < 1e-5）。计算用 fp32 并关掉 TF32，同型号硬件上结果应当逐位相同。

**为什么要比均方根，只比 loss 不够**：训练初期模型接近均匀分布，logits 都很小。这时把权重整体改动 0.1%，loss 只变化 2e-6（低于容差，检测不到），而 logits 的均方根变化 1e-3——灵敏 500 倍。测试里专门覆盖了这种"只有均方根能抓到"的情况。

**数据流接不接得上**：存档时记下"下一个 batch 的前 16 个 token"，恢复后用加载器的 `peek()` 比对。数据分片被重新生成过、或者加载器状态没恢复，这里就会报错。

### ⑤ `init_from`：分叉

本 run 目录里没有任何 checkpoint 时，从 `init_from` 指定的另一个 run 的 checkpoint 起步。缩放律第 4 个点就用它：`configs/train/pretrain_250m_fork9000.yaml` 从主训练的 step 9000 起步，只把 `decay_start` 改成 9000，其余配置完全相同。

## 3. 怎么读训练日志

以本地冒烟为例（`ladder_s1` 22M 参数，合成马尔可夫数据，300 步）。数据的构造决定了 **loss 的理论下界是 ln 4 ≈ 1.386**（每个 token 恰好有 4 个等概率的下一个 token），加上文档分隔符带来的一点不可预测性，约为 **1.390**。

```
step     10  loss 8.8941  lr 6.67e-04  gnorm 19.356
step     20  loss 7.6733  lr 1.33e-03  gnorm 13.580
step     30  loss 9.8566  lr 2.00e-03  gnorm 71.308    ← ① 尖峰
step     40  loss 7.4683  lr 2.00e-03  gnorm 0.198
step     60  loss 6.9959  lr 2.00e-03  gnorm 2.807     ← ② 平台
已存档 ckpt_00000064.pt（time）                        ← 时间预算用完，存档退出
续跑：ckpt_00000064.pt（step 64，已见 0.004B tokens）  ← 重新启动，自检通过
step     70  loss 6.1704
step     80  loss 4.4772
step     90  loss 2.4985
step    100  loss 1.6727                               ← ③ 陡降
step    150  loss 1.5039
step    250  loss 1.4871                               ← ④ 学习率不变，降不下去了
step    280  loss 1.4749  lr 1.40e-03
step    300  loss 1.4379  lr 6.67e-05  val_loss 1.4214 ← ⑤ 衰减段又降一截
```

**① step 30 的尖峰**：warmup 正好在 step 30 结束，学习率刚升到峰值 2e-3，梯度范数跳到 71，loss 从 7.7 跳回 9.9。梯度裁剪把这一步的更新幅度限制住了，10 步之内就自己恢复。按故障处置手册，这属于"尖峰后自己回落"：记下位置就行。如果它不回落，就该回滚到上一个 checkpoint 并降低学习率。

**② 7.0–7.5 附近的平台**：只有 2048 种 token 会出现，如果模型只学会了"出现的总是这 2048 个"、还不会看上下文，loss 就是 ln 2048 ≈ 7.62。平台在这个值附近，说明模型此时基本还没利用上一个 token 的信息。

**③ step 70–100 的陡降**：从 6.2 一口气降到 1.7。模型"突然"学会了看上一个 token。这种先平台、再陡降的形状在语言模型训练中很常见：模型要先学到某种内部机制，loss 才会一下子下来。

**④ 学习率不变时的平台**：step 150 以后几乎不动，验证 loss 在 1.47–1.49，离下界（1.390）还差约 0.08。不是学不会了，而是学习率太大，参数在最优点附近来回跳。（比较时要用验证 loss：训练 loss 里含 z-loss 项，会高出约 0.017。）

**⑤ 衰减段**：最后 10%（step 270–300）学习率线性降到 0，验证 loss 从 1.474 降到 1.421，离下界只差 0.03。**这就是 WSD 最后一段衰减的价值**，也是为什么缩放律分叉必须真的做那 900 步衰减，而不能直接拿中间的 checkpoint 当数据点——中间 checkpoint 停在第 ④ 阶段。

**训练 loss（1.438）比验证 loss（1.421）高**：不是 bug。训练 loss 里含 z-loss 项，验证 loss 不含。在最终模型上把两者拆开算：交叉熵 1.4214 + z-loss 0.0170 = 1.4384，正好是训练日志里的数。

**MFU 18%**：对这么小的模型、没开 `torch.compile`、在 Windows 上，这个数是正常的。它**不能**拿来推算 A100 或 H100 上的速度（跨硬件纪律第 3 条）。

### 自己画曲线

日志全部在 `runs/<名字>/metrics.jsonl`，每行一条 JSON：

```python
import json
import matplotlib.pyplot as plt

recs = [json.loads(l) for l in open("runs/smoke_s1/metrics.jsonl", encoding="utf-8")]
tr = [r for r in recs if r["type"] == "train"]
ev = [r for r in recs if r["type"] == "eval"]
fig, (a, b) = plt.subplots(2, 1, sharex=True)
a.plot([r["step"] for r in tr], [r["loss"] for r in tr], label="train")
a.plot([r["step"] for r in ev], [r["val_loss"] for r in ev], "o-", label="val")
a.axhline(1.390, ls="--", c="gray", label="理论下界")
a.set_yscale("log"); a.legend()
b.plot([r["step"] for r in tr], [r["grad_norm"] for r in tr]); b.set_yscale("log"); b.set_ylabel("grad_norm")
plt.show()
```

如果一个作业在存档之前就被强行杀掉（比如 `SIGKILL`），从上一个 checkpoint 续跑后，中间那几步会被重新训练，`metrics.jsonl` 里就会出现重复的 step。画图时对每个 step 取最后一条记录即可。

## 4. 怎么验证它是对的

```bash
python -m pytest tests/test_trainer.py -v
```

全部在 CPU + fp32 上用极小的模型跑，几秒钟完成。CPU 上 fp32 的计算是确定性的，所以续跑可以要求**逐位相同**。

| 测试 | 证明了什么 |
|---|---|
| `test_resume_is_bitwise_identical_to_uninterrupted` | **核心**：一口气训 12 步，和"训到第 5 步存档退出、再启动续跑到 12 步"，每一步的 loss 和最终的每个权重都逐位相同 |
| `test_resume_detects_changed_weights` | 存档里的权重被改过，续跑时自检报错。两种幅度：0.01（loss 和均方根都能抓到）、0.001（**只有均方根能抓到**） |
| `test_resume_detects_changed_data` | 数据分片被换过，续跑时自检报错 |
| `test_finished_run_is_a_noop` | 已经训完的 run 再启动，一步都不多跑 |
| `test_time_budget_saves_and_exits` | 时间预算用完时存档退出，再启动能接着跑完 |
| `test_signal_saves_and_exits` | 收到信号后，当前步做完再存档退出 |
| `test_nonfinite_grad_skips_update` | 人为让第 3 步的 loss 变成 NaN：那一步权重不变，之后恢复正常，最终权重全部有限 |
| `test_grad_accum_matches_bigger_micro_batch` | micro 4 × 累积 2 与 micro 8 × 累积 1，每步 loss 相差在 1e-5 以内 |
| `test_fork_continues_from_another_run` | 从主训练 step 8 分叉：从第 9 步接着走、学习率从分叉点开始衰减、主训练的目录没被动过 |
| `test_repo_train_configs_are_valid` | 仓库里每个训练配置都能加载 |
| `test_config_rejects_unknown_keys` | 配置里拼错一个字段名，直接报错 |

最后一项加进来之前，正式训练用的 `pretrain_250m.yaml` 其实**根本加载不了**——它是在训练循环写好之前按文档手写的，字段名对不上。没有这个测试，要等到租了 8×H100 开机才会发现。

## 5. 下次怎么自己搭

1. **最小循环**：一个 for 循环，里面前向、反向、`opt.step()`、`zero_grad()`。先在单个 batch 上过拟合（loss 能降到接近 0），证明梯度是通的
2. **加梯度累积**，写"累积 2 次 = batch 翻倍"的等价测试
3. **加梯度裁剪和学习率调度**，把 loss、学习率、梯度范数记到 jsonl
4. **加存档和续跑**，然后**立刻**写逐位续跑测试。这个测试会逼你把所有状态都存全：先只存权重，看测试怎么失败；再加优化器；再加加载器……每加一样，失败的方式都会变
5. **加续跑自检**：先只比 loss，用一个很小的权重扰动试试能不能抓到——你会发现抓不到，然后再加均方根
6. **加退出机制**：时间预算、信号标志
7. **加 NaN 保护**，用 monkeypatch 注入一个 NaN 来测
8. **用合成数据在 GPU 上跑一次冒烟**，对照理论下界读 loss 曲线

### 练习：故意改错

每一个都实际验证过。

| # | 改哪里 | 会失败的测试 | 说明 |
|---|---|---|---|
| T1 | `(loss / self.accum).backward()` 改成 `loss.backward()` | `test_grad_accum_matches_bigger_micro_batch` | `test_resume_is_bitwise_identical...` **仍然通过**：两次运行带着同一个 bug，结果照样彼此一致。**一致性测试抓不到"一致的错误"** |
| T2 | 删掉 `_resume` 里的 `self.loader.load_state_dict(state["loader"])` | `test_resume_is_bitwise_identical...` | 报错是 `ResumeError: 续跑自检失败（数据流）`——是自检拦下的，不是 loss 对不上 |
| T3 | 删掉 `_resume` 里的 `self.opt.load_state_dict(state["optimizer"])` | `test_resume_is_bitwise_identical...` | 两个 `test_resume_detects_changed_weights` **仍然通过**：自检只核对权重和数据，**不核对优化器状态**。只有逐位续跑测试能发现 |
| T4 | `if torch.isfinite(grad_norm):` 改成 `if True:` | `test_nonfinite_grad_skips_update` | 一步 NaN，所有权重都变成 NaN |
| T5 | `for key in ("loss", "logit_rms"):` 改成 `for key in ("loss",):` | `test_resume_detects_changed_weights[0.001-logit_rms]` | 0.01 那组**仍然通过**：扰动大时 loss 自己就能抓到，扰动小时只有均方根可以 |

改完用 `git checkout -- src/mytransformer/train/trainer.py` 恢复。

T1 和 T3 是这一篇最想让你记住的：**测试全绿，只说明被测到的那些性质成立**。写测试时要问自己：如果我在这里犯一个错，哪个测试会失败？如果答案是"没有"，就还缺一个测试。
