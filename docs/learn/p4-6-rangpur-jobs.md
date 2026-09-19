# P4-6 Rangpur 作业脚本

代码：`scripts/rangpur/`（`setup_env.sh`、`smoke.sbatch`、`train.sbatch`、`submit_chain.sh`）
说明：`docs/09-rangpur.md` §3.4、§四

---

## 1. 它解决什么问题

训练循环本身已经能"存档退出、再启动续跑"了。这一部分把它接到 Slurm 上，让它在共享集群上**无人值守地接力**：

- 每个作业最多 4 小时，时间到了会被杀
- 每人同时只能运行 1 个作业
- 你不在电脑前，作业也要一个接一个地跑下去，直到训完

## 2. 怎么读脚本

### `#SBATCH` 开头的行

它们不是注释，是写给 Slurm 的申请单，**在脚本运行之前**就被读取：

```bash
#SBATCH --partition=comp3710        # 去哪个分区排队（课程的 A100）
#SBATCH --account=comp3710          # 记在哪个账户上；不加会卡在 PartitionConfig
#SBATCH --gres=gpu:1                # 要 1 块 GPU；不加就拿不到 GPU
#SBATCH --time=04:00:00             # 最多跑多久，到时强杀
#SBATCH --signal=B:USR1@600         # 强杀前 600 秒，给批处理 shell 发 USR1
#SBATCH --output=logs/train_%j.out  # 日志写到哪；%j 替换成作业号
```

**没有 `--mem`**：课程指南说要加，但实测加了就排不上。脚本按实测来写。

### 信号是怎么送到 Python 的

```
Slurm ──USR1──▶ 批处理 shell（B:）
                  └─ exec python ...   ← python 取代了 shell，所以信号直接到 python
                       └─ _on_signal() 设标志 → 当前步做完 → 存档 → 退出
```

`exec` 的意思是"用这个程序**替换**当前的 shell 进程"，而不是在 shell 下面开一个子进程。所以 `B:` 发给 shell 的信号，其实就发给了 python。

### 两道保险

| 保险 | 什么时候起作用 |
|---|---|
| `--max-minutes 225` | 正常情况：训练 225 分钟后自己存档退出，离 4 小时还有 15 分钟 |
| `--signal=B:USR1@600` | 万一第一道没生效（比如某一步特别慢），被杀前 10 分钟收到信号 |

只靠其中一道都不稳。第二道还依赖 `exec`、依赖信号处理函数已经装好，任何一环出问题都会失效。

### 接力：`afterany`

```bash
prev=$(sbatch --parsable ... train.sbatch)
prev=$(sbatch --parsable --dependency=afterany:$prev ... train.sbatch)
```

`--parsable` 让 `sbatch` 只输出作业号，方便存进变量。`afterany:$prev` 表示"等 `$prev` 结束（不管怎么结束的）再开始"。

为什么是 `afterany` 而不是 `afterok`：作业到时被杀，Slurm 会把它记为失败。用 `afterok` 的话，第一个作业一被杀，后面的就全部取消了。

### 冒烟为什么要"等第一条训练记录"再发信号

```bash
for _ in $(seq 1 150); do
  grep -q '"type": "train"' "$RUN/metrics.jsonl" 2>/dev/null && break
  sleep 2
done
```

信号处理函数要等 `fit()` 开始时才装上。在这之前（导入 torch、构建模型、编译）收到 `SIGUSR1`，Python 会按操作系统的默认行为**直接退出**，什么都不存。等到第一条训练记录写出来，就能确定处理函数已经装好了。

### 冒烟为什么要明确检查"是续跑的"

只看"最后训完了"是不够的。假如信号来得太早、进程被直接杀掉，第 4 步会**从头开始训练**，照样能训完——冒烟看起来通过了，但续跑根本没被验证到。所以脚本最后检查两件事：第 3 步的日志里有 `停下（signal:SIGUSR1）`，第 4 步的日志里有 `续跑：`。

这和训练循环讲解里的教训是同一个：**要问"如果这里出错了，会不会有检查失败"**。

## 3. 怎么验证它是对的

**本地能做的只有语法检查**：

```bash
bash -n scripts/rangpur/smoke.sbatch
```

Slurm、`SIGUSR1`、`torch.compile`（需要 triton）在 Windows 上都没有，所以**第一次在 Rangpur 上跑冒烟，就是这一部分的测试**。

冒烟通过时，日志里应当依次看到：

```
=== Job 600xxx on a100-xx, ...
NVIDIA A100-...                          ← 拿到了 A100
TMPDIR=/scratch/600xxx                   ← A100 节点也有本地盘（§八 待确认的那一项）
=== [1/4] 测试
........  [100%]                         ← 全部测试通过
=== [3/4] 训练开始后发 SIGUSR1……
step     10  loss ...  MFU ...%          ← compile 能用，训练在跑
在 step xx 停下（signal:SIGUSR1）
✓ 收到 SIGUSR1 后存档退出
=== [4/4] 同一条命令再启动……
续跑：ckpt_000000xx.pt（step xx，……）
✓ 从 checkpoint 续跑，自检通过
训练完成：300 步
最终 val_loss 1.4x（本地是 1.421）
=== 冒烟通过
```

**val_loss 应该和本地的 1.421 接近，但不会完全相同**：硬件不同、`compile` 改变了计算顺序，而且中途在不同的步停过。差 0.01 以内都是正常的；如果差很多（比如 0.1），说明有问题。

## 4. 下次怎么自己搭

1. **先写能跑的最小作业**：只有 `#SBATCH` 头、激活环境、`nvidia-smi`。提交到测试分区，确认能排上、能拿到 GPU、日志写得出来
2. **把训练命令放进去**，时间设得很短，确认训练能在作业里跑
3. **加时间预算**：`--max-minutes` 设得比 `--time` 小，确认它会自己存档退出
4. **加信号**：`--signal=B:USR1@...` + `exec`。用 `kill -USR1` 手动发一次试试
5. **串起来**：写 `submit_chain`，用 `afterany`
6. **写冒烟**：把上面每一步要验证的东西放进一个短作业，**并且明确检查每一项**，而不是只看最后有没有报错

### 思考题

这些改动在本地没法实际运行验证，所以是思考题，不是"故意改错"练习。可以在冒烟通过之后，挑一两个到 `a100-test` 上真的试一下。

| # | 如果…… | 会怎样 |
|---|---|---|
| Q1 | 去掉 `train.sbatch` 里的 `exec` | 信号只到 bash，python 收不到。但第一道保险 `--max-minutes` 仍然会让它按时存档退出——所以**大多数时候你察觉不到**。这正是需要两道保险、而且要在冒烟里单独验证信号的原因 |
| Q2 | 冒烟里不等第一条训练记录，一启动就发信号 | 进程在装好处理函数之前就被杀掉，什么都不存。第 4 步从头训练并训完。**没有明确检查的话，冒烟会显示通过** |
| Q3 | `submit_chain.sh` 里 `afterany` 改成 `afterok` | 第一个作业到时被杀，被记为失败，后面排着的作业全部被取消。你第二天早上回来，发现只训了 4 小时 |
| Q4 | 提交前没有 `mkdir -p logs` | 作业瞬间失败，而且没有任何日志文件——因为日志本身就写不出来。`squeue` 里也看不到它了 |
| Q5 | `--max-minutes` 设成 240（和 `--time` 一样） | 算上启动和编译的时间，训练还没到 240 分钟作业就被杀了。只剩信号这一道保险 |
