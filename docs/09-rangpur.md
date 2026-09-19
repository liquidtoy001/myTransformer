# Rangpur 集群使用说明

本项目 P4 冒烟、P5 全部实验、P7、P8 都在学院的 Rangpur 集群（A100 40G）上跑；P6 正式训练租云上 8×H100（见 [04-compute.md](04-compute.md) §五）。这份文档记录 Rangpur 的硬约束、存储方案和作业模板。

以下事实来自实际使用（GAN 训练）、终端实测（2026-09-19）和课程指南 *COMP3710 Getting Started on Rangpur*。文中学号一律写作 `sXXXXXXX`。

---

## 一、硬约束

| 约束 | 实际情况 | 对本项目的影响 |
|---|---|---|
| 调度器 | SLURM（`sbatch` / `srun` / `squeue`） | 用 batch 作业，不用交互会话跑训练 |
| 每人同时运行的作业数 | **1 个**；其余排队显示 `QOSMaxJobsPerUserLimit`，前一个结束后自动开始 | 多个作业天然串行接力 |
| 每个节点的 GPU | **1 块**（`gpu:a100:1`） | **Rangpur 上做不了多卡 DDP**，放到云上 P6 学 |
| `comp3710` 分区单作业时限 | 12 小时 | 实际申请 **4 小时左右**（见 §五） |
| home 配额 | **16 GB**，NFS，登录节点和所有计算节点共享；实测已用 8.6 GB，**只剩 7.5 GB** | 存储是最紧的约束，见 §三 |
| 计算节点本地盘 | 作业内 `$TMPDIR=/scratch/<作业号>`，约 **197 GB**，只属于当前作业 | 当作业内工作区，**不能长期存数据** |
| login0 的 `/scratch` | 登录节点**自己的本地盘**（`/dev/sdb1`，80% 已用，所有人共用） | **不要用**，A100 作业看不到 |
| A100 总数 | 全班共用 **10 块** | 注意占用，见 §六 |

---

## 二、提交规则

| 规则 | 不遵守的后果 | 来源 |
|---|---|---|
| **GPU 作业必须加 `--account=comp3710`** | 一直卡在 `PartitionConfig` | 实测 |
| **不要加 `--mem`** | 作业一直排不上 | 实测 ⚠️ |
| 下载数据用 **`cpu` 分区的 batch 作业**，不在 login 节点上下载 | 占用共享的登录节点 | 实测 + 指南 §5 |
| 正式训练前先在 **`a100-test`** 分区做冒烟（上限 20 分钟） | 在正式分区排几小时队，结果一跑就报错 | 实测 |
| 安装包用 `pip install --no-cache-dir` | pip 缓存撑爆 home | 指南 §3.8 |
| 日志文件名带 `%j` | 重复提交互相覆盖 | 指南 §6 |
| 在脚本里激活环境 | batch 作业从干净环境启动，找不到 torch | 指南 §6 |

> ⚠️ **与官方指南不一致**：指南 §4.2 的示例写了 `--mem=16G`，§6 说"一定要显式设置 `--mem`"；但实际加了 `--mem` 作业就排不上。本文按**实测能跑通的写法**来。如果训练因主机内存不足被杀，再到 `a100-test` 上试一个较小的 `--mem` 值。
>
> 指南 §3.2 让你 `export TMPDIR=$HOME/tmp`——那只适用于安装 Miniconda 那一步。**训练和数据作业里不要这样设置**，否则临时文件会写进只有 16 GB 的 home，而系统本来就给了 197 GB 的本地盘。

---

## 三、存储

### 现状（2026-09-19）

| 目录 | 大小 | 处理 |
|---|---|---|
| `miniconda3`（含 torch 2.13.0+cu130 环境） | 4.5 GB | **复用**，与本地开发环境版本完全一致 |
| └ `pkgs` 包缓存 | 0.9 GB | `conda clean --all -y` 可安全清掉 |
| 课程作业目录（demo2） | 3.6 GB | demo2 已结束：按 §3.5 **备份到本地后删除** |
| `data` | 0.3 GB | 自行判断 |
| **可用** | **7.5 GB** → 清 conda 缓存后 8.4 GB → **删 demo2 后约 12 GB** | |

### 三层存储

| 位置 | 容量 | 寿命 | 放什么 |
|---|---|---|---|
| `$HOME` | 可用约 12 GB（清理 demo2 与 conda 缓存后） | 永久 | 代码、分词器、**P5 数据分片**、最新 checkpoint、编译缓存、指标日志 |
| `$TMPDIR`（计算节点本地） | ~197 GB | **作业结束即失效**，且下个作业可能在别的节点 | HF 下载缓存、原始 parquet、中间文件、wandb 本地文件 |
| 对象存储（云） | 按需 | 永久 | **P6 的 19 GB 数据分片**、历史 checkpoint |

### P5 阶段预算

| 项目 | 大小 |
|---|---|
| 数据（S3 需要 1.99B tokens，所有 P5 实验共用） | 4.0 GB |
| 最新 checkpoint + 写入时的临时副本（S3：1.2 GB × 2） | 2.4 GB |
| torch 编译缓存 | ~0.5 GB |
| **合计** | **~6.9 GB**；清理后可用约 12 GB，余量约 5 GB |

**写 checkpoint 要"先写临时文件、再改名替换"**，不能先删旧的再写新的：作业可能在写到一半时被杀，那样两份都没了。所以峰值是两份。历史 checkpoint 每次作业结束后同步回本地电脑，不留在 Rangpur。

**如果不清理 conda 缓存**：S3 改用 1B 不重复 token 训 2 轮（Muennighoff 等 [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)：同一批数据重复 ≤4 轮，loss 与全新数据几乎无差别）。数据降到 2 GB，峰值约 4.9 GB。

**P7 阶段**：删掉 P5 的数据，腾出空间放 250M 的完整 checkpoint（2.84 GB × 2）。

### 防止撑爆配额

所有作业开头都设置：

```bash
export HF_HOME="$TMPDIR/hf"                               # HF 下载缓存 → 本地盘
export HF_DATASETS_CACHE="$TMPDIR/hf/datasets"
export PIP_NO_CACHE_DIR=1                                 # 不留 pip 缓存
export TORCHINDUCTOR_CACHE_DIR="$HOME/myTransformer/.cache/inductor"  # 编译缓存留 home，跨作业复用
export WANDB_DIR="$TMPDIR/wandb"                          # wandb 在线同步，本地文件放临时盘
```

用 `datasets` 库读数据时**必须 `streaming=True`**，或者确保 `HF_HOME` 指向 `$TMPDIR`。默认会把整个原始 parquet 缓存到 `~/.cache/huggingface`，一下就超配额。

查 home 用量要在 CPU 作业里用 `du -xh --max-depth=1 ~ | sort -h`。`du -sh ~` 在 NFS 上会扫描 conda 环境里的大量小文件，非常慢。

### 3.4 环境隔离：不动课程的 `torch` 环境

`miniconda3/envs/torch` 后面的课程作业还要用，**不要往里装本项目的依赖**，免得版本冲突搞坏作业环境。在它上面叠一个轻量 venv，继承 torch，只额外装本项目需要的几个小包：

```bash
source ~/miniconda3/bin/activate && conda activate torch
python -m venv --system-site-packages ~/mt-venv      # 继承 torch 2.13.0+cu130，不重复安装
source ~/mt-venv/bin/activate
pip install --no-cache-dir -e ~/myTransformer[data,train]
```

`--system-site-packages` 让 venv 直接用课程环境里的 torch，所以 `mt-venv` 只有几百 MB。作业模板里只需 `source ~/mt-venv/bin/activate`。

**本项目在 Rangpur 上的所有文件只放两处**：`~/myTransformer`（代码、数据、checkpoint、编译缓存、日志）和 `~/mt-venv`。收尾时删这两个目录即可，课程环境不受影响（§七）。

### 3.5 开工前：备份并删除 demo2

tar 打成一个文件再传，比 `scp -r` 逐个传大量小文件快，而且能用校验和确认完整。指南 §1 说登录节点可以用来"moving data"，这一步不用开作业。

**1. 在 Rangpur 登录节点打包**（需要约 3.6 GB 临时空间，当前可用 7.5 GB，够）：

```bash
cd ~ && tar cf demo2_backup.tar COMP3710Rangpurfordemo2 && sha256sum demo2_backup.tar
```

**2. 在本地 PowerShell 下载并校验**：

```powershell
scp sXXXXXXX@rangpur.compute.eait.uq.edu.au:~/demo2_backup.tar E:\Documents\COMP3710\
```

```powershell
(Get-FileHash E:\Documents\COMP3710\demo2_backup.tar -Algorithm SHA256).Hash.ToLower()
```

两边的 SHA256 必须完全一致（Linux 输出小写，所以 PowerShell 这边转成小写再比）。

> ⚠️ 不要用 `ssh ... "tar cf - ..." > 文件` 这种流式写法：**Windows PowerShell 5.1 的 `>` 会把二进制流按文本重新编码**，文件会损坏且不报错。

**3. 校验一致后，在 Rangpur 上删除**：

```bash
rm -rf ~/COMP3710Rangpurfordemo2 ~/demo2_backup.tar && df -h ~
```

本地解包：`tar -xf E:\Documents\COMP3710\demo2_backup.tar -C E:\Documents\COMP3710\`（Windows 10 以后自带 `tar`）。

---

## 四、作业模板

> 训练入口 `mytransformer.train.pretrain` 在 P4 实现；以下模板随 P4 放进 `scripts/rangpur/`。

### 4.1 冒烟测试（`a100-test`，≤ 20 分钟）

```bash
#!/bin/bash
#SBATCH --job-name=mt-smoke
#SBATCH --partition=a100-test
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
echo "TMPDIR=$TMPDIR"; df -h "$TMPDIR"          # 第一次跑时确认 A100 节点也有本地盘

source "$HOME/mt-venv/bin/activate"
export HF_HOME="$TMPDIR/hf" PIP_NO_CACHE_DIR=1 TORCHINDUCTOR_CACHE_DIR="$HOME/myTransformer/.cache/inductor"

cd "$HOME/myTransformer"
python -m pytest tests/ -q                      # 先确认 GPU 环境下测试仍然全绿
python -m mytransformer.train.pretrain --config configs/train/smoke_s1.yaml --max-minutes 10
```

### 4.2 训练接力（`comp3710`，每个作业 4 小时）

```bash
#!/bin/bash
#SBATCH --job-name=mt-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600                     # 被杀前 10 分钟给 python 发 USR1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
source "$HOME/mt-venv/bin/activate"
export HF_HOME="$TMPDIR/hf" PIP_NO_CACHE_DIR=1
export TORCHINDUCTOR_CACHE_DIR="$HOME/myTransformer/.cache/inductor" WANDB_DIR="$TMPDIR/wandb"

cd "$HOME/myTransformer"
# exec：让 python 取代 bash 成为作业主进程，B:USR1 才能直接送到 python
exec python -m mytransformer.train.pretrain --config "$CONFIG" --max-minutes 225
```

训练脚本必须做到三件事：

1. **启动即续跑**：找到最新 checkpoint 就恢复（含 dataloader 位置与 RNG 状态）；已经训完就直接退出 0。这样多排几个作业也无害。
2. **两道保险退出**：收到 `USR1` 时存档退出；同时自己计时，到 `--max-minutes`（时限减去 15 分钟）也存档退出。只靠信号不可靠。
3. **续跑自检**：加载后用一个固定 batch 算 loss，和存档时记下的值对比，不一致就报错停下，防止"看着续上了、其实数据流错位了"。

### 4.3 接力提交

每人同时只能跑 1 个作业，所以直接排一串即可：

```bash
N=4                                             # 按需要的作业数调整
prev=$(sbatch --parsable --export=ALL,CONFIG=configs/train/ladder_s3.yaml scripts/rangpur/train.sbatch)
for i in $(seq 2 $N); do
  prev=$(sbatch --parsable --dependency=afterany:$prev --export=ALL,CONFIG=configs/train/ladder_s3.yaml scripts/rangpur/train.sbatch)
done
squeue --me
```

`afterany` 表示前一个无论成功还是被杀都接着跑——正是续跑需要的。

### 4.4 数据准备（`cpu` 分区）

```bash
#!/bin/bash
#SBATCH --job-name=mt-data
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/data_%j.out
#SBATCH --error=logs/data_%j.err

source "$HOME/mt-venv/bin/activate"
export HF_HOME="$TMPDIR/hf" PIP_NO_CACHE_DIR=1

cd "$HOME/myTransformer"
# 原始数据下载到 $TMPDIR（197 GB），分词后：
#   P5 数据 → 写入 $HOME/myTransformer/data/tokens/（约 4 GB）
#   P6 数据 → 逐分片上传到对象存储，不落 home（约 19 GB）
python -m mytransformer.data.build --out "$OUT" --shards "$SHARDS" --workdir "$TMPDIR"
```

**每个作业只处理若干个分片**，处理完一个就上传或写入一个，并留下 `.done` 标记。`$TMPDIR` 在作业结束后失效，所以不能指望下一个作业接着用上一个作业的中间文件。

---

## 五、作业时长

P5 在 Rangpur 上的纯训练时间约 14.7 小时（A100 MFU 按 35% 估），加开销约 17 GPU 小时。

**每个作业申请约 4 小时**，不申请满 12 小时：

- 指南 §5：*"A 5-minute job schedules far faster than a 10-hour one."* 申请时间越短，调度器越容易用空档先排进来。
- 作业短，GPU 释放得勤，同学的作业更容易插进来。
- 总 GPU 小时数不变，只是多重启几次——而续跑本来就要做对。

每个作业扣除约 15 分钟开销（启动、加载、编译、存档、安全余量），有效训练约 225 分钟。

---

## 六、共享资源礼仪

`comp3710` 是全班共用的 10 块 A100。本项目在 Rangpur 上总计约 **17 GPU 小时**，大致相当于跑几次课程作业的 GAN。仍然注意：

- **避开 demo 和作业截止前那几天**
- 调试用 `a100-test` 或本地 GPU，不在 `comp3710` 上试错
- 交互会话（`srun --pty`）用完立刻 `exit`
- 数据处理放 `cpu` 分区，不占 A100

P6 正式训练（约 47 GPU 小时的量）**刻意不放在 Rangpur**：没有助教的明确许可，这个量在共享的 10 块卡上太显眼。

---

## 七、收尾：给后续作业腾空间

项目结束后，按这个顺序处理：

| 步骤 | 内容 |
|---|---|
| 1. 带走成果 | 最终权重传 HuggingFace Hub；`reports/`、`logs/`、指标日志 `scp` 回本地（或 `git push`，它们本来就在仓库里） |
| 2. 确认带走了 | 本地能打开报告、能加载权重跑一次生成 |
| 3. 删除 | `rm -rf ~/myTransformer ~/mt-venv` |
| 4. 核对 | `df -h ~`，应回到开工前的水平 |

课程的 `miniconda3/envs/torch` 从头到尾没被改动，后续作业直接用。

**中途也要清理**：P5 做完、进入 P7 之前，删掉 P5 的数据（`~/myTransformer/data/tokens/`，4 GB）和缩放律模型的 checkpoint，只留汇总结果。P7 需要为 250M 的完整 checkpoint 腾出约 5.7 GB。

---

## 八、待确认

| 事项 | 怎么确认 |
|---|---|
| A100 节点是否也有 197 GB 的 `$TMPDIR` | 冒烟作业里的 `df -h "$TMPDIR"`（模板 4.1 已包含） |
| `$TMPDIR` 在作业结束后是否被清空 | 按"会清空"来设计，不依赖它 |
| `a100-test` 是否也需要 `--account=comp3710` | 模板里先加上；排不上再去掉试 |
| `cpu` 分区的时限与 CPU 核数上限 | `sinfo -p cpu -o "%P %l %c"` |
