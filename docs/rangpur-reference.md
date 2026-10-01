# Rangpur 实测参考：供其他项目复用

整理日期：2026-10-01。证据来自 2026-09-19 至 2026-09-25 的终端记录、GAN 使用经验，以及本仓库的作业脚本和训练日志。**这是一份历史实测记录，不是管理员发布的当前资源承诺。** 本次整理没有连接集群；换账号、换课程或隔一段时间使用时，应重新查询分区和权限。

本文件独立于 Transformer 实验路线，可复制到其他项目。学号统一写作 `sXXXXXXX`；账户名 `comp3710` 是本课程实测值，不代表其他课程也有该权限。项目使用细节见 [09-rangpur.md](09-rangpur.md)。

## 1. 已知环境与证据边界

| 项目 | 实际观察 | 如何理解 |
|---|---|---|
| 登录入口 | `rangpur.compute.eait.uq.edu.au`，进入 `login0` | SSH 登录不等于已经获得计算资源 |
| 调度器 | Slurm；`srun`、`sbatch`、`squeue` 可用 | 安装、数据处理和训练进入计算作业 |
| CPU 节点 | 交互作业分配到 `vcpu-5` | CPU 分区适合环境配置、下载、哈希和解包 |
| GPU | 训练日志反复显示 `NVIDIA A100-PCIE-40GB, 40960 MiB` | 实测是 PCIe 版 A100 40 GB；不是 H100，也不是 A100 SXM |
| GPU 拓扑 | 当时 `sinfo` 显示每个 A100 节点 `gpu:a100:1` | 单节点只能申请一块 GPU |
| GPU 节点池 | 当时观察到约 10 个 A100 节点；训练曾落在 `a100-0/1/3/6/9` | 不保证当前在线数量或空闲数量 |
| 并发 | 此账号多个作业可排队，但同时运行受限；曾见 `QOSMaxJobsPerUserLimit` | 本项目实测不能并发运行多个 GPU 作业；不要推断所有账号的所有分区都相同 |
| 分区时限 | 当时 `comp3710` 为 12 小时；`a100-test` 为 20 分钟 | “只能不到两小时”不是此次查到的系统时限；额外课程约定仍应遵守 |
| home | NFS 共享目录，`df -h ~` 显示总量约 16G | 登录和计算节点可见，空间很紧；不把它视为备份服务 |
| 配额查询 | 登录欢迎信息报告使用量；`quota -s` 却显示 `none` | `none` 不能据此解释为无限空间 |
| 计算节点临时盘 | CPU 作业中 `TMPDIR=/scratch/600077`；A100 冒烟也有作业临时目录 | 节点本地工作区，下一次作业可能换节点 |
| 临时盘容量 | CPU 节点当时 200G 文件系统、剩 197G；A100 当时剩约 140G | 是当时文件系统的空闲量，**不是每人专属配额** |
| 登录节点 scratch | 登录时 `/scratch` 与 `/tmp` 在同一 200G 本地盘，当时剩 40G | 不是计算节点的 scratch，不能用它传递跨节点训练数据 |
| 已有 Python 环境 | `~/miniconda3/envs/torch`：Python 3.11.15，PyTorch `2.13.0+cu130` | 2026-09-19 的安装快照；新项目须检查依赖兼容性 |

单节点一块 GPU，加上该账号实测并发限制，使本项目没有在 Rangpur 验证多卡 DDP。不能据此宣称 Slurm 或整个集群永远不支持跨节点训练。

## 2. 工作放在哪里

| 位置 | 适合做的事 |
|---|---|
| 本地电脑 | 开发、分析指标、画图、保存备份，从本地发起 `scp` |
| 登录节点 | 编辑、Git、小文件操作、`mkdir`、`df`、提交和查询作业、看日志 |
| CPU 作业 | 安装依赖、下载数据、分词、扫描目录、几 GB 的 `tar` / SHA256 / 解包 |
| GPU 作业 | CUDA 检查、GPU 测试、训练、推理和评测 |

操作边界沿用课程指南和本项目的约定：不要在共享登录节点做重计算或大量文件读写。`scp` 仍会使用服务器网络和文件系统，安排大传输时也应考虑负载。`du` 扫描 NFS 上的 conda 小文件会很慢，应放 CPU 作业。

**登录节点：申请 CPU 交互作业。** 以下不带 account 的命令已实际成功；CPU 分区的 account 要求与 GPU 不必相同。

```bash
srun --partition=cpu --time=00:30:00 --pty bash
```

**分配完成后的 CPU 节点：确认位置和临时目录。**

```bash
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID TMPDIR=$TMPDIR"
df -h "$HOME" /scratch /tmp
```

任务完成运行 `exit` 释放交互资源。SSH 曾出现 `Connection reset`；`sbatch` 提交的批处理作业通常独立于 SSH 会话，重新登录后查询，不要因为断线就重复提交。交互作业不应作为长训练的持久运行方案。

## 3. 提交参数：哪些真的试过

- GPU 脚本使用 `--account=comp3710` 后成功运行；此前遗漏账户时曾卡在 `PartitionConfig`。这是本账号的排障经验，不是该状态的唯一原因。
- 本项目成功的 GPU 脚本不设置 `--mem`。此前设置该参数时曾长时间排不上，而课程 PDF 示例要求设置内存。**原因未查明**，不能推断参数永远不能用，或忽略程序的 CPU 内存需求。
- `a100-test` 加 `--account=comp3710` 成功；不加的情形未验证。
- 日志路径的父目录必须在提交前存在；Slurm 不会替你创建 `logs/`。
- 每个日志名包含 `%j`，避免不同作业互相覆盖。
- 作业中显式激活环境，并记录 GPU、Git 提交和配置；不要依赖登录 shell 已激活的状态。

**登录节点：重新查询可见资源和自己的作业。**

```bash
sinfo -o "%P %a %l %D %G"
squeue --me
```

可见分区不等于账号有权使用。当前 CPU 核数、内存上限、QOS、并发和存储保留策略，尚未在这份记录中确认。

## 4. 通用 GPU 检查脚本

这是基于已成功参数整理的模板，**新模板本身尚未提交到集群验证**。在项目根目录保存为 `gpu_check.sbatch`。已有环境必须位于 `~/miniconda3/envs/torch`；不符合时改成自己的环境激活路径。

```bash
#!/bin/bash
#SBATCH --job-name=gpu-check
#SBATCH --partition=a100-test
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpu_check_%j.out
#SBATCH --error=logs/gpu_check_%j.out
set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export PYTHONUNBUFFERED=1
echo "Job=$SLURM_JOB_ID host=$(hostname) TMPDIR=${TMPDIR:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import sys
import torch
print("Python:", sys.version)
print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(512, 512, device="cuda")
y = x @ x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all().item(), "Non-finite GPU output"
print("GPU matrix multiplication passed")
PY
```

**登录节点，项目根目录：**

```bash
mkdir -p logs
sbatch gpu_check.sbatch
squeue --me
```

日志在作业启动后才出现；排队时 `tail` 找不到文件不代表提交失败。查结束状态可用以下命令（Slurm 标准查询方式，本次整理未远程执行）：

```bash
sacct -u "$USER" --starttime today --format=JobID,JobName,State,ExitCode,Elapsed
```

`squeue` 为空只说明没有当前可见的排队或运行作业，不能说明历史作业成功。结合 `sacct`、退出码和日志确认。`Priority`、`Resources` 通常是等待优先级/资源；`Dependency` 是前序作业条件未满足，不能仅因这些状态就修改参数。

## 5. 环境复用与隔离

本项目使用课程 conda 环境提供的 Python/PyTorch，再创建项目专属 venv：`--system-site-packages` 继承基础依赖，额外包安装到 venv。Transformer 的 `~/mt-venv` 当时约 224M，而 miniconda 约 4.5G，避免重复装一套 GPU 库很有价值。

**CPU 作业内：示例创建其他项目的环境。**

```bash
source "$HOME/miniconda3/bin/activate"
conda activate torch
python -m venv --system-site-packages "$HOME/project-venv"
source "$HOME/project-venv/bin/activate"
python -m pip --version
```

安装依赖使用 `python -m pip install --no-cache-dir ...`；具体依赖取自项目锁文件。该方案不是完全隔离：基础环境改变会影响 venv，项目包也可能遮蔽继承的版本。安装后记录实际 `sys.executable`、`torch.__version__` 和 `torch.__file__`，并检查 `python -m pip check`。不要假定其他项目必须升级或降级课程 torch。

`setup_env.sh` 在子进程内激活 conda 不会改变父 shell。因此它运行成功后，当前 shell 直接敲 `conda clean` 仍可能显示 `conda: command not found`；需要先执行上述 `source`，不是安装失败。

## 6. 存储规划：先算峰值

home 保存代码、必要数据、可恢复 checkpoint 和小日志；临时目录保存可重建缓存和中间文件。**不要依赖 `$TMPDIR` 在作业结束后仍存在**：实际清理时间尚未验证，按不可持久设计。提交前检查 home，而不是套用历史“还剩 12G”。

**登录节点，轻量查询：**

```bash
df -h "$HOME"
```

**CPU 作业，扫描目录：**

```bash
du -xh --max-depth=1 "$HOME" 2>/dev/null | sort -h
```

两种容易漏算的峰值：

1. 解包时，home 同时放 tar 和解包结果，通常接近两份数据；上传前就要算清楚。
2. 原子 checkpoint 保存需要新临时文件与已有存档同时存在；保留多个历史文件时，峰值可能超过两份。

本项目完成的 run 只保留权重 `final.pt`，中断的 run 保留模型、优化器、随机数和数据位置等恢复状态。**只保存权重不等于能精确续训。** 删除前先下载、核对校验和、加载验证；不提供“一键删除整个 home”之类的命令。

**CPU/GPU 作业内：把可重建缓存放临时盘。**

```bash
: "${TMPDIR:?需要计算作业提供的临时目录}"
export HF_HOME="$TMPDIR/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PIP_NO_CACHE_DIR=1
export WANDB_DIR="$TMPDIR/wandb"
mkdir -p "$HF_HOME" "$WANDB_DIR"
```

离线 W&B 日志放临时盘时必须在退出前带走；在线模式也应确认同步完成。训练权重和未上传成果不能只留在这里。编译缓存可放项目 home 目录跨作业复用，但会占配额，需要单独计量。

## 7. Windows 上传与完整性校验

Windows PowerShell 5 不支持 `&&`，也通常没有 `sha256sum`。使用 `Get-FileHash`；Linux 输出通常小写，Windows 通常大写，比较时忽略大小写。

**本地 PowerShell：以本项目已有文件为真实示例。**

```powershell
Set-Location E:\Documents\transformer\data
Get-FileHash .\p5_data.tar -Algorithm SHA256
```

`scp` 在本地运行；使用你自己的账号和真实项目路径。不要把 `<作业号>` 的尖括号照抄进文件名。远程 `train_614614.out` 是文件名，`train_<614614>.out` 是另一个不存在的文件名。

**Rangpur CPU 作业：本项目的校验/解包示例。**

```bash
cd "$HOME/myTransformer/data"
sha256sum p5_data.tar
```

先与本地哈希核对，完全一致后才解包：

```bash
tar xf p5_data.tar
```

本项目传输曾得到 SHA256 `33514aab1e620fb0fbcb3e7cffed605c1576c73dc5a3562dae00cf7d3f6f63e8`。它仅对应当时那个包，**不是其他项目的预期哈希**。

不要在 PowerShell 5 用 `ssh ... "tar cf - ..." > backup.tar` 保存二进制流，文本重定向可能破坏内容。使用普通文件传输。Windows 编写的 `.sh` / `.sbatch` 应使用 LF；Git 可设置 `*.sh text eol=lf` 和 `*.sbatch text eol=lf`。

## 8. 断点接力：已经验证了什么

本项目有主动墙钟存档，以及 Slurm 提前信号两道机制：

```bash
#SBATCH --signal=B:USR1@600
```

`B:` 给 batch shell 发信号；脚本最后使用 `exec python ...`，让 Python 替代该 shell。否则需要另外实现 shell 转发。Python 的 signal handler 只设退出标志，训练循环在安全边界保存，避免在处理函数中做大型 I/O。

真实结果：早期 smoke 最终训练成功但没有满足信号测试，脚本正确判为失败；修复后 smoke 600202 通过。后续 s3 诊断在 step 2429 明确因 `SIGUSR1` 保存退出，最终续跑到 5672 步完成。另一次 s1 在 step 303 因主动时间预算退出，随后续跑到 360。**时间退出和信号退出是两种不同验证。**

训练器在首次编译训练步后重新安装信号处理并解除相关屏蔽后，信号测试恢复正常。哪一个底层组件造成旧行为，尚未定位，不把“某库在 C 层替换 handler”写成已证实根因。其他项目必须测试自己的编译器、进程树和恢复流程。

队列接力可用 `afterany`，但前一个失败也会继续启动下一个；它不是成功保证。依赖成功产物的数据流水线一般适合 `afterok`。训练接力用 `afterany` 时，程序必须验证 checkpoint、识别已完成状态，并在不可恢复错误后取消剩余链，避免反复失败占资源。

## 9. 实测速度与估时方法

| 本项目负载 | 稳态观察 | 适用范围 |
|---|---|---|
| s1，22M 参数，524,288 tokens/step | 约 2.84 秒/step，184k tok/s，MFU 约 11% | 此模型、序列长度和软件版本 |
| s2，优化 z-loss 后，524,288 tokens/step | 约 2.53 秒/step，207k tok/s，MFU 约 22% | 不能直接用于其他模型 |
| s3，99.5M 参数，262,144 tokens/step | 约 2.17 秒/step，121k tok/s，MFU 约 32% | 此诊断配置 |

首次编译、评测、NFS 存档会增加耗时；排队时间不包含在纯训练时长中。MFU 来自本项目 FLOPs 口径，不能与任意其他实现直接比较。

新项目先做小规模冒烟，再测稳态每步耗时。申请时长 = 剩余步数 × 实测步耗时 + 编译/加载 + 评测/存档 + 安全余量。早期用假设 25% MFU 估时，实际 s1 约 11%，导致作业未在预算内训完；这是真实发生的估时错误。

换到另一块同型号 GPU 不会自动改变模型架构，但软件、内核、随机数、数据顺序和数值非确定性可能影响结果。恢复时校验状态；速度比较记录节点、GPU、配置和 Git 版本，不能把不同配置的速度差直接归因于卡。

## 10. 其他项目开始前的最小核对表

- 确認课程授权范围、account、分区和当前时限；本项目的成功提交不等于所有用途都获许可。
- 查 home 剩余空间，算上解包、优化器和 checkpoint 临时副本；保留后续课程作业空间。
- CPU 作业内配置环境，GPU 测试分区验证设备与最小真实计算。
- 测试真实的信号存档、续跑和最终产物，不仅看进程有没有报错。
- 记录训练版本与实测步耗时，再排正式作业；共享资源高峰期避免长时间占用。
- 成果下载、校验并实际加载后，再清理项目目录；保留课程基础环境。

## 来源与维护

历史证据：本对话中的 Slurm/磁盘终端记录、本仓库 [train.sbatch](../scripts/rangpur/train.sbatch)、[setup_env.sh](../scripts/rangpur/setup_env.sh)、[实验日志摘要](../reports/journal.md)。课程材料为本地 `COMP3710-Rangpur.pdf`；本次未重新读取 PDF，涉及指南的操作边界沿用既有项目记录。

官方入口：[EAIT batch compute](https://student.eait.uq.edu.au/infrastructure/compute/)。链接供后续核对；本次未联网验证当前页面或政策。更新本文时附上日期、节点、命令和实际输出，把“观察”“推测”“建议”分开。
