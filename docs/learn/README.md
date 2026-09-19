# 学习笔记

每完成一个部分就写一篇，回答四个问题：

1. **它解决什么问题**——不做会怎样
2. **怎么读代码**——按什么顺序看，关键几行为什么这么写
3. **怎么验证它是对的**——每个测试证明了什么
4. **下次怎么自己搭**——从最小版本开始的步骤，外加"故意改错"练习

## 目录

| 篇 | 部分 | 代码 |
|---|---|---|
| [p4-1](p4-1-chunked-loss.md) | 分块交叉熵 | `src/mytransformer/model/transformer.py` 的 `forward` 后半段 |
| [p4-2](p4-2-schedule.md) | WSD 学习率调度 | `src/mytransformer/train/schedule.py` |
| [p4-3](p4-3-loader.md) | 可续跑的数据加载器 | `src/mytransformer/data/shards.py` |
| [p4-4](p4-4-checkpoint.md) | 原子 checkpoint | `src/mytransformer/train/checkpoint.py` |
| [p4-5](p4-5-trainer.md) | 训练循环：续跑、自检、退出；**怎么读训练日志** | `src/mytransformer/train/trainer.py` |

## 怎么用这些笔记

**推荐顺序**：先读笔记的第 1、2 节 → 对着代码看一遍 → 跑第 3 节的测试 → 做第 4 节的练习 → 在一个空目录里不看答案重写一遍。

**"故意改错"练习的规矩**：

1. **先确认工作区干净**（`git status` 没有未提交的改动）。练习最后要用 `git checkout -- 文件` 恢复，它会丢掉这个文件里所有未提交的改动，包括你自己的。
2. 按笔记手动改一处代码
3. 跑笔记里给的测试，看它**怎么**失败——读报错信息，比"它失败了"本身更重要
4. `git checkout -- 那个文件` 恢复
5. 跑一遍全部测试，确认恢复干净：

```bash
python -m pytest tests/ -q
```

每个练习都实际验证过：改完之后，笔记里写"会失败"的测试确实会失败。有几个练习还特意列出了"改错之后**仍然通过**"的测试——它们说明一个测试只守护一个性质，测试全绿不代表代码全对。
