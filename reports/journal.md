# 训练日志

按天记录。这是技术报告第 5、10 节的原始素材，训练期间随手记，不要事后补。

格式见 [docs/08-report-template.md](../docs/08-report-template.md)。

## 2026-09-19 · P1 分词器
- P1-1：写测试时发现 HF BpeTrainer 在语料不够时会默默少给词表（要 600 给 427），train_bpe 加了大小检查
- P1-1：HF tokenizers 默认把文本里的 "<|endoftext|>" 字样识别成 id 0，设 encode_special_tokens=True 关掉
- P1-2：规划里的 BAAI/CCI3-HQ、The Stack 系列都要登录同意条款；python-edu 只有 blob id 没有代码内容。换成 chinese-fineweb-edu-v2、github-code-clean
- P1-2：github-code-clean 里 Go 的标签是 "GO"，第一次抽样漏掉了 Go，重抽
- P1-2：1 GB 语料训 32k 用时 5.7 分钟、约 2 GB 内存。中文 0.70 tokens/字（≈ Qwen2.5），英文 1.45 tokens/word（比大词表差 7–11%）
- P1-2：词表扫描 16k/32k/48k/64k 按 250M 模型的 FLOPs/字节算，32k 正好是谷底；数字一位一切多花 2.2%
- P1-2：zh_wiki 一半是繁体，所有分词器在它上面都更差；P2 决定是否转简体
- 路线图英文验收 1.35 做不到（64k 也只有 1.37），按实测改标准

## 2026-09-20 · P2-1 质量过滤、P2-2 去重
- P2-1：第一版 Gopher 规则在 fineweb-edu 上丢 3.8%（它已被同样规则过滤过，应接近 0）。对照 datatrove 发现分母和重复 n-gram 计数方式不同，改后 99.2%
- P2-1：中文按字套 Gopher 阈值，zh_web 只剩 40%；n×3 后 95.2%，被丢的是整段抄了几遍的文档
- P2-1：finemath 的论坛界面文字（"Reply"）让"重复行按条数"误杀 10%，数学只按字符判断
- P2-2：MinHash 128 个哈希在中文维基上精确率 51–84%（模板条目 J≈0.75，估计噪声推过阈值，赢家的诅咒）；256 个时 86–96%。暴力枚举 2000 篇作为标准答案
- P2-2：不复核 LSH 候选对，精确率掉到 7%；原单元测试没抓到，补了 test_bucket_collision_is_not_a_duplicate
- P2-2：段落去重在代码里会删 Apache 许可证头，代码不做
- 两次用 heredoc 里的 Python 字符串改源码，把正则里的 \n 写成了真换行，改用 Edit 工具
