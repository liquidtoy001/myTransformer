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
