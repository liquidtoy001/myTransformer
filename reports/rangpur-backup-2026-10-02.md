# Rangpur 完整备份与空间清理

完成时间：2026-10-02 19:44（Australia/Brisbane）。用户已授权备份核验后清理项目空间。

## 备份

本地归档：`backups/rangpur/2026-10-02/rangpur-full.tar`，10,183,639,040 字节，约 9.48 GiB。

SHA256：`6016966b2d5bbe0f92edaf80c04c7e4881729995679c34be581480cf10afbb29`。

归档涵盖远端完整 `myTransformer` 和 `mt-venv`，包括隐藏文件、代码、配置、数据、日志、模型、缓存及符号链接信息，另附 Git 和 Python/conda 环境证据。该归档含私有环境信息，不纳入 Git，不公开上传。

原 CPU 作业 622741 已打包但在等待下载时超时。恢复作业 623276 完成最终归档与传输；验收以本报告的最终 SHA256 为准，不使用旧归档哈希。

## 实际通过的检查

- 归档 SHA256 与远端记录一致。
- 14,983 个 manifest 条目的路径、文件内容哈希、文件大小及链接信息核对通过。
- 23 份模型文件（21 份 final.pt、2 份 smoke checkpoint）严格加载，CPU 短前向输出有限；完整 checkpoint 检查恢复状态字段存在。未宣称重新执行所有完整训练或评测。
- 22 份 metrics.jsonl 可解析并检查顶层浮点有限值。
- 清理前在 CPU 作业内重新扫描源目录，逐项匹配 manifest，确认未改变；确认没有其他用户作业，再执行限定目录清理。

原始审计文件均在上述备份目录：`archive-validation.json`、`remote.manifest.json`、`remote.sha256`、`cleanup-output.txt`、`COMPLETE.json`、`workflow.log`。

## 清理结果

仅删除远端 `~/myTransformer` 与 `~/mt-venv`；保留 `~/miniconda3` 和其他项目。完成后再次通过 SSH 确认目录状态与空队列。

home 从使用约 14G、可用 2.8G，降至最新使用约 4.9G、可用约 12G（df 的舍入值）；释放约 9G 空间。删除后的即时读数曾为使用 5.3G、可用 11G，最新查询进一步回落，不能将舍入差值当作逐字节计量。

本地归档是恢复来源，Linux venv 的符号链接未在 Windows 展开；需要恢复环境时参考记录重建。备份目前位于本地 E 盘，尚未验证第二块磁盘或对象存储副本。未来 P7/P8 再使用 Rangpur 时，应重新部署项目和所需数据。
