# 项目 HTML 介绍

`index.html` 是本地分页介绍的维护源，2026-10-03 更新。保留原阶段报告的章节，并加入第二轮消融、组合验证和 Rangpur 备份清理结果；P6 标为准备阶段，DDP 和正式主训练尚未完成。

私有云端地址：https://mytransformer-project.liquidtoy001226508.chatgpt.site

原 Claude Artifact 链接未修改，新地址为本次发布的新版本入口。

本地可直接打开 index.html，也可在仓库根目录运行 `python -m http.server 8769 --bind 127.0.0.1 --directory docs/site` 后浏览 localhost:8769。

发布 checkout：`E:/Documents/transformer-site`，与训练仓库分离，仅包含介绍页和发布配置。Site ID：`appgprj_6abfca664bfc81919e1638d99e41ea0c`。本次发布源码提交 `b26d478757a507511bbea14c77c9fc2e2d72cb80`。后续必须更新这个 Site，不重新创建。

检查：15 个章节、唯一 ID、所有章节链接目标存在、内联 JavaScript 语法通过、本地与发布 HTML 字节一致；本地浏览器检查了新章节的切换和表格显示。云端由发布接口确认 succeeded，访问范围保持私有。

仅发布 HTML，不上传训练数据、权重、完整备份或环境快照。GitHub 训练仓库未在本次推送。
