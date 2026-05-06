# harbor 目录说明

- 这里是当前仓库实际使用的 Harbor 源码，不是只读镜像。
- 本地维护时只关注最终代码状态，不按历史 commit 叙事。
- 这个本地 Harbor 副本承载了当前工作区真正生效的多轮支持，也包含当前 adapter tooling 对来源仓库 / 论文 URL 的元信息支持；文档只描述现状，不回顾演化过程。
- 多轮相关主入口集中在 `src/harbor/`，回归主要看 `tests/unit/` 和根目录 `tests/`。
- 如果 Harbor 行为改变，必须同步检查根目录的 `DESIGN.md`、`IMPLEMENTATION.md`、`USAGE.md`、`TEST.md`。
