# trial 目录说明

- 这里维护 Trial 主循环，是多轮执行、快进、resume、trajectory merge、snapshot 聚合的核心位置。
- 改这里时，要同步检查 `cli/jobs.py`、`job.py`、相关 agent 实现和多轮单元测试。
- 这里的行为变化通常也需要回写根目录 `IMPLEMENTATION.md` 与 `USAGE.md`。
