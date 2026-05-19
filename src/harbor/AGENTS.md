# harbor/src/harbor 目录说明

- 这里是 Harbor 主包源码。
- 多轮任务能力主要分布在 `models/task/`、`cli/`、`multiround/`、`job.py`、`trial/`、`agents/`、`environments/`。
- 改动多轮行为时，要同时检查 `harbor/tests/unit/multiround/` 和根目录 `tests/`。
- 本目录说明只描述当前实现，不解释历史方案。
