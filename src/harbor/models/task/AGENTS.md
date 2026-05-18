# task models 目录说明

- 这里定义任务路径、任务元数据和多轮 `change_types` 归一化。
- 推荐输出格式是 `change_types = [...]`，但当前代码仍兼容旧 `change_type` 写法。
- 变更这里时，要同步检查样例任务 `task.toml` 与 `harbor/tests/unit/multiround/test_task_change_types.py`。
