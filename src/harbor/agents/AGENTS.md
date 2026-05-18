# agents 目录说明

- 这里定义 Harbor agent 接口和具体 agent 实现。
- 多轮支持以 `run_round()` 为准；没有显式多轮实现的 agent 不应被当作多轮 agent 使用。
- 当前多轮主维护对象是 `oracle`、`installed/claude_code.py`、`terminus_2/terminus_2.py`。
