# environments 目录说明

- 这里维护 Harbor 环境抽象和具体环境实现。
- 多轮当前最关键的是 snapshot 捕获/恢复能力，以及 mounted / non-mounted 环境差异。
- 改环境 snapshot 语义时，要同步检查 `trial/trial.py`、多轮单元测试和黑盒 fanout/resume 测试。
