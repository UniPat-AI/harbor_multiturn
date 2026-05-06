# terminus_2 目录说明

- 这里维护 Terminus-2 agent 及其多轮 runtime state 恢复逻辑。
- 重点关注 trajectory round 标记、runtime snapshot、shell cwd、tmux 相关恢复字段。
- 改这里时，要同步检查 `harbor/tests/unit/multiround/test_terminus_resume_state.py` 和根目录 Terminus-2 黑盒回归。
