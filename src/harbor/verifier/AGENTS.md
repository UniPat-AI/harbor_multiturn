# verifier 目录说明

- 这里维护 Harbor verifier 的通用实现。
- 多轮当前主要通过 `trial/trial.py` 的 `_verify_round()` 切换 round-specific tests；改这里时要确认不会破坏多轮 reward 读取契约。
