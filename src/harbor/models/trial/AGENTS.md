# trial models 目录说明

- 这里定义 TrialConfig、TrialPaths 和 TrialResult 等 trial 级别模型。
- 多轮重点字段集中在 verifier config 的 start/max/resume/snapshot 相关参数。
- 字段名改动会同时影响 CLI、Job、Trial 主循环和黑盒 case。
