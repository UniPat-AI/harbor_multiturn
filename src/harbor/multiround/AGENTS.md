# multiround 目录说明

- 这里放 Harbor 多轮本地扩展的纯辅助模块。
- 优先把可单测的 plan、选择、lineage 等多轮决策收敛到这里，减少对上游 Harbor 大文件的改动面。
- 不在这里启动 job、修改文件系统运行产物或引入新的 CLI 参数；副作用仍由调用方负责。
- 改动本目录时，要同步检查 `harbor/tests/unit/multiround/` 和根目录实现/测试文档。
