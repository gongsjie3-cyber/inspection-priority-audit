# 数据来源

- 发布方：City of Chicago / Chicago Department of Public Health。
- 数据集：[Food Inspections, 4ijn-s7e5](https://data.cityofchicago.org/d/4ijn-s7e5)。
- 本项目范围：2019-01-01 至 2025-12-31。
- 下载时间：2026-10-10，精确 UTC 时间见 `manifest.json`。
- `inspections_2019_2025.csv.gz` 是官方公开数据的 11 列导出，共 123,673 行；不包含检查员身份或自由文本违规描述。
- `manifest.json` 保存逐年官方查询地址、原始响应校验值及快照 SHA-256。
- 商户地址、经纬度及许可证编号来自官方公开记录；报告与评分展示使用哈希标识。哈希标识是展示用代号，不构成对可公开关联数据的匿名性保证。
- 数据来源与适用条款归原发布方；本项目不授予原始数据额外权利。
- `audit.sqlite` 由 `python src/pipeline.py` 生成，不纳入版本控制。

该文件是当前历史数据的导出，不是逐时点归档。后续重新下载可能包含回溯修订，无法保证与本快照结果相同。
