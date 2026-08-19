# 结果目录

结果目录按证据级别和实验阶段组织。原始 CSV、checkpoint、运行配置和图表尽量一起保留，不手工改写已完成运行。

## 主要结果

| 目录 | 角色 | 状态 |
|---|---|---|
| expanded_sst2_bert_base_dirichlet_v1 | 经典 baseline 主表 | 3 seed，已审计 |
| pcla_large_scale_constrained_dirichlet_v1 | 组件级 PCLA 主线 | 3 seed，已审计 |
| legend_large_scale_constrained_dirichlet_v2 | 确定性成本模型的 LEGEND 对照 | 已验证的正式对照 |
| formal_comparison_sst2_bert_base_v1 | 经典方法正式比较的历史版本 | 已保留 |

## 其他结果

- formal_v2/：跨数据集/模型矩阵。当前 matrix_status.json 表明运行不完整，不能当作矩阵全部完成。
- _development/：开发和单配置验证。
- diagnostics/：根因分析、失稳复现和消融诊断。
- 其他带 screening、warmstart 或旧版本后缀的目录：保留作研究 provenance，不能自动升级为主结论。

验证命令示例：

~~~powershell
python scripts\validate_results.py --config configs\expanded_sst2_bert_base_dirichlet.json --results results\expanded_sst2_bert_base_dirichlet_v1
~~~

正式报告应使用多 seed 汇总，并明确区分 accuracy、通信量、模拟时间和实际机器运行时间。开发、smoke、单 seed 和诊断结果只能用于排错或说明机制。
