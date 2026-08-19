# 项目地图

## 顶层目录

| 目录 | 内容 | 使用建议 |
|---|---|---|
| configs/ | JSON 实验协议 | 先看配置，再看对应结果目录 |
| data/ | SST-2、QNLI、AG News 及清单 | 数据来源和字段见 data/README.md |
| docs/ | 研究解释和复现笔记 | 从本文件或 REPRODUCE.md 进入 |
| models/ | tokenizer/config 文件 | 权重需本机准备，不进 Git |
| reports/ | 教师汇报版材料 | 由 results/ 生成，不作为训练输入 |
| results/ | 原始运行记录 | 看 results/README.md 判断证据级别 |
| scripts/ | 运行、验证和绘图入口 | 优先使用现成脚本，不手工改 CSV |
| src/ | 核心训练和分配逻辑 | 主要入口为 src/fedhello_sst2.py |
| tests/ | 分配与调度测试 | 改分配逻辑后必须先运行 |

## 文档分工

- PCLA.md：面向理解方法的中文说明和图表。
- PCLA工作报告.md/html：教师汇报材料；HTML 中有本机路径，GitHub 上优先看 Markdown。
- FEDHELLO_REPRODUCTION_NOTES.md：Fed-HeLLo 机制复现边界、假设和命令。
- PCLA_LARGE_SCALE_CONSTRAINED_V1.md：组件级 PCLA 主线的正式协议与解释。
- PCLA_COMPONENT_*、PCLA_ONLINE_*：筛选、消融和被拒绝假设的实验记录。
- RESULTS.md：经典 baseline 主表的结果解释。
- LEGEND失稳原因与正式结果.md：LEGEND 诊断和确定性修正的 provenance。

## 为什么不移动旧结果

结果目录名称被配置文件、绘图脚本、论文材料和 checkpoint 共同引用。当前重整采取“增加索引、不破坏路径”的方式：保留原始路径作为 provenance，同时在 README.md、results/README.md 和本项目地图中提供可读入口。
