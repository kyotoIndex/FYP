# 实验来源、环境与资源口径

本版于 2026-10-19 从本地 FYP 已完成主线整理，在独立 checkout 中准备正式仓库 PR。当前算法代码与发布结果以同一目录结构收录，便于对应配置、实现和轨迹。

## 代码与结果如何对应

| 内容 | 发布位置 |
|---|---|
| 当前完整训练与汇总源代码 | `federated_finetuning/experiments/scra_diagnostics_20261017/scra/` |
| 实验协议 | 同实验目录的 `plan*.json`、`plans_batch/`、`plans_batch2/`、`plans_c2fed/` |
| RoBERTa 两批训练轨迹 | `federated_finetuning/runs/scra_batch*_20261018/*/d0/d0_rounds.json` |
| 原诊断 / 基础参照轨迹 | 同 runs 下 `scra_diagnostics_20261017_v1*` 和 `headroom_*` |
| Qwen 38 组轨迹与身份 | `runs/c2_fed_20261018/*/{fed.json,identity.json}`（相对 `federated_finetuning`） |
| Qwen 15 组运行的源码副本 | 对应运行目录的 `source_snapshot/` |
| 原报告与复算入口 | [RESULTS](RESULTS.md)、`scripts/reproduce_results.py` |

Qwen 的不同实验阶段使用过三个旧版 `c2fed.py`；随后用当前代码各完整重放一个代表运行。保存报告中这三个重放均通过，版本核对放行全部三种子比较。当前核心训练文件保持原实验版本，包装、准备和文档改动放在独立脚本中。

RoBERTa 批次 1 的身份记录较早，源码版本字段不完整；批次 2 身份中的 `model.py` 与后来当前文件版本有差异。发布包保存原身份与现有实现，并提供新工作目录重训入口。若需逐位重演该历史训练，缺少的精确源码快照仍需从原执行环境恢复；本版离线复算使用保存轨迹，三份原结果 JSON 可完整重建。

## 环境和外部资产

- Python 3.12；PyTorch 2.8.0+cu128；Transformers 4.57.1；NumPy 2.2.6。
- 训练身份记录的 GPU：NVIDIA GeForce RTX 4090 D，24 GB。
- RoBERTa-base 版本：`e2da8e2f811d1448a5b465c236feacd80ffbac7b`。
- Qwen2.5-0.5B Base 版本：`060db6499f32faf8b98477b0a26969ef7d8b9987`。
- 官方 MultiNLI 1.0 原始 JSONL，经纯领域客户端分区与 premise-group 验证留出；参数与行数见 [REPRODUCE](REPRODUCE.md)。

`prepare_assets.py` 固定模型版本，并在生成数据清单时记录原始来源。模型、tokenizer、数据各自遵循上游条款；下载准备阶段保留模型仓库附带的许可文件。当前仓库未另行指定整体开源许可证。

## 资源口径

当前训练为 **masked rank-R 实现**：完整 A/B 是 Parameter，未选组件的梯度清零，Adam 状态与 amp 起点副本仍是完整形状。服务器在内存中筛选本轮上传集合，模拟组件通信；尚无真实分布式网络层。

| 数量 | 本版口径 |
|---|---|
| 弱端训练可变组件 | 每 adapter 为 r，强端为 R |
| 逻辑 LoRA 因子上传 | 弱端 / 全量 = r/R；RoBERTa 还需加上共同分类头 |
| 实际进程与网络 | 一张 GPU 串行模拟客户端；稀疏网络编码未实现 |
| 当前优化器状态 | 完整 rank 形状，梯度 mask 后保持 |
| RoBERTa 峰值显存 | 已保存运行约 1557–1558 MiB；各运行时间见成本表 |
| C1 的峰值节省 10–14% / 计算节省至多约 8% | folded / Select-N-Fold 参数化下的分析估算；激活为粗估 |

详细核算见 [resource_case](designs/resource_case_2026-10-18.md) 与 `python docs/designs/checks/c1_resource.py`。论文叙述区分准确率实证、逻辑上传预算和 folded 资源估算，资源表分别标注。

## 本次正式化的验证

本次工作重新安装独立环境，运行现有模型/算法测试、离线生成并比对三份原始结果 JSON、重建实验图及可编辑方法图，并检查配置、文档链接和仓库内容。GPU 完整训练沿用已完成运行，新的完整重训由读者按 [REPRODUCE](REPRODUCE.md) 执行。

验证的具体数量和结果见 [RELEASE_CHECK](RELEASE_CHECK.md)。
