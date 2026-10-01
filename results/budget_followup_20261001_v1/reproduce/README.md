# 复算与复现

这一页按“结果 → 实际参数 → 执行源码 → 数据输入”的顺序，帮你定位任意一条实验。

## 1. 复算已有结果

在结果目录下运行：

```bash
python3 reproduce/verify_tables.py
python3 reproduce/verify_public.py
python3 reproduce/verify_handoff.py
```

| 脚本 | 核对什么 |
|---|---|
| `verify_tables.py` | 从 `data/` 重新计算 README 里引用的全部数字 |
| `verify_public.py` | 按清单核对每个发布文件的哈希，并检查没有清单之外的文件 |
| `verify_handoff.py` | 核对完整实验矩阵、实际配置、按方法名称整理的表，以及文档里的链接 |

三个脚本只用 Python 标准库。

重新生成方法名称表和图：

```bash
python3 reproduce/build_reader_tables.py
python3 figures/gen_fig_budget_followup.py
```

生成表只用标准库，画图需要 NumPy 和 matplotlib。输入哈希和样本数分别记在 [BUILD_MANIFEST.json](../reader/BUILD_MANIFEST.json) 和 [figure_numeric_audit.json](../figures/figure_numeric_audit.json)。

## 2. 查实际配置

[INDEX.csv](actual_configs/INDEX.csv) 按研究、方法名称、阶段和种子索引了 428 条执行配置：

- **组件选择：** 192 条验证，配置文件的哈希与逐种子原始记录一致。
- **服务器控制：**
  - 128 条开发和 108 条验证，从原始记录里提取；
  - 每条的完整参数单独存成 JSON，在 [actual_configs/server_feedback/](actual_configs/server_feedback/)；
  - 选方法之前写的 108 份计划文件作为历史保留，以索引里的执行配置为准。
- **关停续训：** 24 条续训的配置在 [configs/3_controller_switch_off/](configs/3_controller_switch_off/)。起点模型、原轨迹和续训轨迹的哈希记在 [逐种子配对记录](../data/3_controller_switch_off/paired_values.jsonl) 里。

提取出的 JSON 调整了排版并替换了路径，所以会同时记录三个哈希：执行时的原配置、原始记录文件、公开文件。配置里内嵌的哈希字段仍指向执行时的原件。

## 3. 查执行源码

[executed_sources/](executed_sources/) 保存三部分实验用过的每一版冻结源码，每版都有逐文件清单。配置里的源码清单字段，以及逐种子结果里的源码哈希，可以对应到具体版本。

| 研究 | 训练入口 | 分析脚本 |
|---|---|---|
| 组件选择 | [训练](executed_sources/1_budget_selection/validation02/scripts/run_pcla_v7.py)、[已发表方法](executed_sources/1_budget_selection/native04/scripts/run_pcla_v7_native.py) | [导出与统计](executed_sources/1_budget_selection/analysis_scripts/) |
| 服务器控制 | [完整 LoRA 与控制器](executed_sources/2_server_feedback/core02/scripts/run_pcla_v7.py)、[已发表方法](executed_sources/2_server_feedback/native03/scripts/run_pcla_v7_native.py) | [导出与统计](executed_sources/2_server_feedback/analysis_scripts/) |
| 关停续训 | [恢复状态并关停](executed_sources/3_controller_switch_off/policy02/scripts/run_policy_switch.py) | [配对分析](executed_sources/3_controller_switch_off/analysis_scripts/) |

其他版本用于技术检查、计时、诊断、断点恢复和初始化修复。每条正式结果用的是哪一版，由它的配置和原始记录里的源码哈希确定。第一部分各版相对仓库提交 `a3f72f4`（PR #3）的完整差分，在 `executed_sources/1_budget_selection/*_vs_base.patch`。

## 4. 环境与数据输入

[运行环境记录](environment/reference_run_identity.json)：

- **软件：** Python 3.11.15、PyTorch 2.5.1+cu121、Transformers 4.57.1；
- **模型：** BERT-base-uncased，版本 `86b5e0934494bd15c9632b12f734a8a67f723594`，FP32，关闭 TF32；
- **线程：** 每张 GPU 固定 4 个 CPU 线程；
- **哈希：** 显卡、模型与分词缓存、数据清单、源码。

依赖见仓库的 [requirements.txt](../../../requirements.txt) 和 [补充依赖](../../../requirements_pcla_repair_extra.txt)。

[data_manifest/](data_manifest/) 是数据切分清单，只包含样本编号，不含文本：

- 每个客户端的训练和本地读数样本；
- 512 条公共数据；
- 1,024 条和 2,048 条诊断样本；
- 4,096 条评价样本；
- 分词设置和缓存身份。

切分方法是先按规范化文本分组，再分配角色。同一种子下各方法共用参与顺序和初始化。

## 5. 重新训练

完整重跑需要：

- 公开的 SST-2 原文；
- 固定版本的模型；
- 对应的分词缓存；
- 上面的切分清单。

关停续训和复用更新的诊断还需要对应的起点检查点和更新张量。它们存放在实验归档里，哈希见 [SOURCES](../SOURCES.md)。

把配置里的路径占位符换成本地路径，另存为新配置，再运行：

```bash
python "$TRAIN_SCRIPT" --config "$CONFIG" --out "$NEW_OUT" --device cuda
```

`TRAIN_SCRIPT` 取上表的训练入口，`CONFIG` 取配置索引里对应的文件。恢复和关停脚本的额外参数见脚本本身。新的运行会产生新的配置、源码和输入哈希，输出请放在新目录，不覆盖原实验记录。
