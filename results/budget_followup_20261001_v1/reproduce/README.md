# 复算、执行配置与训练复现

从这页可以沿“结果 → 实际参数 → 执行源码 → 数据输入”定位一条实验。阅读入口使用方法名称，归档保留执行时的字段和哈希以核对来源。

## 先复算已经完成的结果

在结果目录运行：

```bash
python3 reproduce/verify_tables.py
python3 reproduce/verify_public.py
python3 reproduce/verify_handoff.py
```

三个脚本只用Python标准库，分别核对正文数字、发布文件哈希、完整矩阵与实际配置/名称表/链接。图表与新名称表的重建入口是：

```bash
python3 reproduce/build_reader_tables.py
python3 figures/gen_fig_budget_followup.py
```

名称表生成只用标准库；绘图需要NumPy和matplotlib。[阅读表清单](../reader/BUILD_MANIFEST.json) 与 [图表核验记录](../figures/figure_numeric_audit.json) 保存输入哈希和样本数。表图由归档生成，重建属于结果整理。

## 查实际配置

[INDEX.csv](actual_configs/INDEX.csv) 以研究、方法名称、开发/验证阶段和seed索引428条执行配置：

- 组件选择：192条验证，配置原文件SHA与逐seed原值一致；条件方法的28份配置已经补齐。
- 服务器控制：128条开发、108条验证，配置来自已接受原值内的`config_original`。完整参数的独立JSON保存在 [actual_configs/server_feedback/](actual_configs/server_feedback/)。
- 关停：24条续训的配置与父模型绑定位于 [对应配置目录](configs/3_controller_switch_off/)，父状态/原轨迹/续训轨迹的哈希同时记录在 [逐seed配对原值](../data/3_controller_switch_off/paired_values.jsonl)。

第二组原来的108份选择前计划保存在历史配置目录。实际配方从上述索引进入；已接受配置中的历史描述字段按原样保留，最终接受状态由结果原值内的队列/审计记录闭合。

提取JSON做了排版和路径脱敏：执行时原配置SHA、来源容器SHA、公开文件内容SHA分别有记录。`expanded_sha256`等内嵌字段仍指向执行时原件，公开文件的核对依据是`PUBLIC_MANIFEST.json`。

## 查实际执行源码

[executed_sources/](executed_sources/) 保存三组研究使用过的冻结快照；每个快照有逐文件源码清单。实际配置的源码清单字段、逐seed结果的源码SHA及公开清单共同定位对应版本。

| 研究 | 执行入口 | 分析与核验 |
|---|---|---|
| 组件选择 | [训练代码](executed_sources/1_budget_selection/validation02/scripts/run_pcla_v7.py)、[原生方法](executed_sources/1_budget_selection/native04/scripts/run_pcla_v7_native.py) | [导出/统计源码](executed_sources/1_budget_selection/analysis_scripts/) |
| 服务器控制 | [完整LoRA与控制器](executed_sources/2_server_feedback/core02/scripts/run_pcla_v7.py)、[原生方法](executed_sources/2_server_feedback/native03/scripts/run_pcla_v7_native.py) | [导出/统计源码](executed_sources/2_server_feedback/analysis_scripts/) |
| 关停续训 | [完整状态恢复与关停驱动](executed_sources/3_controller_switch_off/policy02/scripts/run_policy_switch.py) | [配对分析源码](executed_sources/3_controller_switch_off/analysis_scripts/) |

组件选择条件方法与独立计时分别有对应冻结快照，实际配置/原值的SHA标出所用版本。旧版本用于解释恢复、初始化、封存与技术检查的变更；正式结果按逐项接受的版本定位。

## 环境与大型输入

[代表技术运行的环境原件](environment/reference_run_identity.json) 记录：Python3.11.15、PyTorch2.5.1+cu121、Transformers4.57.1，以及显卡、模型/分词缓存、数据清单和源码哈希。模型版本为BERT-base-uncased的 `86b5e0934494bd15c9632b12f734a8a67f723594`；模型使用FP32，训练每GPU固定4CPU线程，TF32关闭。依赖文件见仓库 [requirements.txt](../../../requirements.txt) 与 [补充测试依赖](../../../requirements_pcla_repair_extra.txt)。

[data_manifest/](data_manifest/) 提供切分角色、样本编号、分词设置与缓存身份。角色包括50客户端训练/本地读数、公共512、诊断1024/2048及评价4096，划分先按规范化文本分组，再分角色；同seed的方法复用参与与初始化随机性。

模型检查点和客户端更新张量保存在实验档案，体积与取得说明见 [SOURCES](../SOURCES.md)。完整训练复现准备公开SST-2原文本、固定模型版本、对应分词缓存及角色清单；关停和旧更新复用诊断另外需要所绑定的父检查点/更新张量。实际配置内的路径占位符映射到复现环境后保存为新配置，输出保存到新目录。

普通训练驱动的调用形式是：

```bash
python "$TRAIN_SCRIPT" --config "$CONFIG" --out "$NEW_OUT" --device cuda
```

`TRAIN_SCRIPT`取上表相应入口，`CONFIG`取实际配置索引中的参数和本地输入路径。恢复与关停驱动的额外参数见相应脚本。新环境运行会产生自己的配置/源码/输入哈希，原实验档案与复现输出分别保存。

## 数据字段怎样读

主端点是实际通信返回模型的原参数准确率；固定100轮、半额度、EMA、末5/末20轮与NLL各自分列。第一组独立seed401计时与质量12seed并列，第二组固定三个seed用于主时间比较。逐seed配对差先在seed内相减再求t95，缺测字段保留空值。规则级费用中的支持级读数重复字段按支持去重。定义与完整参数见 [METHODS](../METHODS.md)。
