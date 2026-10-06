# 接手说明

[README](README.md) 讲结果和结论。这一页补充四样 README 里没有展开的东西，方便你接着写论文或查细节：

1. 研究是怎样一步步走到这里的；
2. 我已经完成了哪些工作，每项的完整数据在哪里；
3. 执行和成本记录怎么读；
4. 写论文时，每部分材料可以怎么用。

## 1. 研究脉络

最初的问题是：联邦 LoRA 里，每个客户端只更新部分低秩组件时，能不能根据局部分数把有限的更新量放到更有用的位置。这个问题有两层：

- 分数能不能识别短期收益；
- 整个训练方法在同样的资源约束下是否更好。

三个 PR 依次把这两层拆开：

| 阶段 | 做了什么 | 主要发现 | 入口 |
|---|---|---|---|
| PR #2 | 8 种方法 × 5 个种子 × 30 轮；比较按分数挑、同层同数量随机编号、随机组件；另做单步评分诊断 | PCLA 末 5 轮平均 78.40%，同名额随机编号 78.81%；分数能预测单步收益 | [PR #2](https://github.com/kyotoIndex/pcla-experiments/pull/2)、[结果](../core_shared_hh_20260918_v3/RESULTS.md)、[评分诊断](../rank_signal_20260918_v1/README.md) |
| PR #3 | 聚合检查与开源实现对照、单模型训练、100 轮训练、SVD 重拆、个性化、AG_NEWS | 聚合没有问题；波动来自标签偏斜和部分参与；SVD 重拆后按分数挑比随机编号高 3.23 个百分点 | [PR #3](https://github.com/kyotoIndex/pcla-experiments/pull/3)、[结果](../group_meeting_followup_20260924_v1/README.md) |
| 本 PR 第一部分 | 固定通信量下比较 16 种方法，含完整 LoRA 和已发表方法 | 编号选择有 +0.61 个百分点的作用，整套方法比完整 rank8 低 4.36 个百分点 | [README 第一部分](README.md#第一部分通信量相同时挑组件比不过完整-lora) |
| 本 PR 第二部分 | 服务器分别控制 LoRA 和分类头的更新幅度 | 单轮收益来自缩小步长；整段训练比 rank8 低 1.30 个百分点；服务器上训练分类头最实用 | [README 第二部分](README.md#第二部分服务器控制更新幅度单轮有效整段训练不行) |
| 本 PR 第三部分 | 从第 60 轮关掉控制器，续训 20 轮 | 关掉后先大幅下降，再慢慢恢复；“控制器拦住学习”不成立 | [README 第三部分](README.md#第三部分从第-60-轮关掉控制器) |

写论文时引用 PR #3 的 SVD 结果，有两个数各管一件事：

- **SVD 条件下按分数挑比随机编号高 3.23 个百分点**，区间 [−0.57, +7.03]：回答“SVD 下挑组件有没有用”。
- **SVD 带来的选位收益比原拆法多 3.63 个百分点**，区间 [+0.46, +6.81]：回答“SVD 是否改变了挑组件的作用”。

本 PR 第一部分则检验了：换成固定通信量和单投影组件后，SVD 配方还能不能胜出。

## 2. 已完成的工作和完整数据

| 内容 | 规模 | 数据 |
|---|---:|---|
| 组件选择开发 | 117 种配方 × 3 个种子 = 351 条训练 | [配方参数与结果](reader/component_development.csv)、[逐种子结果](data/1_budget_selection/development/seed_endpoints_351.csv)、[全部曲线](data/1_budget_selection/development/all_recipe_curves.html) |
| 组件选择验证 | 16 种方法 × 12 个种子 = 192 条训练 | [逐种子结果](reader/component_validation_seeds.csv)、[逐轮曲线](data/1_budget_selection/validation/v_round_curves.csv.gz)、[参数滑动平均曲线](data/1_budget_selection/validation/v_ema_rounds.csv.gz) |
| 组件选择计时 | 117 种配方，各跑一次 | [全部计时结果](data/1_budget_selection/costs/cost_endpoints.csv)、[16 种方法的质量与时间](reader/component_methods.csv) |
| 挑法诊断 | 96 个 | [逐客户端收益](data/1_budget_selection/diagnostics/selection_bank/cohort_client_gains.jsonl.gz)、[排序](data/1_budget_selection/diagnostics/selection_bank/cohort_client_ranks.jsonl.gz)、[组合选择](data/1_budget_selection/diagnostics/selection_bank/cohort_selection16.jsonl) |
| 拆法诊断 | 48 个 | [读数](data/1_budget_selection/diagnostics/coordinates/direction_overall_readouts.jsonl)、[按方法名称汇总](reader/coordinate_diagnostics.csv)、[几何关系](data/1_budget_selection/diagnostics/raw/direction_geometry.jsonl.gz) |
| 服务器控制开发 | 32 种配方 × 4 个种子 = 128 条训练 | [配方参数与结果](reader/server_development.csv)、[完整原始记录](data/2_server_feedback/development/accepted_originals.jsonl.gz) |
| 服务器控制验证 | 9 种方法 × 12 个种子 = 108 条训练 | [逐种子结果](reader/server_validation_seeds.csv)、[完整原始记录](data/2_server_feedback/validation/accepted_originals.jsonl.gz)、[控制器每轮选择](data/2_server_feedback/validation/controller/controller_originals.jsonl.gz) |
| 更新控制诊断 | 96 个复用 + 24 个新训练 | [各规则读数](data/2_server_feedback/diagnostics/rules.csv)、[LoRA 与分类头可加性](data/2_server_feedback/diagnostics/four_corners.csv)、[汇总](data/2_server_feedback/diagnostics/summary/) |
| 关停续训 | 24 条 × 20 轮 | [逐种子配对值](reader/switch_off_seed_pairs.csv)、[完整配对记录](data/3_controller_switch_off/paired_values.jsonl)、[费用](data/3_controller_switch_off/costs/) |
| 复现材料 | 源码、配置、数据切分、原始记录 | [复现说明](reproduce/README.md)、[428 条配置索引](reproduce/actual_configs/INDEX.csv)、[文件来源](SOURCES.md) |

上面六项训练合计 920 条，包括开发、验证、计时和续训。诊断单独计数：服务器控制诊断里的 96 个对象复用了第一部分已有的客户端更新，这部分只新增了服务器端处理和读数，没有重新训练客户端。

每条结果进入汇总之前都逐项核对过：配置、源码、通信记录、客户端步数、返回模型和费用。配置索引把方法名称、阶段、种子和实际执行的配置对应起来。选方法之前写的计划文件保留作历史；复现时以已接受结果里的执行参数为准。

## 3. 执行和成本记录怎么读

成本分四类记录，用途各不相同：

| 类别 | 包含什么 | 怎么用 |
|---|---|---|
| 必要执行时间 | 初始化、建立状态和优化器、客户端训练、打分、服务器决策、消息处理、预算控制 | 比较各方法生成返回模型的计算代价，README 里的时间都是这一类 |
| 研究读数 | 准确率和损失评价、参数滑动平均、保存 | 单独记录，不算进方法成本 |
| 技术检查与失败尝试 | 恢复一致性、消息与硬件检查、初始化失败、导出或审计失败 | 作为执行费用说明，原件和计时都保留 |
| GPU 占用时长 | 排队、预留和多个任务共用资源的总跨度 | 和单条运行的必要时间分开算 |

计时口径：

- **第一部分：** 质量来自 12 个验证种子，时间来自种子 401 在独占 4090D 上的单独运行，两者并列展示。同名额随机编号对照没有单独计时。
- **第二部分：** 时间比较用固定的三个种子，其余种子的时间另有描述表。

执行中出现过的问题都已修复，原始记录保留：

- **第一部分：**
  - 训练结束后封存结果时，一处 JSON 键类型比较报错。两条已完成的 100 轮训练，用原检查点、指标和通信日志恢复了元数据，训练本身没有重跑；之后的运行都用修复版。
  - 计时早期的技术检查、GPU 身份预领、审计和导出解析也有修复记录。
  - [历史执行索引](data/1_budget_selection/costs/historical_execution_index.json) 登记了 825 份去重后的原件和来源哈希，[收集回执](data/1_budget_selection/costs/historical_collection_receipt.json) 核对缺件为零。
- **第二部分：** 有 14 次训练前的输出目录检查失败。原日志和部分初始化费用归档后，任务重新绑定，正式验证全部完成。
- **第三部分：** 24 条正式续训全部完成；两次 CPU 输入预检失败，各自留有来源哈希。

没有测到的项目（例如部分历史缓存和随机数状态、未计时的 CPU 工作）保存为空值或 unknown，不按零计。

服务器控制诊断的费用表里，“每份更新的独立读数总时间”在多条规则行重复出现，汇总时要按更新去重。

## 4. 写论文时怎么用这些材料

**图和章节的对应：**

| 论文需要 | 材料 |
|---|---|
| 固定通信量下的质量与成本 | [图 1](figures/fig_1_fixed_budget.png)（[PDF](figures/fig_1_fixed_budget.pdf)）、[关键配对比较](reader/key_paired_comparisons.csv) |
| 单轮收益与整段训练的差别 | [图 2](figures/fig_2_server_feedback_curves.png)（[PDF](figures/fig_2_server_feedback_curves.pdf)） |
| 关停检验 | [图 3](figures/fig_3_controller_switch_off.png)（[PDF](figures/fig_3_controller_switch_off.pdf)） |
| 设置章节的参数表和数据切分 | [METHODS](METHODS.md)、[配方参数表](reader/) |
| 背景与基线（聚合、异质性） | [PR #3 的结果](../group_meeting_followup_20260924_v1/README.md) |

**写作时几处需要讲清楚的地方：**

- **准确率和损失是两个目标。** 控制器的平均损失更低、准确率更低，这组取舍要在论文里保留。
- **主读数统一用额度用完时返回的原参数模型。** 固定第 100 轮、半额度、参数滑动平均、末 5 轮都另有字段；控制器方法的第 100 轮和返回模型可能不是同一个。
- **方法之间的差值统一用同一种子下的配对差。** 区间以种子为单位，检查点、客户端和网格点都不当作独立样本。
- **拆法诊断有两种读数。** 一种保留原幅度，一种把更新幅度对齐后只看方向变化。比较时先在同一起点模型、同一种子内比，再对种子汇总。
- **个性化和共享模型是两个目标。** PR #3 的“共享训练后本地微调”结果来自客户端本地评价；本 PR 评价的是共享模型在统一评价集上的表现。两者在论文里分开组织。
- **数据集覆盖。** 固定通信量和服务器控制的实验都在 SST-2 上完成，PR #3 另有 AG_NEWS。如果要加第二个数据集，README 的“下一步建议”里给了最小的方案。

**我建议的叙述顺序：**

1. 把完整 rank8 作为强参照；
2. 用同名额随机编号对照说明编号选择的实际作用；
3. 把服务器上训练分类头作为简单而实用的方案；
4. 用诊断和关停检验解释单轮收益为什么没有变成最终准确率。

这样四块可以连成一条完整的叙述。论文题目和贡献的写法，等你看完结果后再一起商量。
