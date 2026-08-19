# LEGEND 失稳原因与正式结果

## 1. 结论

最初的 LEGEND 大规模实验中，seed 21 最终准确率退化到 50.72%。该问题不是数据文件损坏、实验未完成、rank 超预算或聚合代码报错，而是以下反馈链造成的：

> 单机 wall-clock 计时噪声 → LCD 整数 depth 阈值变化 → 某个客户端多训练一层 → 强 non-IID 下模型轨迹分叉。

原实现把本机实际测得的客户端训练时间用于下一轮 LCD depth 决策。实际训练时间会受到 GPU 调度、系统负载等因素影响，不受随机 seed 控制。因此，相同 seed 的两次运行也可能产生不同的 LoRA depth。

## 2. 定位证据

对 seed 21 使用完全相同的代码、数据和随机种子重新运行后：

| 运行 | 最终准确率 |
|---|---:|
| 首次 wall-clock 反馈运行 | 50.72% |
| 相同 seed 重跑 | 84.06% |

逐客户端、逐轮比较两次运行的 300 条分配记录后，只发现一处不同：

| 轮次 | 客户端 | 首次运行 | 重跑 |
|---:|---:|---|---|
| 10 | 16 | 层 6–11 | 层 7–11 |

客户端 16 的正样本比例为 12.30%，属于标签高度偏斜的客户端。首次运行中，wall-clock 状态使它在第 10 轮额外参与了第 6 层的更新。从这一轮开始，两次运行的准确率轨迹出现分叉。

这说明严重失稳不是由某个固定 seed 必然造成，而是 LCD 对非确定性计时输入过于敏感。整数 depth 决策是不连续的，微小时间差可能改变一整层是否训练；在强 non-IID 和部分参与条件下，这个差异会被后续非凸优化放大。

## 3. 消融实验

为区分动态 depth 与递增 rank 的作用，只对 seed 21 进行了诊断消融：

| 设置 | 最终准确率 | 结论 |
|---|---:|---|
| wall-clock 动态 depth + 等差 rank，首次运行 | 50.72% | 发生永久退化 |
| 固定 depth + 等差 rank | 86.05% | 没有永久退化 |
| 动态 depth + uniform rank | 80.80% | 有波动，但没有永久退化 |
| wall-clock 动态 depth + 等差 rank，相同 seed 重跑 | 84.06% | 结果不可重复 |

这些结果表明：

1. 递增 rank 本身不会必然导致崩塌；
2. 动态 depth 本身会增加波动，但不会必然永久崩塌；
3. 严重问题来自非确定性时间反馈、离散 depth 和递增 rank 更新之间的交互；
4. 同一 seed 得到 50.72% 或 84.06%，说明 wall-clock 反馈不适合直接用于单机受控算法对比。

## 4. 修正方法

论文中的完成时间模型为：

\[
t_i^h=\tilde t_i+k_i^h\mu_i^h+
\sum_{l\in\mathcal L_i^h}r_{i,l}\beta_i^h
\]

修正后仍然按照该结构计算设备状态，但使用实验配置中固定的异构能力参数：

\[
\hat t_i^h
=1+
k_i^h f_i^{\mathrm{compute}}+
\frac{\sum_{l\in\mathcal L_i^h}r_{i,l}}{r_{\mathrm{base}}}
f_i^{\mathrm{communication}}
\]

然后使用论文中的 EMA：

\[
\bar t_i^h=
0.8\bar t_i^{h-1}+0.2\hat t_i^h
\]

本机实际训练时间仍用于报告模拟总时间，但不再反向决定下一轮模型结构。

这样处理的理由是：

- 我们没有论文中的 80 台真实 Jetson 和动态 Wi-Fi 环境；
- 单机 wall-clock 不能可靠代表某个模拟客户端的设备能力；
- 确定性成本模型保留了论文 LCD 的完成时间、depth、rank 和 EMA 关系；
- 相同 seed 可以复现相同分配。

在当前固定 6/9/12 资源组中，设备能力不会随轮次切换。因此 LCD 每轮重新计算后会稳定到相同的可负担 depth。这是固定设备状态下的合理结果，不代表论文真实平台中的 time-varying depth 已经被完整复现。

## 5. 修正后的正式结果

结果目录：

```text
results/legend_large_scale_constrained_dirichlet_v2
```

三个 seed 均完整运行 30 轮，并通过结果验证器：

```text
seed=7  metrics=30 allocations=300 importance=12
seed=13 metrics=30 allocations=300 importance=12
seed=21 metrics=30 allocations=300 importance=12
VALID: complete experiment outputs and allocation invariants
```

### 5.1 分 seed 结果

| Seed | 最终准确率 | 最后 5 轮平均 | 全轮平均 | 通信量 | 模拟时间 |
|---:|---:|---:|---:|---:|---:|
| 7 | 84.60% | 78.77% | 73.75% | 830.32 MB | 171.55 s |
| 13 | 88.59% | 88.41% | 84.05% | 809.23 MB | 171.59 s |
| 21 | 86.05% | 79.89% | 73.28% | 814.53 MB | 171.64 s |

### 5.2 三 seed 汇总

| 指标 | LEGEND |
|---|---:|
| 最终准确率 | **86.41% ± 2.02%** |
| 最后 5 轮平均准确率 | 82.36% |
| 全轮平均准确率 | 77.03% |
| 模拟时间 | 171.59 s |
| 通信量 | 818.03 MB |
| 达到 0.7 的 seed | 3/3 |

## 6. 与主实验的比较

| 方法 | 最终准确率 | 模拟时间 | 通信量 |
|---|---:|---:|---:|
| FedLoRA | 89.43% | 314.00 s | 1,353.52 MB |
| FedHeLLo | 88.41% | 183.89 s | 834.15 MB |
| PCLA | 87.02% | 100.19 s | 211.18 MB |
| LEGEND | 86.41% | 171.59 s | 818.03 MB |
| Random Component | 78.99% | 73.15 s | 211.18 MB |

在当前统一协议下：

- LEGEND 比 FedLoRA 低 3.02 个百分点，通信量低 39.56%；
- LEGEND 比 FedHeLLo 低 1.99 个百分点，通信量低 1.93%；
- PCLA 比 LEGEND 高 0.60 个百分点；
- PCLA 比 LEGEND 的通信量低 74.18%，模拟时间低 41.61%。

因此，加入 LEGEND 后，当前证据仍支持 PCLA 的主要结论：在强 non-IID 和部分参与条件下，PCLA 的组件重要性分配比资源驱动的深层连续分配更适合当前低通信预算。

## 7. 汇报边界

这组实验应称为：

> 基于论文 Algorithm 1 的 LEGEND 重实现，并迁移到与 PCLA 相同的 SST-2/BERT-base 协议。

不能称为：

- LEGEND 官方代码复现；
- 论文 80 台 Jetson 平台复现；
- 论文原始超参数和硬件结果的数值复现。

当前重现了：

- 靠近输出端的连续 LoRA depth；
- 公差为 1 的递增 rank distribution；
- 总 rank 与客户端资源约束；
- 基于完成时间的 LCD depth 计算；
- EMA 设备状态；
- 仅聚合实际贡献该层的客户端。

没有重现：

- 80 台 Jetson 的真实计算状态；
- Wi-Fi 带宽测量；
- 每 20 轮切换设备性能模式；
- RoBERTa/DeBERTa/Llama 的论文完整任务组合；
- 论文的 100 轮、batch size 4、cosine scheduler 等全部训练设置。

