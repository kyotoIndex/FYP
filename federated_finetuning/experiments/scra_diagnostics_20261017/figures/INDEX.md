# 阶段 D 图表索引

由 `make_figures.py` 生成（每次运行都会覆盖本文件）。上游：`docs/designs/roadmap_2026-10-18_amplified_restricted.md` 阶段 D；叙述见 `docs/designs/research_two_directions_2026-10-17.md` §4–§8。

## 复现

在仓库根目录创建并启用 Python 虚拟环境后安装绘图依赖：

```bash
python -m pip install -e '.[figures]'
python federated_finetuning/experiments/scra_diagnostics_20261017/figures/make_figures.py
```

本次环境：Python 3.12.14，matplotlib 3.10.9，numpy 2.2.6，中文字体 Arial Unicode MS（PDF 内嵌 TrueType）。脚本只读 `federated_finetuning/runs/` 与 `docs/designs/`，只写本文件夹。

## 口径

- **核心弱端**：非强端、且所在领域没有强端的 30 个客户端（由 `runs/scra_diagnostics_20261017_v1_vref/d0/identity.json` 的 `client_domains` 与 `strong_clients` 计算），即客户端 0, 1, 2 … 39。所有图（包括 20% 强端的运行）都用这一固定集合，与 SUMMARY 的 `core10` 一致。
- **后期准确率**：第 110、120、…、150 轮五次评估的核心弱端准确率均值；前期 / 中期 / 后期窗口为 10–50 / 60–100 / 110–150 轮。
- **轮间波动**：第 50 轮起相邻两次评估（间隔 10 轮）之间全员宏平均准确率变化的绝对值的均值。
- **配对差值**：同一 seed 的两种方法相减（seed 共享参与者日程与初始化）；95% 区间用 t(2) = 4.303。

## 图表

| 文件 | 内容 | 数据来源 |
|---|---|---|
| `fig1_core_curves.pdf/.png` | 核心弱端准确率随轮次变化：r8、vref、L2、amp。上行 seed 2040，下行 3 个 seed 均值（阴影为最小–最大）；右列放大第 50–150 轮，并标出后期均值。 | 曲线：`runs/headroom_strongfrac_20261017_v1/fedavg_r8_seed*/result.json`（history）；L2 / vref seed 2040：`runs/scra_diagnostics_20261017_v1{,_vref}/d0/d0_rounds.json`；其余：`runs/scra_batch_20261018/*/d0/d0_rounds.json`（eval） |
| `fig2_oscillation.pdf/.png` | 各方法轮间波动的横向条形图，按“参照 / 受限训练 + 服务器端修正 / 放大受限训练 / 第二批对照”分组；3-seed 方法画均值并叠加各 seed 点，单 seed 方法标注。 | 同上，另加 `runs/scra_diagnostics_20261017_v1_{rcs,fixedshrink,weaklr}` 与 `runs/scra_batch2_20261018/*/d0/d0_rounds.json` |
| `fig3_head_mechanism.pdf/.png` | 一步机制（每快照 24 次抽样，误差线为抽样间标准差）：(a) 分类头沿 r8 方向的步长，收缩 vs amp；(b) 冻结弱端头部时的 LoRA 乘积步长（压力转移）；(c) 头部互换分解：shrink、shrink_fullhead、vref_restrhead 的核心弱端准确率减 r8。 | (a)(b)：`runs/scra_diagnostics_20261017_v1/d3/round_00{20,60,120}.json` 的 `summary`；(c)：`d2/round_*.json` 的 `summary`（与 `rows` 逐次重算一致） |
| `fig4_ablation_robustness.pdf/.png` | (a) 批次 1 消融（seed 2040）相对 amp 的差值：α = 2 / 8、拥有者平均、前缀分配，竖线为 L2 与 vref；(b) 稳健性：基准、r = 1、rank 混合 {1,2,4}、20% 强端下 L2 / vref / amp 的后期准确率与追回比例。 | `runs/scra_batch_20261018/*/d0/d0_rounds.json`，vref r = 1 来自 `runs/scra_diagnostics_20261017_v1_budget8` |
| `fig5_batch2_decomposition.pdf/.png` | 批次 2 分解：(a) 四个机制对照的配对差值（条为均值，标记为各 seed，黑线为 95% t 区间）；(b) 三个小差值对照按前 / 中 / 后期的变化；(c) 外部基线的配对差值；(d) 各方法后期准确率。 | `runs/scra_batch2_20261018/*/d0/d0_rounds.json` 与批次 1 的 amp / vref / L2、sweep 的 r8 |
| `cost_table.md` | 每种方法与每次运行的峰值显存、墙钟时间、节点。 | 各运行 JSON 的 `peak_memory_mib` / `elapsed_seconds` 或 `cost` |
| `checks.md` | 全部数值核对项（本脚本 vs summary.json / SUMMARY.md / REPORT*.md / 叙述文档）。 | — |

## 各图要点

1. **图 1**：L2 在 seed 2040 上第 50 轮后反复摆动（全员准确率第 50 轮后最低 70.07），amp 与 r8 / vref 的曲线几乎重合。3-seed 后期均值：r8 80.19，vref 80.17，amp 80.12，L2 77.85；L2 的范围带在中后期明显更宽。冷启动（第 10 轮，seed 2040）amp 已与 r8 持平。
2. **图 2**：受限训练且不修正前向的方法都在摆动（L2 2.69、RCS 1.86、Fed-PLoRA 式 2.72），波动随收缩加强而下降但降不到参照水平（固定收缩 1.19）；amp 为 0.35，与 r8 0.34 相当；α = 2 时摆动部分回来（0.97），α = 8 时没有（0.36）。FSLoRA 系对照都平稳，差别在准确率而不在稳定性。
3. **图 3**：收缩规则的分类头步长是 r8 的 1.61–1.91 倍，amp 为 1.00–1.01。头部互换：给收缩规则换上完整训练的头部，差距从 -3.83 / -2.28 / -1.20 pp 缩小到 -1.24 / -0.45 / -0.43；反过来给 vref 换上受限头部，差距变为 -1.30 / -1.20 / -0.35。冻结弱端头部后压力转到 LoRA：LoRA 步长 2.27–2.49 倍（收缩 1.10–1.17，amp 0.99–1.00）。
4. **图 4**：α = 2 -0.58，α = 8 -0.14，拥有者平均 -0.45，前缀分配 -1.55 pp；三种稳健性设置的追回比例 1.01 / 0.89 / 0.92（基准 0.96）。单 seed，判据为 0.3 pp。
5. **图 5**：amp − A1a +0.42（状态 sketch 效应），A1a − A1b -0.11（eps 混淆可忽略），A1b − A1c +2.62（零填充聚合的代价），amp − A2 +0.76（前向放大不等于调学习率）；外部：FSLoRA 式 +2.95、Fed-PLoRA 式 +2.96、FLASC 式 +0.79。分阶段看，amp − A1a 从前期起就为正（+0.34 / +0.39 / +0.42），amp − A2 前期最大（+1.18 / +0.79 / +0.76）。
6. **成本表**：见 `cost_table.md`；所有运行峰值显存约 1557 MiB，RoBERTa-base 上受限训练不省显存。

## 数值核对与不一致

共 712 项核对，通过 712 项（明细见 `checks.md`）。核对对象：批次 1 / 2 的 `summary.json`（精确）与 `SUMMARY.md`（逐格、按印出精度），D2 / D3 的 `rows` 重算与 `REPORT_D2.md` / `REPORT_D3.md`，早期判别运行各自的 `REPORT.md` 与叙述文档 §4 表格，叙述文档 §7 中第 10 轮的数字。

**没有数值不一致。** 图中每个数字都能在对应的 SUMMARY / REPORT 中按印出精度找到。

**口径差异与说明：**

- batch 1 SUMMARY.md 的 “mean ± sd” 用总体标准差（ddof=0，6/6 行吻合；ddof=1 吻合 0/6 行），batch 2 SUMMARY.md 用样本标准差（ddof=1）。同一组数据因此在两份汇总中写作 80.19 ± 0.06 与 80.19 ± 0.07（r8），80.12 ± 0.14 与 80.12 ± 0.17（amp），L2 0.53 与 0.65；research_two_directions §7 沿用前者，§8 沿用后者。图中不画标准差（用各 seed 点与最小–最大范围）。
- D2 与 D3 是两次独立重放，都包含 shrink 与 shrink_fullhead，结果略有差异（r8 基线本身也不完全相同，三个快照的 r8 准确率 D2 / D3 为 74.42 / 74.47、78.80 / 78.86、80.86 / 80.92）：第 20 轮 shrink D2 -3.83 / D3 -3.76；第 20 轮 shrink_fullhead D2 -1.24 / D3 -1.27；第 60 轮 shrink D2 -2.28 / D3 -2.34；第 60 轮 shrink_fullhead D2 -0.45 / D3 -0.44；第 120 轮 shrink D2 -1.20 / D3 -1.08；第 120 轮 shrink_fullhead D2 -0.43 / D3 -0.38 pp；shrink 的头部步长 D2 / D3 为 1.93 / 1.91、1.64 / 1.66、1.67 / 1.61。图 3(c) 三条规则都取 D2（vref_restrhead 只在 D2 中），(a)(b) 都取 D3（amp 与 fh_hajek 只在 D3 中），因此同一面板内的规则来自同一次重放。
- 批次 1 `SUMMARY.md` 与 `summary.json` 中 r8 / r2 / 前缀异构（sweep 运行）的 `elapsed_seconds` 为空，因为 sweep 的 `result.json` 把时间放在 `cost` 字段；成本表从 `cost.elapsed_seconds` 补上。批次 2 `SUMMARY.md` 没有成本表，批次 2 的成本直接取各 `d0_rounds.json`，没有可对照的汇总。
- 批次 1 消融表中 amp 的波动 0.37 是 seed 2040 单次值，主表的 0.35 是 3-seed 均值；图 2 用 3-seed 均值。
- 图 3 的误差线是 24 次抽样之间的标准差（REPORT 中的 “± std”），反映参与者抽样的离散程度，不是均值的标准误。
- 图 1 从第 10 轮画起（第 0 轮所有方法都是 33.33%，会压缩纵轴）。
