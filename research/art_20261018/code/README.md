# A-SCRA 训练前诊断 D0 / D1 `scra_diagnostics_20261017_v1`

2026-10-17 创建，同日按[批判性审视](../../../docs/designs/critical_review_2026-10-17_ascra.md)修订。状态：**代码完成，CPU 测试 8 项通过，尚未在 HPC 上运行。**

这些诊断检验的是 A-SCRA **依赖的前提在真实 RoBERTa + AdamW 上是否成立**，不评价方法本身好坏。判据全部来自设计稿：[algorithm_design_2026-10-17_sampling_calibrated_rank_aggregation.md](../../../docs/designs/algorithm_design_2026-10-17_sampling_calibrated_rank_aggregation.md)，见 §5.6、§5.7、§6、§7.1。

## 设置

与强端比例扫描相同：MNLI，RoBERTa-base，seed2026 数据划分，50 个客户端，冷启动，150 轮，每轮 10 个客户端，每端 20 步，AdamW lr 2e-4，R=8，r=2，缩放 2。

- **10% 强端**：客户端 10、11、12、40、41。
- **训练种子**：2040。每轮参与者与 headroom 实验完全相同。

## D0：一次带日志的联邦训练（约 15–20 分钟）

**训练规则：** 覆盖均衡分配 + 拥有者平均，即设计稿消融阶梯中的 L2。

> 与设计稿原文的差别：原文写的是 Fed-PLoRA 独立随机分配。这里改用覆盖均衡分配，是为了保证每一轮、每个块都有弱端贡献者，使 o = D − m 每轮都能计算；同时它也是 A-SCRA 实际使用的分配方式。

**每轮额外计算但不参与训练的量：**
- Hájek 聚合结果，以及它与拥有者平均之差；
- 每个块上的 o = 强端均值 − 弱端均值；
- 弱端组内离散度，以及两种估计量按设计计算的方差；
- **A-SCRA 影子量**：每个强端从同一起点、用同一批数据，对每个模块随机选 2 个分量再训练一次，得到 m̂；据此计算在线 κ 和 A-SCRA 本轮会选择的 a。D0 中 a 只记录，不使用。

**快照：** 在第 20、60、120 轮保存全局状态，供 D1 使用。

## D1：固定状态的对比训练（每个快照约 5 分钟，3 个快照并行）

**选取的客户端：**
- 全部 5 个强端；
- 非强端领域（fiction、slate、telephone）每个领域取 3 个弱端；
- 强端领域（government、travel）中被降级为弱端的客户端，每个领域取 1 个。

**每个客户端用同一批数据跑：**
- 1 次"全部训练"；
- 28 次"两分量训练"。采用拉丁方设计：每个模块都恰好经历全部 28 种两分量组合各一次，同一次运行中不同模块的组合不同，与真实训练一致。

**计算的量：** 数据差异 D、训练方式效应 m（弱端，即 oracle）与 m̂（强端，即 A-SCRA 实际能估计到的）、两种估计量的误差、κ 与 a*。弱端组按总体中的比例加权：核心弱端 30/45，降级弱端 15/45。

## 判据（自动写入报告）

| 来源 | 量 | 判据 | 对应设计稿 |
|---|---|---|---|
| D0 | 拥有者平均与 SCRA 之间系统性差距的相对大小 | ≥ 0.05 | F2 |
| D0 | o 在相邻 10 轮窗口之间均值的余弦 | ≥ 0.3 | F3 |
| D0 | 校准占优的交叉轮数 K* | ≤ 50；≤ 0 表示没有权衡，Hájek 方差反而更小 | F3 |
| D0 | 弱端离散度在同一模块各分量间的 CV² | ≥ 0.1 时才考虑 Neyman（E1） | E1 |
| D0 | 后 1/3 轮与前 1/3 轮更新范数之比 | ≤ 3 | F9（本阈值为代码中新设，设计稿未给数值） |
| D1 | ‖e_SCRA‖ < ‖e_owner‖ | 成立则 SCRA 方向占优 | F1 / 5.6 节 |
| D1 | 弱端 m 与强端 m̂ 的余弦 | ≥ 0.5 | F11 / A4 |
| D1 | 由弱端 κ 与强端 κ 分别推出的 a 之差 | ≤ 0.2 | F11 / A4 |

审视后新增的报告项：
- 修正后的 κ：先在 10 轮窗口内平均向量，再做内积；原来的逐轮 κ 有自相关偏差（审视 C1）；
- D0（即 L2）在固定弱端上的后期准确率，与强端扫描中同一 seed 的全员 r8、全员 r2、前缀 L0 对比，以及追回比例（审视 C5，需要 `runs/headroom_strongfrac_20261017_v1` 也在 HPC 上）；
- D1 的乘积空间误差与 κ（审视 C6）。

另外会报告：相对训练方式效应（弱端、强端）、‖D_core‖ 对 ‖D_demoted‖（领域效应是否存在）、在线 κ 与 a 的轨迹、准确率曲线。

## HPC 提交步骤

1. 同步本目录到 HPC 的相同位置（`<hpc-login>` 换成你平时的登录地址）：

```bash
rsync -av --exclude '__pycache__' /Users/liming/Documents/FYP/federated_finetuning/experiments/scra_diagnostics_20261017/ <hpc-login>:/home/student/t330025090/FYP/federated_finetuning/experiments/scra_diagnostics_20261017/
```

2. 先提交 D0，再提交依赖于 D0 成功完成的 D1：

```bash
cd /home/student/t330025090/FYP/federated_finetuning/experiments/scra_diagnostics_20261017 && mkdir -p logs && jid=$(sbatch --parsable run_d0.sbatch) && sbatch --dependency=afterok:$jid run_d1.sbatch
```

   **方差匹配参照**（审视 C4，可与 D0 同时提交，约 15 分钟）：所有客户端都完整训练，服务器只采用与 A-SCRA 相同的分量贡献，并做 Hájek 校准，得到"任何异构方案在期望上能达到的最好结果"：

```bash
cd /home/student/t330025090/FYP/federated_finetuning/experiments/scra_diagnostics_20261017 && SCRA_PLAN=plan_vref.json sbatch run_d0.sbatch
```

3. 全部完成后生成报告。只用 CPU，几秒钟，可以在登录节点运行：

```bash
/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python run.py analyze
```

   方差匹配参照的报告（只有 D0 部分）：

```bash
/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python run.py analyze --plan plan_vref.json
```

4. 中途失败时，重新提交即可。D0 会从 checkpoint（每 5 轮保存一次）续跑；D1 已完成的快照会自动跳过：

```bash
sbatch run_d0.sbatch
```

## 需要带回本地的文件

把整个 `runs/scra_diagnostics_20261017_v1/`（以及 `runs/scra_diagnostics_20261017_v1_vref/`）目录下载回来，可以排除 `d0/checkpoint.pt` 和 `d0/final_state.pt`。其中：

- `diagnostics_report/REPORT.md`、`report.json`：判据汇总；
- `d0/d0_rounds.json`：每轮的标量与准确率曲线；
- `d0/d0_stats.pt`：逐块累积的统计量，约 25MB；
- `d1/round_*.pt`：分组均值与逐客户端标量，每个约 30MB；
- 再加上 `logs/`。

## 文件

```text
plan.json            全部参数与判据阈值
run.py               入口：d0 / d1 / analyze
run_d0.sbatch        单 GPU，默认续跑
run_d1.sbatch        数组 0-2，对应第 20 / 60 / 120 轮快照
scra/core.py         覆盖均衡分配、块向量、按分量屏蔽训练、拥有者 / Hájek / 混合聚合、评测
scra/d0.py           D0 训练与逐块统计、A-SCRA 影子量
scra/d1.py           D1 固定状态的对比训练
scra/analyze.py      判据计算与报告
scra/data.py, model.py   复制自 headroom 实验（含嵌套强端集合）
tests/test_scra.py   微型随机 RoBERTa + 合成数据的 CPU 测试
```

## 本地验证（已通过）

```bash
cd /Users/liming/Documents/FYP/federated_finetuning/experiments/scra_diagnostics_20261017
../../.venv/bin/python -m pytest -q tests -p no:cacheprovider
```

7 项测试：
1. 覆盖均衡发牌：同一客户端不重复，计数正确，每个分量至少 2 个贡献者；
2. 包含概率的经验频率等于 r/R；
3. 蒙特卡洛：拥有者平均的偏差符合命题 1，Hájek 的偏差约为 0；
4. 块向量与状态互相转换逐位一致；按分量屏蔽后，只有指定分量发生变化；
5. **前缀分配 + 拥有者平均时，D0 与已完成的 headroom 朴素异构实现一致**：最大差 1.6×10⁻⁷，而训练中权重变化约 0.05；
6. D0 中断后续跑，与连续运行的最终权重和全部统计量逐位相同；
7. D0 → D1 → analyze 全流程可以运行，误差分解恒等式的数值误差 < 10⁻⁶。

这些测试只用随机微型模型和合成数据，**不是 MNLI、CUDA 或效果证据**。

## 已知局限

- D0 只有 1 个 seed，诊断量本身也有抽样误差；
- D1 的 D 和 m 是在 3 个固定状态上测的，训练过程中会变化；
- "分块拼接不稳定"（F9）在 AdamW 下的表现形式未知，D0 只能观察拥有者平均训练时的趋势。SCRA（a=1）本身是否稳定，要到正式消融时才能看到。

---

## 判别实验（2026-10-17 晚）：RCS、弱端小学习率、通信预算 1/8

背景与动机：[research_two_directions_2026-10-17.md](../../../docs/designs/research_two_directions_2026-10-17.md)。四个实验都使用 seed 2040、10% 强端，参与日程与 D0 / vref / 强端扫描完全相同，可以直接对比。

| 配置文件 | 实验 | 弱端训练 | 收缩 | 聚合 | 回答的问题 |
|---|---|---|---|---|---|
| `plan_rcs.json` | **RCS 在线版** | 2 个分量（受限） | 按模块在线估计 c_j（强端重放，向量 EMA β=0.9，先平均向量再求比值，c ∈ [0.2, 1]） | Hájek | 拼接过冲的修正能否转化为准确率，接近 vref |
| `plan_weaklr.json` | 弱端小学习率 | 2 个分量，lr × 0.6 | 无 | Hájek | 简单的固定学习率缩放能否达到同样效果 |
| `plan_budget8.json` | 通信预算 1/8 | 全部 8 个分量 | 无 | Hájek，每个弱端只上传 1 个分量 | 方向一在更低预算下是否开始掉点 |
| `plan_fixedshrink.json`（可选） | 固定收缩 | 2 个分量 | 固定 0.63 | Hájek | 与 RCS 唯一的区别是收缩不自适应；也做强端重放，计算量与 RCS 相同 |

0.6 和 0.63 都取自 D1 中三个快照的收缩系数（0.56 / 0.62 / 0.70），在运行前写定，没有按准确率调。

### 提交

1. 同步整个目录（新增了配置文件，`core.py`、`d0.py`、`analyze.py` 也有修改）：

```bash
rsync -av --exclude '__pycache__' --exclude 'logs' /Users/liming/Documents/FYP/federated_finetuning/experiments/scra_diagnostics_20261017/ <hpc-login>:/home/student/t330025090/FYP/federated_finetuning/experiments/scra_diagnostics_20261017/
```

2. 一次提交四个（每个约 15 分钟，各占 1 张 GPU）：

```bash
cd /home/student/t330025090/FYP/federated_finetuning/experiments/scra_diagnostics_20261017 && mkdir -p logs && for p in rcs weaklr budget8 fixedshrink; do SCRA_PLAN=plan_$p.json sbatch --job-name=scra-$p run_d0.sbatch; done
```

3. 全部完成后逐个生成报告：

```bash
for p in rcs weaklr budget8 fixedshrink; do /home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python run.py analyze --plan plan_$p.json; done
```

4. 下载 `runs/scra_diagnostics_20261017_v1_rcs`、`_weaklr`、`_budget8`、`_fixedshrink` 四个目录，可以排除 `d0/checkpoint.pt` 和 `d0/final_state.pt`。

### 怎么读结果

每个报告的准确率表都会列出 r8、r2、L0、L2、vref 和本次运行，并给出一行 **"This run recovers X% of the vref − L2 core-weak gap"**：0% 等于 L2，100% 等于 vref。

| 结果 | 含义 |
|---|---|
| RCS 追回 ≥ 60%，且明显高于 weaklr 与 fixedshrink | 自适应收缩有效，方向二有主方法 |
| RCS ≈ fixedshrink ≈ weaklr，且都很高 | 有效的是"收缩"，不是"自适应"；方法退化为简单的学习率调整 |
| RCS 与 fixedshrink 都很低 | 固定状态下的误差降低没有转化为准确率，方向二的这个思路不成立 |
| budget8 ≈ vref | 方向一在 1/8 预算下仍然没有损失，需要更低的预算才能看到方法的作用 |
| budget8 明显低于 vref | 方向一找到了会掉点的预算区间，可以在这里比较不同的上传策略 |

只有 1 个 seed，单 seed 噪声约 0.2–0.4pp；差距明显的结论才可靠，接近的需要补 seed。

另外，RCS 的报告会在逐轮日志里记录 `shrink_mean / min / max`，可以看收缩系数随训练的变化，与 D1 的 0.56 → 0.70 对照。

## D2：单轮波动诊断（2026-10-17 夜）

**要回答的问题：** 受限训练造成的差距，是不是"每一轮拼接带来的波动"，而不是期望意义上的偏差？如果是，这个波动出在 LoRA 分量上，还是出在分类头上？

**做法：**
- 从 L2 主运行（`runs/scra_diagnostics_20261017_v1`）第 20、60、120 轮的快照出发，各模拟 24 个轮次。参与者和分量分配都用快照之后真实轮次的日程。
- 每个参与者在同一批数据上同时做完整训练和受限训练。
- 用 7 种规则各更新**一步**（不继续训练），然后评估：
  - r8：全员完整训练，平均全部分量；
  - vref：完整训练，弱端只贡献分配到的分量，Hájek；
  - L2：受限训练，拥有者平均；
  - hajek：受限训练，Hájek；
  - shrink：受限训练乘以 0.63 后 Hájek；
  - shrink_fullhead / vref_restrhead：把上面两种规则的分类头互换，用来判断问题是否出在头部。
- 不训练联邦模型，只训练本地、评估。不需要新的 D0。

**提交（在实验目录下）：**
```bash
sbatch run_d2.sbatch
```
这是一个 3 个任务的数组（3 个快照），每个约 30 分钟，需要 `runs/scra_diagnostics_20261017_v1/d0/snapshots/` 已经在 HPC 上。跑完后生成报告（CPU，几秒）：
```bash
/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python run.py d2report --plan plan_d2.json
```
**下载：** `runs/scra_diagnostics_20261017_v1/d2/`（3 个 json，较小）和 `runs/scra_diagnostics_20261017_v1/diagnostics_report/REPORT_D2.md`。

**事先写定的判读（3 个快照中至少 2 个满足）：**

| 结果 | 判据 | 下一步 |
|---|---|---|
| 差距来自单轮波动 | 一步后的准确率（与同一轮的 r8 相减）离散度：shrink ≥ 2 × vref，且 LoRA 乘积空间配对误差 shrink ≥ 2 × vref | 修正受限分量的本地训练动力学 |
| 单轮波动不是原因 | 准确率离散度比 < 1.5 | 先做多轮滚动诊断，再谈方法 |
| 期望偏差不是主要项 | shrink 的 LoRA 偏差² / 方差 < 1 | 确认服务器端缩放已到上限 |
| 问题出在头部 | shrink_fullhead / shrink ≤ 0.6，或 vref_restrhead / vref ≥ 2 | 改头部的聚合方式（成本更低） |
| 问题出在 LoRA | shrink_fullhead / shrink ≥ 0.8，且 vref_restrhead / vref < 1.5 | 同第一行 |

## D3：可实现修正的离线筛选（2026-10-18）

**为什么做：** D2 显示，受限训练造成的单步损失主要出在分类头上。在受限 LoRA 旁边训练出来的头，沿完整训练方向走了 1.64–1.93 倍，而服务器做普通平均时这部分不会抵消。D2 用来定位问题的"完整训练得到的头"在实际中拿不到，D3 只比较**部署时真正能用**的规则。

**做法：** 与 D2 相同：同样的快照，用之后真实轮次的日程，只更新一步，与同一轮的 r8 配对相减。每一轮的本地训练如下：
- 强端：完整训练一次，再从同一起点重放一次受限训练（每个模块随机取一对分量）。
- 弱端：完整训练（只作参照）、受限训练、冻结头部的受限训练、放大前向的受限训练（α = R/r = 4），共四次。

| 规则 | 内容 | 能否部署 |
|---|---|---|
| r8、vref、shrink、shrink_fullhead | 与 D2 相同的参照 | 只作参照 |
| shrink_h40 / h55 / h70 | LoRA ×0.63，弱端头部位移再乘 0.40 / 0.55 / 0.70 | 能（固定常数） |
| shrink_hstrong / shrink_hnone | 头部只取强端 / 这一轮不更新头部 | 能 |
| hajek_h55 | LoRA 不收缩，只把弱端头部乘 0.55，检验只修头部够不够 | 能 |
| fh_hajek / fh_shrink | 弱端训练时冻结头部，头部只取强端 | 能 |
| **amp_hajek** | **放大前向的受限训练：前向用"起点 + 4 × LoRA 变化"，上传原始变化；Hájek；头部普通平均；没有任何服务器端常数** | 能 |
| amp_fullhead | 同上的 LoRA + 完整训练得到的头，单独看 LoRA 部分 | 只作诊断 |
| **rcs2** | LoRA 每个模块一个系数、头部一个系数，都用强端重放在 24 轮内累积估计（模拟在线估计） | 能 |
| oracle2 | 同上，但系数用弱端自己的完整训练算，即标量校准的理想上界 | 只作参照 |

放大前向的直觉：受限训练时，每一步只有 2/8 的分量在动，函数变化慢，残差一直存在，头部和这 2 个分量就被迫多承担。把前向中 LoRA 的变化放大 R/r 倍后，本地训练的函数变化速度接近完整训练。在 Adam 下参数步长不随梯度缩放，所以上传的原始位移会接近完整训练时每个分量的位移。实现里梯度用的是放大模型真实梯度的 1/α，方向相同，梯度裁剪看到的尺度也和普通训练一致。

**提交（在实验目录下）：**
```bash
sbatch run_d3.sbatch
```
这是一个 3 个任务的数组，每个约 50–70 分钟（本地训练次数是 D2 的两倍），需要 HPC 上已有 `runs/scra_diagnostics_20261017_v1/d0/snapshots/`。跑完后生成报告（CPU，几秒）：
```bash
/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python run.py d3report --plan plan_d3.json
```
**下载：** `runs/scra_diagnostics_20261017_v1/d3/`（3 个 json）和 `runs/scra_diagnostics_20261017_v1/diagnostics_report/REPORT_D3.md`。

**事先写定的判读（3 个快照中至少 2 个满足）：** closure 指"从 shrink 到 vref 的核心弱端差距被追回的比例"。

| 问题 | 判据 | 结果为真时的下一步 |
|---|---|---|
| 放大前向不靠常数就能修好 | amp_hajek closure ≥ 0.6 | 主方法候选为放大受限训练，跑完整 D0 确认 |
| 放大不比校准差 | amp_hajek ≥ rcs2 − 0.2pp | 同上 |
| 强端重放校准可以实现 | rcs2 closure ≥ 0.6，且与 oracle2 相差 ≤ 0.3pp | 放大不成立时，用 RCS v2（头部 + LoRA 校准）跑完整 D0 |
| 头部乘一个常数就够 | shrink_h40/55/70 中最好的 closure ≥ 0.6 | 若 rcs2 不成立，说明系数从强端迁移到弱端有问题，先检查系数 |
| 弱端的头部带有有用信息 | closure(hstrong) ≤ 最好的头部常数 − 0.2 | 说明不能简单丢弃弱端头部（FedBABU 式冻结不是答案） |

如果没有任何规则的 closure ≥ 0.6，说明"一步"这个视角漏掉了机制，在设计方法前要重新思考。

## 批量验证：放大受限训练（2026-10-18）

**为什么做：** D3 中，放大前向的受限训练在三个快照上都完全追平了 vref，但那只是"更新一步"。这一批用完整的 150 轮训练来验证它，并一次性跑完消融和稳健性检验。判读标准在下方，提交前已经写定。

**方法（`weak_amp_alpha: "auto"`）：** 弱端仍然只训练分配到的 r 个分量。训练时，前向用的是"起点 + (R/r) × LoRA 变化"，上传的是原始变化；服务器做 Hájek 聚合，没有任何常数。如果弱端 rank 不同（`lora.weak_rank_choices`），每个客户端的 rank 固定，α_i = R/r_i，用覆盖均衡的异构发牌，Hájek 按每个客户端的包含概率 r_i/R 加权。

| # | 运行 | seed | 内容 |
|---|---|---|---|
| 00–02 | amp_s2040/2041/2042 | 3 个 | **主方法**（10% 强端，R=8，r=2） |
| 03–06 | L2_s2041/2042、vref_s2041/2042 | 2 个 | 补齐多 seed 的下界和理想参照（seed 2040 已有；r8、r2、前缀已有 3 个 seed） |
| 07 | amp_owner_s2040 | 2040 | 消融：拥有者平均代替 Hájek |
| 08 | amp_prefix_s2040 | 2040 | 消融：所有弱端都训练第 0、1 个分量（不做覆盖均衡） |
| 09–10 | amp_a2 / amp_a8 | 2040 | 消融：α = 2 / 8，检验 α = R/r = 4 是否合适 |
| 11–12 | amp_r1、L2_r1 | 2040 | 稳健性：r = 1（上传 1/8；vref 参照用已有的 budget8） |
| 13–15 | amp_het、vref_het、L2_het | 2040 | 稳健性：弱端 rank 混合 {1, 2, 4} |
| 16–18 | amp_sf20、vref_sf20、L2_sf20 | 2040 | 稳健性：20% 强端 |

共 19 次运行。每次约 15 分钟，`--array=0-18%4` 同时跑 4 个，总共约 1.3 小时。顺序即优先级：主比较排在最前面。

**提交前同步（整个实验目录，代码有更新）：**
```bash
rsync -av --exclude '__pycache__' --exclude 'logs' /Users/liming/Documents/FYP/federated_finetuning/experiments/scra_diagnostics_20261017/ <hpc-login>:/home/student/t330025090/FYP/federated_finetuning/experiments/scra_diagnostics_20261017/
```
**提交（在实验目录下）：**
```bash
sbatch run_batch.sbatch
```
某个任务中断后，重新提交同一个任务会从检查点继续，例如 `sbatch --array=5 run_batch.sbatch`。

**全部跑完后生成汇总（CPU，几秒）：**
```bash
/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python run.py batchreport --plan plans_batch/00_amp_s2040.json
```
结果写到 `runs/scra_batch_20261018/SUMMARY.md` 和 `summary.json`。

**下载（只取小文件；检查点和快照都很大，不需要下载）：**
```bash
rsync -av --include '*/' --include 'd0_rounds.json' --include 'identity.json' --include 'SUMMARY.md' --include 'summary.json' --exclude '*' <hpc-login>:/home/student/t330025090/FYP/federated_finetuning/runs/scra_batch_20261018/ /Users/liming/Documents/FYP/federated_finetuning/runs/scra_batch_20261018/
```

**事先写定的判读**（指标：核心弱端第 110–150 轮的平均准确率；单 seed 噪声约 0.2–0.4pp）：

| 问题 | 判据 | 含义 |
|---|---|---|
| M1 主方法有效 | 3 个 seed 的均值 ≥ vref − 0.3pp，且 ≥ L2 + 1.0pp，且每个 seed 都高于 L2 | 放大受限训练成为主方法 |
| M2 稳定 | 轮间波动的均值 ≤ 1.5 × vref | 消除了受限训练的摆动 |
| A1 Hájek 是否必要 | amp_owner ≤ amp − 0.3pp | 不成立就从方法中去掉 Hájek |
| A2 覆盖均衡是否必要 | amp_prefix ≤ amp − 0.3pp | 不成立就去掉覆盖均衡分配 |
| A3 α = R/r 是否合适 | amp ≥ max(α=2, α=8) − 0.3pp | 不成立说明倍数需要重新推导 |
| R1 / R2 / R3 稳健性 | 在 r=1、异构 rank、20% 强端下，追回 L2 → vref 差距的 ≥ 70%（差距 < 0.5pp 时记为"无空间"） | 方法能否推广 |

M1 和 M2 都成立后，下一步是：与 FLASC、FSLoRA、Fed-PLoRA 比较，做文献核对，以及考虑扩展到更大模型。如果 D3 通过而 M1 不成立，说明"只看一步"漏掉了多轮效应，要先用 D3 的方法检查放大训练过程中的状态，再考虑修改方法。

## 第二批：定位对照（2026-10-18，路线图阶段 A；已按 FIX_BATCH2_CONTROLS_2026-10-18.md 修订）

**为什么做：** 第一批确认了放大受限训练（amp）有效。文献核对发现最接近的是 FSLoRA。这一批回答三个问题：
- 和 FSLoRA 的本质区别（只放大增量、不对状态做 sketch）是否有实际作用；
- 这是不是就等于调大学习率；
- 和已发表方法的风格相比怎样。

参照（amp、vref、L2、r8 的 3 个 seed）沿用第一批，不重跑。

**机制对照（每一对只差一个因素）：**

| # | 运行 | 内容 | 与谁比较、隔离什么 |
|---|---|---|---|
| 00–02 | fslora_st | FSLoRA 的前向值（被选分量的整个乘积 × R/r，未选分量不参与），**梯度路径与 amp 相同**；覆盖均衡 + Hájek | amp vs 它：只差"是否对状态做 sketch"（A1a） |
| 03–05 | fslora_hajek | FSLoRA 前向，真实梯度（梯度大 R/r 倍）；覆盖均衡 + Hájek | fslora_st vs 它：梯度尺度与 Adam eps 的影响（A1b） |
| 06–08 | fslora_zerofill | FSLoRA 前向，真实梯度；覆盖均衡 + 零填充平均 | fslora_hajek vs 它：聚合方式（A1c） |
| 09–11 | lrcomp | 受限分量的 LoRA 学习率 × R/r（头部不变），服务器把弱端 LoRA 位移 ÷ R/r | amp vs 它：是否等价于调学习率（A2，一阶相同） |

**外部定位（多个因素同时不同，不能作为机制证据）：**

| # | 运行 | 内容 |
|---|---|---|
| 12–14 | fslora_paper | FSLoRA 式，**在我们统一的 AdamW 协议下**（原论文本地用 SGD）：独立随机选分量，零填充平均 |
| 15–17 | fedplora | Fed-PLoRA 式：独立随机选分量，拥有者平均，不放大 |
| 18–20 | flasc | FLASC 式：弱端完整训练，按幅值上传**全局**前 r/R 的 LoRA 元素，零填充平均 |

需要说明的几点：
- **冷启动等价性只是近似。** 当 B = 0 时，FSLoRA 的前向和 amp 只在 Adam eps → 0 的极限下相同。用测试中的小模型、eps = 1e-8 走一步，两者的更新相差 12.4%（eps = 1e-12 时为 0.01%），与梯度裁剪无关。fslora_st 就是为了去掉这个混淆。单元测试确认，冷启动时 fslora_st 和 amp 的整个本地训练完全一致。
- **FLASC 的预算是"异构预算"。** 强端全量上传，弱端上传 r/R，与 amp 每个客户端的预算一致，不是"所有客户端同预算"。Top-K 在全部适配器上全局选取；稀疏编码的索引开销没有计入。FLASC 运行中，owner / Hájek 诊断字段把所有块都当作已上传，没有意义，训练只用零填充平均。
- FSLoRA 按论文公式实现，没有用它的公开代码。
- 所有方法都用统一设置：AdamW、梯度裁剪 1.0、FedAvg 服务器；FLASC 只做上传稀疏。
- 每次运行的 `identity.json` 都记录源代码哈希（`code_hashes`）。

共 21 次运行，每次约 15 分钟，同时跑 2 个，总共约 2.5–3 小时。

**同步（代码有更新）后，先跑一个冒烟测试：**
```bash
sbatch --array=0 run_batch2.sbatch
```
确认 `logs/batch2-*_0.out` 正常输出每 10 轮的准确率之后，再提交其余任务：
```bash
sbatch --array=1-20%2 run_batch2.sbatch
```
**跑完后生成汇总：**
```bash
/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python run.py batch2report --plan plans_batch2/00_fslora_st_s2040.json
```
**下载（只取小文件）：**
```bash
rsync -av --include '*/' --include 'd0_rounds.json' --include 'identity.json' --include 'SUMMARY.md' --include 'summary.json' --exclude '*' <hpc-login>:/home/student/t330025090/FYP/federated_finetuning/runs/scra_batch2_20261018/ /Users/liming/Documents/FYP/federated_finetuning/runs/scra_batch2_20261018/
```

**事先写定的判读**（核心弱端第 110–150 轮）。同一 seed 的参与者日程和初始化相同，所以按 seed 配对相减。报告列出每个 seed 的差值、均值 ± 标准差和 95% t 区间（3 个 seed 时区间很宽，t = 4.30），以及第 10–50 / 60–100 轮的分段差值。判读以"均值超过阈值且所有 seed 同向"为准：

| 比较 | 判据 | 成立时 | 不成立时 |
|---|---|---|---|
| A1a amp − fslora_st | 均值 ≥ 0.3pp 且 3 个 seed 都 > 0 | "只放大增量、不对状态做 sketch"的主张成立 | 主张缩小为"对 FSLoRA 式放大的新解释（Adam 下进展匹配、头部与分量过冲）及诊断" |
| A1b fslora_st − fslora_hajek | \|均值\| ≥ 0.3pp 且同号 | 真实梯度这一实现在 AdamW 下本身会改变结果，作为实现效应报告，不归为状态 sketch | — |
| A1c fslora_hajek − fslora_zerofill | 均值 ≥ 0.3pp 且 3 个 seed 都 > 0 | Hájek 对 FSLoRA 式训练同样有帮助 | — |
| A2 amp − lrcomp | \|均值\| < 0.3pp | 把方法表述为"让本地进展与完整训练匹配"，前向放大只是一种实现 | 前向形式本身有作用，报告哪一方更好 |
| E1 amp − fslora_paper | 均值 ≥ 0.3pp 且同向 | 外部定位 | — |
| E2 amp − fedplora | 均值 ≥ 1.0pp 且同向 | 外部定位 | — |
| E3 amp − flasc | 均值 ≥ −0.3pp | amp 只训练 r 个分量，却不差于完整训练的 FLASC 式方法 | — |
