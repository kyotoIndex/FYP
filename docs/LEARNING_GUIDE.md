# 学习指南：从 LoRA 到 AmpLoRA 实验

本仓库的当前主线是 **AmpLoRA**：异构 rank 联邦 LoRA 中，用本地前向放大改善弱客户端的受限训练，再按实验设置选择覆盖聚合。历史记录中的 ART、amplified restricted training 和 `amp` 对应这一主线。建议先理解一个客户端的一步，再理解服务器的一轮，最后阅读多 seed 结果。

先用 [METHOD](METHOD.md) 查精确定义，用 [RESULTS](RESULTS.md) 查正式结果，用 [PROVENANCE](PROVENANCE.md) 追溯结果来源。准备运行时按照 [REPRODUCE](REPRODUCE.md) 设置环境、模型和数据。

## 1. 先理解 LoRA 的一个组件

读 [`scra/model.py`](../federated_finetuning/experiments/scra_diagnostics_20261017/scra/model.py) 的 `LoRALinear`：

$$
W=W^0+sBA=W^0+s\sum_{k=1}^{R}b_ka_k^\top.
$$

把 $A$ 的第 $k$ 行和 $B$ 的第 $k$ 列合在一起看成一个组件。全局模型保留 $R$ 个组件，弱客户端只改变 $r$ 个组件；LoRA 自身的固定 `scale` 与本地放大因子 $R/r$ 是两个量。

手算：令输入为标量 $x=1$、$s=1$，两个组件乘积分别为 2 和 3，适配器输出为 5。弱端只选中第一个组件，训练后它的乘积为 2.1。普通受限前向为 5.1；当 $R/r=2$，AmpLoRA 前向为 $5+2(5.1-5)=5.2$。本轮开始时两者都为 5，未选中的第二个组件仍提供 3。

对应练习：找到 `lora_a`、`lora_b` 的形状，确认它们在 `core.to_blocks` 中如何组成 `[J, R, d_out+d_in]`。

## 2. 跟踪一次本地训练

读 [`core.local_train` 与 `_local_steps`](../federated_finetuning/experiments/scra_diagnostics_20261017/scra/core.py)，再回到 `LoRALinear.lora_update` 的 `amp` 分支。按以下顺序跟踪：加载本轮全局状态 → 保存起点 → 创建新的 AdamW → 放大前向 → backward → mask 未选组件梯度 → clip → optimizer step → 返回原始终点状态 → 清除临时放大设置。

关键实现只有一行：

```python
update = update + (alpha - 1.0) * (update - start).detach()
```

前向值是 $B_0A_0+\alpha(BA-B_0A_0)$，额外放大项不参与反传。它放大的是本轮 **乘积增量**，起点和未选组件保留；服务器采用的则是原始 $\Delta A,\Delta B$。

继续上一个例子：让选中组件 $a_0=1,b_0=2$，终点为 $a=1.02,b=2.06$，乘积从 2 变为 2.1012。放大前向的总值为 $5+2\times0.1012=5.2024$；上传的该组件位移是 $(\Delta b,\Delta a)=(0.06,0.02)$。服务器随后在因子空间聚合这些位移。

动机详读 [METHOD §2](METHOD.md#2-本地前向只放大本轮乘积增量)。Adam 的近似尺度不变性解释了为什么单纯放大梯度与前向补偿有不同作用；实验还直接比较了前向放大和学习率补偿。

## 3. 再理解覆盖分配与服务器聚合

读 [`core.balanced_allocation` 与 `block_aggregates`](../federated_finetuning/experiments/scra_diagnostics_20261017/scra/core.py)。一次分配是在每个 adapter 上把弱端的总预算 $Wr$ 均匀发给 $R$ 个组件。`counts[j,k]` 记录每个组件有多少弱贡献者，聚合器据此计算概率。

手算一个完整例子：有 1 个强端、3 个弱端，参与者权重都为 1，$R=4,r=2$。弱覆盖数可为 $(2,2,1,1)$。对于第一个组件，给定该覆盖数，每个弱端被选中的概率为 $2/3$；第四个则为 $1/3$。balanced 同 rank 聚合使用这个 `counts/W`，而 $r/R=1/2$ 是平均分配意义下的边际概率。

只看第一个组件的一个标量位移：强端上传 1，拥有该组件的两个弱端分别上传 3、5，另一个弱端没有上传。

| 聚合 | 手算 | 结果 |
|---|---|---:|
| owner | $(1+3+5)/3$ | 3.00 |
| Hájek | $[1+(3+5)/(2/3)]/[1+2/(2/3)]$ | 3.25 |
| zero-fill | $(1+3+5+0)/4$ | 2.25 |

zero-fill 等于 owner 乘以贡献者权重份额 $3/4$。Hájek 通过逆概率改变强弱端相对权重。理解这个例子后，再读 [METHOD §3–§4](METHOD.md#3-分配与包含概率) 的一般公式。

当前两套主结果分别固定使用 RoBERTa 的 Hájek 和 Qwen 的 zero-fill。典型弱覆盖约为 2.25 和 1.125，这为聚合配置提供解释；代码中的方法表决定聚合器，在线覆盖切换留作后续研究。

## 4. 按三个维度阅读方法名称

读 [`c2fed.SPEC`](../federated_finetuning/experiments/scra_diagnostics_20261017/scra/c2fed.py)。先识别本地训练、组件分配、聚合，再读方法名。

| 名称 | 弱端本地训练 | 服务器采用的组件 | 聚合 |
|---|---|---|---|
| `r8` | 全部 $R$ 组件 | 全部 | 样本加权平均 |
| `vref` / `vref_zf` | 全部 $R$ 组件 | balanced 分配的 $r$ 个 | Hájek / zero-fill |
| `amp` / `amp_zf` | 选中 $r$ 个，增量前向放大 | 同一分配集合 | Hájek / zero-fill |
| `L2` | 选中 $r$ 个，普通前向 | 同一分配集合 | owner |
| `fslora` / `fslora_zf` | sketch 前向、真实梯度 | balanced 分配集合 | Hájek / zero-fill |
| `fslora_paper` | sketch 前向、真实梯度 | 独立随机集合 | zero-fill |

`r8` 是沿用的运行名：Qwen 中它表示全局 rank 64。FSLoRA 式 sketch 将选中组件的整个状态放大、未选组件移出前向；AmpLoRA 保留完整起点，只放大本轮增量。`fslora_st` 则保留 sketch 前向值、使用普通乘积梯度路径，用于分离状态和梯度因素。RoBERTa 对应名字可在 [`batch2.LABELS`](../federated_finetuning/experiments/scra_diagnostics_20261017/scra/batch2.py) 查到。

最值得先读的两对比较是 `amp` 对 `vref`、`amp_zf` 对 `vref_zf`，它们保持分配和聚合一致，比较受限本地训练与完整本地训练。再读 `amp_zf` 对 `amp`，观察聚合变化。`L2` 对 `amp` 同时改变本地放大与聚合，需要结合单因素对照解释。

## 5. 从原始曲线读到主结果

先读 [RESULTS](RESULTS.md) 的主结果和对照，再按 [PROVENANCE](PROVENANCE.md) 找对应运行；实际重跑与重新汇总的命令以 [REPRODUCE](REPRODUCE.md) 为准。

RoBERTa 路径：读 [`batch.py`](../federated_finetuning/experiments/scra_diagnostics_20261017/scra/batch.py)、[`batch2.py`](../federated_finetuning/experiments/scra_diagnostics_20261017/scra/batch2.py)，再查[机制研究记录 §7–§8](designs/research_two_directions_2026-10-17.md)。主要输入是各 seed 的 `d0/d0_rounds.json`（部分历史参照使用 `result.json`）和身份配置。指标先平均核心弱客户端，再平均第 110–150 轮评测点，最后跨 seed 汇总。

Qwen 路径：读 [C2 设计 §14](designs/c2_1_design_2026-10-18.md)及 [`c2fed.py`](../federated_finetuning/experiments/scra_diagnostics_20261017/scra/c2fed.py) 的 report 函数。每次运行的 `fed.json` 包含逐轮训练记录、评测曲线与机制诊断；主指标取第 40–60 轮核心弱端准确率均值。

| 设置 | 完整本地训练、部分上传 | AmpLoRA，同聚合 | 全组件训练与上传 |
|---|---:|---:|---:|
| RoBERTa，Hájek | `vref` 80.17% | `amp` 80.12% | `r8` 80.19% |
| Qwen，zero-fill | `vref_zf` 80.48% | `amp_zf` 80.41% | `r8` 82.19% |

以上为三个 source seeds 的均值。比较方法时先在相同 seed 内作差，再汇总这些差；客户端和轮次用于构造该 seed 的指标。RoBERTa 原批次一与批次二的标准差分别采用 `ddof=0`、`ddof=1`，阅读 `±` 时同步看汇总代码。

建议依次回答四个问题：受限训练损失多少精度？放大后与同聚合的 vref 差多少？更换聚合改变多少？部分上传与全上传又差多少？例如 Qwen 的 82.19 → 80.48 → 80.41 对应部分上传代价约 1.72pp，放大后受限训练的附加差约 0.07pp。

## 6. 最后读实现和资源的对应关系

当前复现路径保留完整 rank-$R$ 参数，用梯度 mask 控制哪些组件改变；完整形状的 Adam 状态、本地 `task_state` 和增量张量仍存在于实验进程。部分上传在聚合时通过集合筛选实现，LoRA 逻辑上传预算为 $r/R$；RoBERTa 的完整分类头另计，计入后弱端总上行约为完整方案的 75%。

把未选组件折进冻结权重、只保存 $r$ 个训练组件的是[资源核算](designs/resource_case_2026-10-18.md)中的 folded 实现设想与估算口径。当前代码的显存应从实际运行记录读取，资源模型的估算另列。详见 [METHOD §5](METHOD.md#5-一轮算法与实现预算)。

完成这条学习路径后，可用[论文提纲](thesis/thesis_outline_2026-10-19.md)串起背景、诊断、方法和两套实验，再沿[文献核对](designs/literature_check_2026-10-18_amplified_restricted.md)学习各组件与既有工作的联系。论文提纲和 designs 目录保留研究底稿，当前阅读口径以 [METHOD](METHOD.md)、[RESULTS](RESULTS.md) 和代码为准。
