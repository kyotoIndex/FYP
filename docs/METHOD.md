# AmpLoRA：方法与实现

AmpLoRA（Amplified restricted LoRA，暂名）研究异构 rank 联邦微调：强客户端训练全部 LoRA 组件，弱客户端只训练分配到的组件，通过本地前向放大补偿受限训练的动态差异。本版文档统一用 AmpLoRA；历史设计中的 **ART / amplified restricted training** 和运行名 **amp** 指同一本地训练规则，**amp_zf** 是该规则配合 zero-fill 聚合的版本。

本文以当前代码说明方法。原始[算法设计](designs/algorithm_design_2026-10-18_amplified_restricted.md)、[Qwen 实验设计与冻结结论](designs/c2_1_design_2026-10-18.md)保留研究过程；其中包含概率、资源和覆盖选择的概括性表述，以本文的实现说明为准。

## 1. LoRA 组件与客户端预算

对一层冻结权重 $W^0$，LoRA 写成

$$
W=W^0+sBA=W^0+s\sum_{k=1}^{R}b_ka_k^\top.
$$

$A\in\mathbb R^{R\times d_{in}}$、$B\in\mathbb R^{d_{out}\times R}$，$s$ 是适配器的固定 scale。一个 component / block 是同一索引的 $A$ 行与 $B$ 列，代码把它存成 $[b_k;a_k]$。$R$ 是全局 rank，$r_i$ 是客户端的训练与逻辑上传预算；本地放大因子另记为 $\alpha_i=R/r_i$，与 LoRA 自身的 scale 区分。

第 $t$ 轮从全局因子 $(A_0,B_0)$ 出发。强客户端训练全部组件；弱客户端在每个 adapter 上获得集合 $S_i$，大小为 $r_i$。未选组件保持本轮全局值并参与前向。RoBERTa 还训练分类头，Qwen 使用冻结的原有 LM head 做标签词预测。

代码：[LoRALinear 与 TaskModel](../federated_finetuning/experiments/scra_diagnostics_20261017/scra/model.py)、[组件转换与本地训练](../federated_finetuning/experiments/scra_diagnostics_20261017/scra/core.py)、[Qwen 模型接口](../federated_finetuning/experiments/scra_diagnostics_20261017/scra/llm.py)。

## 2. 本地前向：只放大本轮乘积增量

弱客户端的 LoRA 前向值为

$$
s\left[B_0A_0+\alpha_i(BA-B_0A_0)\right]x.
$$

它保留完整全局起点，只放大当前本地训练造成的乘积变化。这里的 $BA-B_0A_0$ 含有 $B_0\Delta A+\Delta BA_0+\Delta B\Delta A$，包括二阶项；实现没有将它简化为因子增量的线性近似。

`LoRALinear.lora_update` 用以下形式实现：

```python
update = B(A(x))
start = B0(A0(x))
update = update + (alpha - 1) * (update - start).detach()
```

`detach()` 让额外放大项只改变前向值，autograd 沿普通 `BA` 分支反传；损失及其残差来自放大后的前向。

**这条反向规则是方法定义的一部分。** 对给定的本层输出梯度，本层 $A,B$ 的梯度等于放大模型局部梯度的 $1/\alpha_i$。但传给本层输入、进而传给前面各层的梯度也只走未放大的 $BA$ 分支，而放大模型在这里的输入 Jacobian 是 $s[(1-\alpha_i)B_0A_0+\alpha_iBA]$。所以从整个网络看，AmpLoRA 的本地更新不是"放大模型的真实梯度除以 $\alpha_i$"，而是由上式前向值和这条反向规则共同定义的更新规则。保留真实层间梯度的变体（参数梯度仍除以 $\alpha_i$）作为单独的对照检验。随后 `core._local_steps` 将未选 $A$ 行、$B$ 列的梯度清零，再裁剪梯度并执行 AdamW。每次客户端训练重新创建 optimizer；发布配置使用 `weight_decay=0`，使这些零梯度组件保持原值。

本轮结束后撤去临时 `amp` 设置，服务器采用原始因子位移

$$
\Delta\theta_{ik}=[b_{ik}^{end}-b_{0k};\ a_{ik}^{end}-a_{0k}],\quad k\in S_i.
$$

**上传与聚合采用的位移没有乘 $\alpha_i$，也没有除以 $\alpha_i$。** 服务器逐组件聚合因子位移，更新 $A,B$；它没有先计算各客户端的 $\Delta(BA)$ 再平均。两种聚合因 LoRA 的双线性结构而有差别。

### 为什么放大

Adam 首步在梯度远大于 $\epsilon$ 时，标量位移近似 $-\eta\,\mathrm{sign}(g)$。本轮起点的一阶降损由各被训练组件的贡献相加。若这些贡献近似可互换，只训练 $r_i/R$ 的组件会留下相应的 LoRA 推进缺口。$R/r_i$ 的前向增益旨在补上这个缺口，减少头部和已选组件承担的额外补偿。

这给出一个机制假设及可测对照：`vref` 完整本地训练，但只让服务器采用分配到的组件；`amp` 受限本地训练，采用同样的分配与聚合。两者比较直接衡量本地受限训练的代价。RoBERTa 的分类头诊断、前向放大与学习率补偿对照见[原始设计 §4–§7](designs/algorithm_design_2026-10-18_amplified_restricted.md)。

## 3. 分配与包含概率

`balanced_allocation` 对每个 adapter 独立分配。设本轮弱参与者数为 $W$、统一弱 rank 为 $r$，它将 $Wr$ 个槽位尽量平均分给 $R$ 个组件，使 `counts[j,k]` 为 $\lfloor Wr/R\rfloor$ 或 $\lceil Wr/R\rceil$，然后随机排列客户端并发牌，每人获得 $r$ 个不同组件。

Hájek 使用的弱端包含概率为

$$
\pi_{jk}=\mathrm{counts}[j,k]/W.
$$

这是给定本轮覆盖数后的概率。对随机 bonus 取平均后，边际概率为 $r/R$；**同 rank 的 balanced 实际聚合使用 `counts/W`**。例如 $W=3,R=4,r=2$ 时，覆盖数可以为 $(2,2,1,1)$，对应概率 $(2/3,2/3,1/3,1/3)$。

弱 rank 混合时，`hetero_allocation` 按当前覆盖最少的组件分配，以随机顺序和 tie-break 保持对称；`d0` 为每个弱客户端向聚合器传入 $\pi_i=r_i/R$。独立随机分配与前缀分配也保留为实验对照。

若 $Wr<R$，有些组件本轮没有弱贡献者；若也没有强贡献者，其更新为零。代码通过 `covered` mask 处理这一情况。

## 4. 三种聚合及 zero-fill 收缩

下面省略 adapter 下标。$q_i$ 是参与客户端的样本数权重，$O_k$ 是训练或上传组件 $k$ 的客户端集合；强端包含概率为 1。`core.block_aggregates` 实现：

| 聚合 | 组件位移 |
|---|---|
| owner | $\displaystyle\frac{\sum_{i\in O_k}q_i\Delta\theta_{ik}}{\sum_{i\in O_k}q_i}$ |
| Hájek | $\displaystyle\frac{\sum_{i\in O_k}(q_i/\pi_{ik})\Delta\theta_{ik}}{\sum_{i\in O_k}q_i/\pi_{ik}}$ |
| zero-fill | $\displaystyle\frac{\sum_{i\in O_k}q_i\Delta\theta_{ik}}{\sum_{i\in P_t}q_i}$ |

Hájek 用逆包含概率修正强弱贡献比例。zero-fill 将未上传的组件位移按零计入所有参与者的平均，因此严格满足

$$
\Delta\theta_k^{zf}=\rho_k\Delta\theta_k^{owner},\qquad
\rho_k=\frac{\sum_{i\in O_k}q_i}{\sum_{i\in P_t}q_i}.
$$

这是对 **owner** 的组件级收缩。强弱混合时 Hájek 还改变了相对权重，zero-fill 与 Hájek 的关系应直接从上式各自计算。RoBERTa 分类头始终按全部参与者的样本数平均。

### 覆盖选择在本仓库中的含义

验证过的两套设置分别采用以下配置：

| 设置 | $R/r$ | 典型弱覆盖 $m=W r/R$ | 主结果采用 |
|---|---:|---:|---|
| RoBERTa-base / MNLI | $8/2$ | $9\times2/8=2.25$ | `amp`，balanced + Hájek |
| Qwen2.5-0.5B / MNLI 标签词 | $64/8$ | $9\times8/64=1.125$ | `amp_zf`，balanced + zero-fill |

弱覆盖较少时，zero-fill 提供较小的服务器步长；Qwen 的服务器步长对照支持这一解释。这里的覆盖数是约 1 强、9 弱时的说明值，实际参与者与样本权重按轮变化。**当前实现通过 method 配置选择聚合器，未实现覆盖阈值判断或在线自动切换。** 两套验证设置为配置选择提供经验依据，连续自适应聚合保留为后续研究。

Qwen 的方法配置表见 [`c2fed.SPEC`](../federated_finetuning/experiments/scra_diagnostics_20261017/scra/c2fed.py)。

## 5. 一轮算法与实现预算

1. 按 seed 与轮次选参与者；对弱端分配每层组件集合。
2. 从同一全局状态开始本地训练。强端完整训练；弱端启用 $\alpha_i=R/r_i$ 的放大前向并 mask 未选组件梯度。
3. 从本地终点减去全局起点，得到未放大的因子位移；聚合仅采用对应上传集合。
4. 按配置执行 Hájek 或 zero-fill，更新全局因子；RoBERTa 分类头另按样本数平均。
5. 按预定轮次评测、保存曲线及断点。

上述 $r_i$ 描述训练可变组件数和逻辑上传预算。当前实验使用 **masked rank-R** 实现：完整 $A,B$ 仍是 `Parameter`，optimizer 接收完整张量，梯度 mask 后 Adam 状态仍按完整形状分配；amp 还保留完整起点副本。`task_state()` 返回完整任务状态，实验进程在内存中通过集合筛选模拟部分上传。LoRA 有效上传元素数为完整的 $r_i/R$，若带分类头，总量还要加上完整头部。

按实际发送的内容计：RoBERTa-base 的分类头（dense $768\times768+768$，out_proj $768\times3+3$）共 592,899 个参数，多于 24 个 Q/V 适配器全部 rank 8 因子的 294,912 个。弱端上传分类头加 2/8 的 LoRA 因子（73,728），**总上行约为完整方案的 75%**。Qwen 不新增分类头，弱端的 LoRA 上行是完整因子的 1/8。两个模型的下行都是完整的全局适配器（RoBERTa 另含分类头）。

[资源核算](designs/resource_case_2026-10-18.md)讨论的是另一种 **folded / Select-N-Fold** 实现：把未选组件折入冻结权重，只将选中因子作为训练参数。该方案的 LoRA 训练状态可随 $r_i$ 缩小，相关显存数字属于分析估算。发布的 masked 训练代码保留原实验路径；`c2fed` 也明确注明未折叠组件、该实验不作资源测量。复现实测资源时应使用相同实现口径。

## 6. 证据与方法定位

三个 source seeds 的主结果中，RoBERTa `amp` 为 80.12%，`vref` 为 80.17%；Qwen `amp_zf` 为 80.41%，`vref_zf` 为 80.48%。各自指标是核心弱客户端的后期评测窗口均值；Qwen 的全组件上传参照 `r8` 为 82.19%，所以在这一设置里部分上传仍有 1.72 个百分点的代价。正式结果见 [RESULTS](RESULTS.md)，来源对应见 [PROVENANCE](PROVENANCE.md)，循序阅读见[学习指南](LEARNING_GUIDE.md)。

AmpLoRA 的工作重点是受限本地动力学诊断、放大规则与覆盖聚合的分解验证。放大因子、保留冻结组件和锚定增量等组件有既有工作基础，与 FSLoRA、Fed-PLoRA、FLASC 的关系及原始来源见[文献核对](designs/literature_check_2026-10-18_amplified_restricted.md)。后续论文可从这些明确的组件关系和已完成对照组织贡献。
