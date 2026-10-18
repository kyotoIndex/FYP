# 文献核对：放大受限训练是否已有先例

日期：2026-10-18。状态：**文献核对，没有新实验，没有改代码。** 读到什么程度逐篇标注在 §4 的表里（全文 / 代码 / 只有摘要）。

上游：
- [两个方向的深入研究](research_two_directions_2026-10-17.md) §5（D2）、§6（D3）
- [强端稀缺下的近邻核对](algorithm_design_2026-10-17_scarce_strong_hetrank.md) §2（Fed-PLoRA、raFLoRA 等异构 rank 近邻表，本文不重复）

---

## 0. 结论先行

| 问题 | 回答 |
|---|---|
| 判定 | **部分已有先例（partially anticipated）** |
| 最接近的工作 | [FSLoRA](https://arxiv.org/abs/2501.19389)（ICML 2026）。它的前向已经对被选分量乘 R/r，在 Adam 下本地函数变化的速度与我们相同，在冷启动（B = 0）时两者完全一样（在 Adam eps → 0 的极限下；eps = 1e-8 时小模型一步相差 12.4%，见 [FIX_BATCH2 §2、§10](../../federated_finetuning/experiments/scra_diagnostics_20261017/FIX_BATCH2_CONTROLS_2026-10-18.md)） |
| 已有的成分 | ① 对被训练分量乘 R/r（FSLoRA、IST 理论、HeteroFL Scaler、rank dropout 的均值归一化）；② 冻结分量以当前全局值留在前向（Fed-PLoRA、FedPLT）；③ 只放大相对锚点的偏离量，被屏蔽的部分取锚点值（Mixout、DARE，不在联邦场景里） |
| 没找到的部分 | 三者合在一起：**只放大本轮增量、不对当前状态做 sketch**，冻结分量留在前向，上传未放大的增量。动机也没找到：在 Adam 下匹配本地函数变化的速度，以消除共同训练的分类头和被训练分量的过冲 |
| 审稿风险 | ① "FSLoRA + Fed-PLoRA 的组合"；② "一阶等价于给受限分量调大学习率，再在服务器端缩回" |

---

## 1. 被核对的方法（精确表述）

设 R 个 rank-1 分量 b_k a_k^T。弱端被分到集合 S，|S| = r，α = R/r。本轮开始时的全局值记为 b0_k、a0_k。

**前向：**

```
W = W_global + Σ_{k∉S} b0_k a0_k^T + Σ_{k∈S} [ b0_k a0_k^T + α (b_k a_k^T − b0_k a0_k^T) ]
```

**梯度：** 只经过未放大的项，即真实梯度除以 α。在 Adam 下方向相同（只差 ε 的影响）。

**上传与聚合：** 上传未放大的位移，服务器做覆盖均衡的分量分配，再做 Hájek 聚合。

**机制：** Adam 的参数步长不随梯度大小缩放。只训练 r/R 个分量时，每步的函数变化约降为 r/R，残差一直存在，分类头和被训练分量沿共识方向持续前进，于是过冲。D3 中，单个客户端的分量位移、头部步长和本地损失在放大后都与完整训练一致（§6 of 上游）。

### 1.1 三种等价写法（用来确定要和谁比）

1. **参数层面（一阶）：** b_k a_k^T − b0_k a0_k^T = Δb a0^T + b0 Δa^T + Δb Δa^T。放大前向在一阶上等价于两步：被训练分量在 Adam 下用 α 倍学习率（分类头的学习率不变）；服务器把这些分量的位移除以 α。二阶项不同：放大前向里交叉项的系数是 α，调学习率是 α²。所以需要比较的近邻有两类：**按分量 / 按因子调学习率**的工作，以及**聚合端归一化**的工作。
2. **乘子层面：** 全局 LoRA 乘子是 s/R（s 为 lora_alpha）。放大前向相当于对本轮增量使用"本地 rank 的乘子" s/r，当前状态仍用 s/R。所以需要比较的近邻是**按 rank 设定乘子或学习率**的工作。
3. **sketch 层面：** FSLoRA 的前向是 W0 + α Σ_{k∈S} b_k a_k^T。它减去我们的前向，差值是

   ```
   ξ_S = α Σ_{k∈S} b0_k a0_k^T − Σ_{k=1}^{R} b0_k a0_k^T
   ```

   也就是对本轮开始时的状态做 sketch 引入的误差。它对 S 取期望为 0，但对单个客户端不为 0。B = 0 时 ξ_S = 0，两者的前向值完全一样；本地更新还受梯度尺度影响，只在 Adam eps → 0 时一致（在 Adam eps → 0 的极限下；eps = 1e-8 时小模型一步相差 12.4%，见 [FIX_BATCH2 §2、§10](../../federated_finetuning/experiments/scra_diagnostics_20261017/FIX_BATCH2_CONTROLS_2026-10-18.md)）。**所以我们的做法可以精确地说成"只对增量做 sketch，不对状态做 sketch"。**

---

## 2. 检索范围

- **arXiv：** 用 export API 做摘要字段检索，例如 `abs:LoRA AND abs:sketching`、`abs:submodel AND abs:federated AND abs:scaling AND abs:update`、`abs:"partial training" AND abs:federated AND abs:rescaling`、`abs:"rank dropout"`、`abs:"block coordinate" AND abs:Adam AND abs:"fine-tuning"`、`abs:LoRA AND abs:heterogeneous AND abs:rank AND abs:amplify`，共 10 组；最后用 id_list 核对了本文所有引用的编号与标题。
- **通用检索（覆盖 Google Scholar / OpenReview 索引的页面）：** 约 12 组，包括 "partial model training rescale update inverse of trained fraction"、"LoRA train subset of rank components amplification / compensation factor"、"heterogeneous rank federated LoRA scaling factor alpha/r"、"federated partial training classifier head co-adaptation overshoot"。
- **Semantic Scholar：** API 返回 429（限流），**没有完成**。这是本次核对的一个缺口。
- **代码：** 读了 FSLoRA 公开实现的 `src/LoRA_Sketching.py`、`models.py`、`main.py`，以及 LyCORIS 的 `lycoris/modules/locon.py`。

任务里给的 FLASC 编号 2404.15182 不对，正确的是 [2406.05233](https://arxiv.org/abs/2406.05233)。

---

## 3. 比较维度

| 维度 | 含义 |
|---|---|
| 缩放什么 | 整层输出 / 整个低秩乘积 / 只有本轮增量 / 梯度或学习率 / 聚合后的位移 |
| 在哪里缩放 | 本地前向 / 本地优化器 / 服务器聚合 |
| 冻结部分是否留在前向 | 以当前全局值参与前向，还是被丢弃（置零、截断） |
| 针对的失败 | 无偏性、激活尺度不一致、正则化、收敛速度、过冲 / 拼接 |
| 优化器 | 我们的论证依赖 Adam 的尺度不变性；SGD 下"放大前向"和"放大梯度"的含义不同 |

---

## 4. 逐篇核对

### 4.1 最接近：FSLoRA 及其后续

**[FSLoRA](https://arxiv.org/abs/2501.19389)**（Fang 等，ICML 2026；读了 arXiv HTML v3 全文和公开代码）

- **缩放什么：** 定义 3.1 的 sketch 为 S = (r/k) Σ_{j∈I} e_j e_j^T，|I| = k，I 均匀抽取，所以 E[S] = I。本地损失是 ℓ(W0 + B S A)。放大的是**被选分量的整个乘积** b_k a_k^T，不是本轮增量。
- **冻结部分：** 未选中的分量乘 0，**不参与前向**。
- **梯度：** 按链式法则，∇_B 和 ∇_A 都带 S，被选分量的梯度带 r/k 因子。论文里的本地更新（式 4）写成 SGD。
- **聚合：** 上传被选的行 / 列的位移；服务器按 1/N 对所有客户端做零填充平均（式 5），不按贡献者数归一化。
- **动机：** 无偏性，论文称它"对收敛分析至关重要"。附录 C 报告，破坏无偏性的按重要性 sketch 表现更差。论文没有讨论 r/k 因子对每步函数变化、过冲或分类头的影响。
- **与我们的关系：**
  1. 在 Adam 下（按论文的写法：Adam 作用于 B、A，前向是 B S A），被选分量的增量在前向中同样被放大 r/k = R/r 倍，本地函数变化的速度与我们相同。冷启动时两者相同（§1.1 第 3 条）（在 Adam eps → 0 的极限下；eps = 1e-8 时小模型一步相差 12.4%，见 [FIX_BATCH2 §2、§10](../../federated_finetuning/experiments/scra_diagnostics_20261017/FIX_BATCH2_CONTROLS_2026-10-18.md)）。
  2. 差别一：FSLoRA 对当前状态也做 sketch。只要适配器不为 0，每个客户端的本地起点就偏离全局模型 ξ_S，而且 ξ_S 的大小随适配器的累积而增大。我们的本地起点就是全局模型。
  3. 差别二：在 SGD 下，FSLoRA 的梯度也带 R/r。每步函数变化约为完整训练的 R/r 倍，过度放大，靠"零填充平均"在期望上抵回。我们去掉了梯度里的 α，上传原始位移并对贡献者做 Hájek 聚合。
  4. 差别三：目标不同。FSLoRA 追求模型和梯度的无偏，我们追求本地训练动力学与完整训练一致。
  5. FSLoRA 的 RoBERTa 实验也训练分类头（附录 I 的目标模块里有 "classification head"），但没有分析头部与受限分量的相互作用。
- **代码层面的观察（谨慎）：**
  - 公开代码用的是 AdamW，与论文式 4 的 SGD 不同。
  - 代码把 B 重参数化为 p = BS，再对 p 做 Adam。在 Adam 下，这与"对 B 做 Adam、前向用 BSA"并不等价：Δp·A 项不被放大，只有 p·ΔA 项被放大。冷启动时 p = 0，所以**这一实现在冷启动时不放大函数变化**。代码注释说这是"数学上等价的实现"，这在 SGD 加 S² 梯度修正时成立。
  - 代码用 `'lora_B' == n` 判断参数名，而 PEFT 的参数名是 `...lora_B.default.weight` 这样的完整路径。按字面看，这个条件不会成立，sketch 就不会生效。我们没有运行代码验证。
  - 所以如果把 FSLoRA 作为对照，应该**按论文的式子实现**，不要直接用它的代码。

**[Adaptive Federated LoRA with Independent Sampling](https://arxiv.org/abs/2505.23555)**（只读了摘要）：联合优化 LoRA sketching 比例和客户端抽样概率，应属于 FSLoRA 的框架。看不出改变了缩放的对象。

### 4.2 被训练的部分整体缩放，冻结 / 丢弃部分不参与前向（dropout 一族）

| 工作 | 缩放什么 / 在哪里 | 冻结部分在前向？ | 动机 | 阅读深度 |
|---|---|---|---|---|
| [HeteroFL](https://arxiv.org/abs/2010.01264) Scaler | 每个参数层的**整个输出表示**乘 1/r^{p−1}（宽度缩减比），只在训练时；推理时不缩放 | 否（客户端只拿到左上角子矩阵） | 与 inverted dropout 相同：让子模型训练的参数尺度与全模型推理一致 | 全文（ar5iv） |
| [FedRolex](https://arxiv.org/abs/2212.01548) | 沿用 static BN 和 scaler 模块（PreResNet18）；Transformer 实验未提 | 否（滚动抽取子模型） | 让全局模型各部分均匀地被训练 | 全文（ar5iv） |
| [FjORD](https://arxiv.org/abs/2102.13451) | 有序 dropout，按宽度截取 | 否 | 嵌套子模型 | 摘要 / 已知内容 |
| [Federated Dropout](https://arxiv.org/abs/1812.07210)（Caldas 等） | 丢弃单元得到子模型 | 否 | 降低客户端开销 | 只读了摘要，未核实是否缩放 |
| [IST 理论](https://arxiv.org/abs/2306.16484)（Shulgin & Richtárik，ICML 2024） | 梯度在**被 sketch 的模型** C_i x 处计算，C_i 是乘以 n 的置换 sketch，使 (1/n)Σ C_i = I | 否 | 完美重构，便于分析；结论是 IST 一般不收敛到原问题的解，存在不可消除的邻域 | 全文（HTML 前半） |
| LyCORIS rank dropout（[LyCORIS](https://arxiv.org/abs/2309.14859)） | 训练时丢弃 rank 维度的中间激活；`rank_dropout_scale=True` 时把掩码除以自身均值，即整体乘约 R/保留数 | 否 | 正则化 | 代码 |
| [LoRA Meets Dropout](https://arxiv.org/abs/2403.00812)、[LoRA Dropout](https://arxiv.org/abs/2404.09610)、[FedLoDrop](https://arxiv.org/abs/2510.12078) | 对 LoRA 的行 / 列或隐藏单元做 dropout | 否 | 过拟合控制、通信 | 只读了摘要 |

**共同点与区别：** 这一族都对"保留下来的部分"整体乘 1/保留率，目的是期望一致或尺度一致。它们都**丢弃**未选中的部分。按 §1.1 第 3 条，丢弃再整体放大，就等价于"增量放大 + 状态 sketch 误差"。这一族没有一篇讨论 Adam 下的函数变化速度或过冲。

### 4.3 冻结部分以当前值留在前向，但不缩放

| 工作 | 冻结部分 | 缩放 | 聚合 | 是否讨论过冲 | 阅读深度 |
|---|---|---|---|---|---|
| [Fed-PLoRA](https://arxiv.org/abs/2602.16936)（ICLR 2026） | Select-N-Fold：未选的分量用上一轮全局值折进冻结权重，留在前向 | 无（正文的 PLoRA 求和没有 R/r；附录有 rsLoRA 变体，作为扩展） | 每个分量只在训练过它的客户端之间平均 | 否；只说随机选择可以避免分量长期不更新 | 全文（HTML 前半） |
| [FedPLT](https://arxiv.org/abs/2605.02337) | 冻结的子层保持本轮全局值，前向用完整结构 | 无；本地是普通 SGD | 每个子层在训练过它的客户端之间按样本数加权平均 | 否；只说轮换可以避免长期欠训练 | 全文（HTML 前半） |
| [Partial Parameter Updates](https://arxiv.org/abs/2509.22418) | 前向标准，用完整的本地参数；只有反传和优化器步被限制在切片上 | 无；内层 AdamW，外层 Nesterov SGD | 按训练该参数的节点数逐元素归一化 | 否；附录提到切片式外层更新比本地变化大得多，水平切片会发散 | 全文 |
| [FedPart](https://arxiv.org/abs/2410.11559) | 每轮只训练一层或几层，其余层作为"锚" | 无（二值掩码） | 普通平均 | 针对"层失配"，不是过冲 | 全文 |
| [Deng 等](https://arxiv.org/abs/2511.06132) 子模型训练理论 | **置零**（w ⊙ m 进入前向），不是留在当前值 | 无 | 未选参数用上一轮全局值填补后平均 | 收敛到掩码后的目标 F_p，与原目标的差距随掩码比例增大；打乱划分优于独立随机 | 全文（HTML 前半） |
| [Fed-HeLLo](https://arxiv.org/abs/2506.12213)、[FedRA](https://arxiv.org/abs/2311.11227)、[EmbracingFL](https://arxiv.org/abs/2406.15125) | 层级的部分训练 | 无 | 层级的拥有者平均 | 否 | 见上游笔记 |
| [FLASC](https://arxiv.org/abs/2406.05233) | 本地训练**完整** LoRA，只稀疏上传 | 无 | 零计入平均 | 否；对应我们的 vref，不受限训练 | 摘要 + 上游笔记 |
| [FedPA-LoRA](https://arxiv.org/abs/2608.15381) | 客户端保留本地因子，向 rank 专属的全局参照做乘积对齐 | 摘要中没有 | 乘积空间 | 针对"聚合失配"与"初始化失配"的权衡 | 只读了摘要 |

**共同点与区别：** 这一族的前向结构与我们一致，也就是 L2 / hajek 基线。它们都不缩放，因此都会遇到我们在 D2 中测到的过冲：单步沿 r8 方向的 LoRA 步长 1.55–1.86，头部步长 1.64–1.93。没有一篇报告这个现象。Fed-PLoRA 的随机选择、FedPLT 的轮换（以及上游笔记中 FedHera 的尾部加权），处理的都是覆盖问题，不是步长问题。

### 4.4 只放大相对锚点的偏离量，被屏蔽的部分取锚点值（形式上最接近）

| 工作 | 公式 | 与我们的异同 | 阅读深度 |
|---|---|---|---|
| [Mixout](https://arxiv.org/abs/1909.11299)（ICLR 2020） | mixout(w; u, p) = (1/(1−p))·((I−M)u + Mw − p·u)。被丢弃的参数取 u（微调时为预训练权重），保留的参数变成 u + (w − u)/(1−p) | **形式几乎相同**：被屏蔽的部分取锚点值并留在前向，保留部分相对锚点的偏离放大 1/保留率。区别：锚点是预训练权重，不是本轮起点；掩码每步重采样，所有参数在期望上都被训练；按参数放大，不是按乘积；动机是对 u 的自适应 L2 正则与期望一致。不在联邦场景中 | 全文（HTML） |
| [DARE](https://arxiv.org/abs/2311.03099) | 对微调增量随机丢弃比例 p，其余乘 1/(1−p) | 只放大增量，以保持函数（"近似原来的嵌入"）。但它是训练后的模型合并操作，不涉及训练 | 摘要 |

**含义：** "锚点 + 放大的掩码偏离量"这一代数形式本身**不新**。如果把 Mixout 的锚点取为本轮起点、掩码在一轮内固定，就得到我们的参数级版本。我们的放大作用在乘积上，这只是细节。所以不能把"只放大增量"本身当作贡献。

### 4.5 按 rank 或按因子调乘子 / 学习率

| 工作 | 缩放什么 | 与我们的关系 | 阅读深度 |
|---|---|---|---|
| [LoRA+](https://arxiv.org/abs/2402.12354) | B 的学习率是 A 的 λ 倍 | 按因子区分，来自宽度趋于无穷时的特征学习分析；与"训练了多少个分量"无关 | 已知内容 |
| [rsLoRA](https://arxiv.org/abs/2312.03732) | 乘子从 α/r 改为 α/√r | 全局的 rank 稳定化，不区分哪些分量被训练 | 摘要 |
| [Learning Rate Scaling across LoRA Ranks](https://arxiv.org/abs/2602.06204) | 用 SignSGD 代替 Adam 推导最优学习率与 rank 的关系：Init[A]、乘子 α = r^{-1} 时，最优学习率与 rank 无关；α = 1 时 η ∝ r^{-1/2} | **给我们的机制提供了理论支撑**：在 Adam 类优化器下，每步的特征变化取决于"乘子 × 参与训练的分量数"。我们的情形是乘子固定为 1/R、只训练 r 个分量，按它的分析，增量的乘子应改为 1/r，这正是 §1.1 第 2 条。它没有讨论只训练一部分分量的情形 | 全文（HTML 前半） |

**审稿风险：** 审稿人会说"这只是给受限分量用了与本地 rank 匹配的乘子或学习率"。按 §1.1 第 1 条，一阶上确实如此。能站住的回答有两点：
1. 单独调学习率不行，必须同时在服务器端除以 α。否则上传的位移大 α 倍，聚合后的过冲反而更大。D2 中弱端学习率 × 0.6 的结果说明，学习率和收缩不能互相替代。
2. 放大前向不改变优化器的超参数和状态，头部学习率保持不变，上传量也天然处在完整训练时单个分量的尺度上。

这两点需要一个消融来确认：比较"学习率 × α + 服务器 ÷ α"与放大前向（见 §6.4）。

### 4.6 聚合端的归一化与外推

| 工作 | 缩放什么 / 在哪里 | 与我们的关系 | 阅读深度 |
|---|---|---|---|
| [FedNova](https://arxiv.org/abs/2007.07481) | 服务器把每个客户端的位移除以它的本地步数（或 ‖a_i‖₁），再乘有效步数 τ_eff | 思想相近：按"有效进展量"归一化，消除目标不一致。但它只改聚合，不改本地轨迹。我们的 D2 / D3 说明，本地轨迹不对时，聚合端的标量修正（RCS、固定收缩）最多追回 25% | 已知内容 |
| [FedExP](https://arxiv.org/abs/2301.09604)（ICLR 2023） | 服务器步长 ≥ 1，按 POCS 外推自适应选取 | 全局的服务器端外推，不区分分量，不改本地训练 | 摘要 |
| [Lookahead](https://arxiv.org/abs/1907.08610) | 慢权重向快权重插值 | 只是名义上相关，不是部分训练 | 已知内容 |

### 4.7 块坐标下降

| 工作 | 是否补偿步长 | 与我们的关系 | 阅读深度 |
|---|---|---|---|
| [Parallel Coordinate Descent](https://arxiv.org/abs/1212.0873)（Richtárik & Takáč） | 同时更新 τ 个块时，用 ESO 参数 β ≥ 1 **缩小**步长，防止块间耦合造成过冲 | 方向相反：在经典的 Lipschitz 步长下，部分更新不需要放大，多块并行时反而要缩小。我们需要放大，是因为 Adam 的步长按参数归一化，与函数变化无关 | 已知内容 |
| [Stochastic Subspace Descent](https://arxiv.org/abs/1904.01145) | 每步把梯度投影到随机子空间 | 随机子空间方法通常把投影缩放到无偏，但**本次没有逐式核实**这篇的缩放约定 | 只读了摘要 |
| [BAdam](https://arxiv.org/abs/2404.02827)、[LISA](https://arxiv.org/abs/2403.17919) | 按块 / 按层轮流或抽样做 Adam，就我们所知没有按块的步长补偿 | 集中式微调，没有拼接，也没有共同训练的头部问题 | 本次未逐篇核实 |

### 4.8 分类头与主体的相互作用

- [FedBABU](https://arxiv.org/abs/2106.06042)：联邦训练时冻结分类头，只训练和聚合主体。我们 D3 中的 fh_hajek（弱端冻结头部）就是这一做法的受限版本，结果更差：LoRA 的过冲从 1.1–1.2 倍升到 2.3–2.5 倍，这就是"压力转移"。所以 FedBABU 式的做法**不能**替代放大。
- [LP-FT](https://arxiv.org/abs/2202.10054)：集中式微调中，随机初始化的头部会扭曲预训练特征，所以先训练头部再微调全部参数。这同样是头部与主体之间的相互作用，但方向和场景都不同。
- 检索"partial training + classifier overshoot / co-adaptation"，**没有找到**描述"受限主体使共同训练的头部过冲"的工作。

---

## 5. 综合：哪些成分已有

| 成分 | 是否已有 | 代表工作 |
|---|---|---|
| 对被训练的分量乘 R/r（或 1/保留率） | **已有** | FSLoRA、IST 理论、HeteroFL Scaler、LyCORIS rank dropout |
| 冻结分量以当前全局值留在前向 | **已有** | Fed-PLoRA（Select-N-Fold）、FedPLT、Partial Parameter Updates |
| 只放大相对锚点的偏离量，被屏蔽部分取锚点值 | **已有**，但不在联邦 / 部分训练场景 | Mixout、DARE |
| 按贡献者归一化的分量聚合 | **已有** | Fed-PLoRA、raFLoRA、FedPLT（Hájek 加覆盖均衡分配是我们 SCRA 的已有部分，不算本方法的新颖点） |
| 以上组合用于异构 rank 联邦 LoRA：只对增量做 sketch，不对状态做 sketch；上传未放大的增量 | **未找到** | — |
| 动机：Adam 下每步函数变化与训练的分量数成正比 → 残差持续 → 共同训练的头部和分量过冲 | **未找到**（Learning Rate Scaling across LoRA Ranks 给出了集中式的、按 rank 的理论基础，但没有涉及部分训练） | — |
| 现象：受限训练使头部和 LoRA 沿共识方向过冲 1.6–1.9 倍；冻结头部会把压力转移给 LoRA | **未找到** | — |

---

## 6. 结论

### 6.1 判定

**部分已有先例。** 数学形式本身可以由已有成分拼出来：FSLoRA 的 R/r 因子，加上 Fed-PLoRA 的冻结结构，加上 Mixout 式的"只放大偏离量"。冷启动时，它与 FSLoRA 相同（在 Adam eps → 0 的极限下；eps = 1e-8 时小模型一步相差 12.4%，见 [FIX_BATCH2 §2、§10](../../federated_finetuning/experiments/scra_diagnostics_20261017/FIX_BATCH2_CONTROLS_2026-10-18.md)）。没有找到先例的是三件事：
1. 不对状态做 sketch、只放大增量的这个具体变体；
2. 它在 Adam 下的动机（匹配函数变化的速度，而不是无偏）；
3. 它所针对的失败（共同训练的头部与被训练分量的过冲，以及跨客户端拼接）的发现和诊断证据。

### 6.2 最接近的工作

1. **[FSLoRA](https://arxiv.org/abs/2501.19389)**：放大因子相同，Adam 下本地速度相同，冷启动时等价。区别在于状态 sketch 误差 ξ_S、零填充聚合，以及动机（无偏）。
2. **[Fed-PLoRA](https://arxiv.org/abs/2602.16936)**：前向结构相同（冻结分量留在前向），但没有放大。它就是我们的 L2 / hajek 基线。
3. **[Mixout](https://arxiv.org/abs/1909.11299)**：代数形式最接近，场景和动机完全不同。

### 6.3 能经得住审稿的主张措辞

**可以这样写（中文）：**

> 在异构 rank 联邦 LoRA 中，弱端受限训练造成的精度差距来自本地训练动力学，而不是通信：在 Adam 下，只训练 R 个分量中的 r 个，每步的函数变化约为完整训练的 r/R，本地残差长期存在，共同训练的分类头和被训练的分量都会沿共识方向过冲。我们让冻结分量以当前全局值留在前向，只把被训练分量的**本轮增量**在前向中放大 R/r 倍，上传未放大的增量，并按覆盖做 Hájek 聚合。这与 FSLoRA 的无偏 sketch（放大被选分量的整个乘积、丢弃其余分量）不同：我们不对当前状态做 sketch，本地起点与全局模型一致；两者只在适配器为 0 时（冷启动）一致。

**英文版（供论文用）：**

> We show that under Adam, restricting a client to r of R LoRA components slows its per-step function change by roughly r/R, which leaves a persistent residual that both the co-trained classifier head and the trained components overshoot. We therefore keep frozen components in the forward pass at their current global values and amplify only the *current-round increment* of the trained components by R/r, while uploading the un-amplified increment for coverage-balanced Hájek aggregation. Unlike sketch-based heterogeneous LoRA (FSLoRA), which rescales the whole sketched adapter and drops unselected components, our rule sketches only the increment, not the state, so each client starts from the exact global model; the two coincide only when the adapter is zero.

**不能这样写：**
- "首次在部分训练 / 子模型训练中放大更新"（HeteroFL、FSLoRA、IST、rank dropout 都有 1/保留率因子）；
- "首次让冻结部分留在前向"（Fed-PLoRA、FedPLT）；
- "无偏"或"有收敛保证"（我们没有做这方面的分析；无偏正是 FSLoRA 的卖点）；
- "等价于完整训练"（D3 中位移与完整训练的余弦为 0.992–0.997，不是逐位相同）；
- "节省显存"（RoBERTa-base 规模下是名义的，见上游 §0.1）。

### 6.4 判定成立所依赖的对照（建议在多轮实验里补上）

1. **FSLoRA 式前向**（状态也做 sketch，丢弃未选分量），在我们的设置里分别配零填充平均和 Hájek 聚合。如果它与放大前向一样好，那么"不对状态做 sketch"就没有实际价值，主张应缩小为"FSLoRA 式放大在 Adam 下修复了过冲，这是一个新的解释"。
2. **"学习率 × α + 服务器 ÷ α"**（§1.1 第 1 条）。如果与放大前向等价，就应把方法表述为"让本地进展与完整训练匹配"，而不是强调前向放大这一实现形式。
3. 冷启动早期（B = 0）FSLoRA 与我们相同，所以两者的差别只会在中后期出现。比较时要报告分阶段的结果，不能只报告最终精度。

### 6.5 本次核对的局限

- Semantic Scholar API 被限流，没有完成检索；没有直接访问 Google Scholar 界面，用通用检索代替。
- FedPA-LoRA、FedLoDrop、Federated Dropout、Stochastic Subspace Descent 只读了摘要；BAdam、LISA、LoRA+、FedNova 依据已知的方法描述，本次没有重新逐篇核实。
- 几篇长文（Fed-PLoRA、FedPLT、Deng 等、IST）只读了 HTML 的前半部分，附录里可能还有缩放相关的变体（例如 Fed-PLoRA 附录 D.5 的 rsLoRA 变体，我们只知道它存在）。
- FSLoRA 代码层面的两点观察（重参数化在 Adam 下不等价、参数名判断按字面不会成立）只是读代码得出的，没有运行验证。
