# Batch-2 对照代码审查与修改清单

日期：2026-10-18  
审查对象：`federated_finetuning/experiments/scra_diagnostics_20261017/` 下第二批定位对照（batch 2）  
状态：**代码审查记录；batch 2 尚未实际运行，本文不是实验结果。**

---

## 0. 结论

这批代码的主体实现是正确的、可以运行：

- `model.py` 中的 FSLoRA forward、`core.py` 中的 `random_allocation` / `topk_sparsify` / `zerofill` 聚合、`d0.py` 中的 `weak_forward`、`weak_lr_compensation`、`weak_upload` 路径都已接通；
- `make_batch2.py` 能生成 15 个 plan，`run_batch2.sbatch` 的 array 与 plan 数一致；
- 本地测试通过：

```text
26 passed
```

但当前版本**不建议直接用于论文结论**。主要问题不是代码跑不起来，而是部分对照会混合多个机制，导致无法把结果归因到想验证的因素。

本次建议分三档：

| 优先级 | 问题 | 是否必须在正式提交前修 |
|---|---|---|
| P0 | 缺少 `balanced + zerofill` 的 FSLoRA 对照 | 是 |
| P0 | A1 的 FSLoRA vs amp 不是纯 state-sketch 对照，存在梯度尺度/Adam epsilon 混淆 | 是，至少加诊断或说明边界 |
| P1 | FLASC 强端上传预算未定义；Top-K 粒度需核对原文 | 是，写清或补对照 |
| P1 | P3/P4/P5 是混合外部基线，不能作为机制证明 | 是，改报告措辞 |
| P1 | 3 seed 均值 + 0.3pp 硬阈值统计上偏弱 | 建议 |
| P2 | 缺少 source code hash、部分测试覆盖和注释 | 建议 |

---

## 1. P0：补充 `fslora_zerofill`（同分配、零填充）对照

### 现状

当前 batch 2 中与 FSLoRA 有关的 run 是：

- `fslora_hajek`：`allocation=balanced`，`aggregation=hajek`，`weak_forward=fslora`
- `fslora_paper`：`allocation=random`，`aggregation=zerofill`，`weak_forward=fslora`

其中 `fslora_paper` 同时改变了：

1. 分量分配方式：random vs balanced；
2. 聚合方式：zerofill vs Hájek；
3. 服务器是否按包含概率归一化。

因此它不能回答：

> 在相同覆盖均衡分配下，零填充平均是否比 Hájek 差？

也不能单独证明“state sketch 是否有害”。

### 修改

在 `make_batch2.py` 的 `METHODS` 中加入：

```python
("fslora_zerofill", {
    "aggregation": "zerofill",
    "allocation": "balanced",
    "weak_forward": "fslora",
}),
```

即新增 3 个 run：

```text
fslora_zerofill_s2040
fslora_zerofill_s2041
fslora_zerofill_s2042
```

并在 `scra/batch2.py` 中：

- 加入 `fslora_zerofill` 到 `METHODS` / `LABELS`；
- 新增判据：

```text
P1b：fslora_hajek − fslora_zerofill ≥ 0.3pp（3 seed 均值）
```

### 验收

新增 run 的 `identity.json` 必须同时满足：

```json
{
  "allocation": "balanced",
  "aggregation": "zerofill",
  "weak_forward": "fslora"
}
```

报告表必须能单独列出：

```text
amp
fslora_hajek        (balanced + Hajek)
fslora_zerofill     (balanced + zerofill)
fslora_paper        (random + zerofill, external baseline)
```

这样 A1 才能拆成“forward 差异”和“aggregation 差异”两部分。

---

## 2. P0：A1 存在梯度尺度 / Adam epsilon 混淆

### 问题

设计稿认为：冷启动 `B = 0` 时 FSLoRA forward 和 amp 相同，因此 A1 可以隔离“是否对 state 做 sketch”。

这个说法只在 Adam 的 `eps → 0` 且没有梯度裁剪等理想条件下近似成立。

- FSLoRA forward 中被选分量乘了 `R/r`，真实梯度也乘 `R/r`；
- amp 的 forward 放大增量，但额外项被 `detach`，梯度不乘 `R/r`；
- Adam 的逐参数尺度不变性只能部分吸收这个差异，`eps=1e-8` 和统一的 `gradient_clip=1.0` 会留下差异。

审查时用当前代码做了合成数值检查：同一 batch、同一初始化、同一组被选分量、`B=0`、一步本地更新：

| Adam eps | amp 与 FSLoRA 一步更新的相对差 |
|---:|---:|
| `1e-8` | 约 `12%` |
| `1e-10` | 约 `0.56%` |
| `1e-12` | 约 `0.006%` |

这是合成 RoBERTa 的一步诊断，不是 MNLI 结果，但说明：

> **A1 的早期差异不能全部归因于 state sketch；其中可能包含梯度尺度和 Adam epsilon 的效应。**

### 修改方案 A（推荐）：增加 straight-through 的 FSLoRA 诊断变体

新增一个只用于机制归因的 forward 模式，例如：

```text
weak_forward = "fslora_st"
```

含义：

- 前向值与 FSLoRA 相同：被选分量整体乘 `R/r`，未选分量不参与；
- 但梯度按未放大的更新计算，与 amp 的梯度路径一致。

在 `model.py` 中可用 straight-through 形式实现（示意）：

```python
# hidden = inputs @ A.T，shape [batch, R]
# factor 是长度 R 的向量，被选分量为 R/r，未选分量为 0
scaled_update = (factor * hidden) @ B.T       # FSLoRA 前向值
plain_update  = hidden @ B.T                  # amp 式未放大梯度路径
update_st = scaled_update.detach() + plain_update - plain_update.detach()
return base(inputs) + update_st * self.scale
```

注意：这只是示意，正式实现应放在 `model.py` 当前 `LoRALinear.forward` 的位置，并保持 `amp`、`sketch`、`sketch_st` 三种模式互斥。目标只有一个：

> 前向等价于 FSLoRA，梯度等价于 amp。

这样：

- `amp vs fslora_st` 只隔离“state 是否被 sketch”；
- `amp vs fslora_hajek` 则保留“FSLoRA 原论文式真实梯度”的完整差异。

### 修改方案 B（最低成本）：只加诊断，不改方法

如果不想增加完整 3 seed run，至少：

1. 在 batch 2 报告中记录 early/mid/late 三段；
2. 增加一个 `eps=1e-12` 的 s2040 短诊断；
3. 记录 FSLoRA 和 amp 在同一 batch 上的梯度范数比；
4. 在论文中明确：

> 冷启动等价性是 Adam 的理想极限；在 `eps=1e-8` 和梯度裁剪下只是近似。

### 验收

至少增加一个测试：

```python
def test_fslora_st_matches_amp_at_cold_start(...):
    # B = 0, 同一 batch, 同一 sets, 同一 seed
    # 一步本地训练后，params 应近似相等（例如 atol=1e-5）
```

同时保留现有测试：

- `test_fslora_sketch_forward_and_gradients`
- `test_batch2_local_variants`

---

## 3. P1：FLASC 的强端上传预算必须写清楚

### 现状

当前 `flasc` 的实现是：

- 弱端：完整训练，Top-K 上传 `r/R` 的 LoRA 元素；
- 强端：完整训练，**全量上传**；
- 服务器：zero-fill 平均。

这和“所有客户端同样字节预算的 FLASC”不是同一件事。

### 修改

二选一，并在 README/报告中明确：

#### 方案 A：定义为异构预算 baseline

写出：

> 强端具有 8/8 上传预算，弱端具有 2/8 上传预算；因此这是 heterogeneous-budget FLASC-style baseline，不是同预算 FLASC。

这是最省事的做法，但论文中不能再说“同等字节”。

#### 方案 B：增加同预算 FLASC 对照

增加配置项，例如：

```json
"weak_upload": "topk",
"strong_upload": "topk",
"upload_fraction": 0.25
```

并保证：

- 所有客户端都只上传相同比例的 LoRA 元素；
- 头部上传规则单独说明；
- 服务器 zero-fill 分母与参与客户端数一致。

### 还需要核对 Top-K 粒度

当前 `topk_sparsify(delta, fraction)` 是对**整个 `[J, R, D]` 张量全局取最大幅值**。请核对 FLASC 原文是按：

- 全模型全局；
- 每层；
- 每个 LoRA 矩阵；

中的哪一种。如果实现粒度不同，应把 baseline 标为“FLASC-style global top-k”，不能称为原文方法。

### 验收

- `identity.json` 中记录 `weak_upload` 和 `strong_upload`（若有）；
- 测试断言 Top-K 保留下来的元素数正确；
- 报告中写明总上传字节数或参数数，而不只是“相同预算”。

---

## 4. P1：P3/P4/P5 只能作为外部定位对照

当前 batch 2 的 P3–P5 是：

```text
P3: amp vs fslora_paper   (random + zerofill)
P4: amp vs fedplora      (random + owner)
P5: amp vs flasc          (full train + topk)
```

这些对比同时改变了 local training、allocation、aggregation 或多个因素，不能作为机制证明。

### 修改

在 `batch2.py` 和报告中区分两类判据：

#### 机制对照

- `amp vs fslora_hajek`：同一 allocation + aggregation，隔离 state sketch；
- `amp vs lrcomp`：同一 allocation + aggregation，隔离 forward amplification vs local lr；
- `fslora_hajek vs fslora_zerofill`：同一 forward + allocation，隔离 aggregation。

#### 外部定位

- `amp vs fslora_paper`；
- `amp vs fedplora`；
- `amp vs flasc`。

外部定位可以写“与已有方法风格相比的表现”，不能写“证明 amplification 机制有效”。

另外，当前 FSLoRA 对照使用 AdamW；FSLoRA 原论文的局部更新是 SGD。报告中应写：

> FSLoRA-style under our unified AdamW protocol

除非另外运行了 SGD 版本，否则不要写“FSLoRA as published”。

---

## 5. P1：统计报告增加 per-seed 配对差和区间

当前 `verdicts2` 只用 3 个 seed 的均值与 0.3pp 阈值判断，容易受单个 seed 影响。

建议在 `batch2.py` 增加：

1. 每个 seed 的 `amp − control` 差值；
2. 3 个 seed 是否同向；
3. paired mean 和 95% 区间；
4. 若可能，主结论再补 2 个 seed。

报告表建议至少包含：

```text
method | s2040 | s2041 | s2042 | mean | paired mean diff vs amp | 95% CI | direction
```

判据不要只写 `True/False`，要写：

- 均值是否超过阈值；
- 有几个 seed 同向；
- 区间是否跨 0。

---

## 6. P2：其他代码与审计改进

### 6.1 记录 source code hash

`identity.json` 目前有 plan、数据摘要、模型 revision、torch 版本和 GPU，但没有本实验源码 hash。建议增加：

```json
{
  "code_hashes": {
    "scra/model.py": "...",
    "scra/core.py": "...",
    "scra/d0.py": "...",
    "scra/batch2.py": "..."
  }
}
```

否则以后无法证明结果对应哪一版实现。

### 6.2 修正 `amp` 注释

`model.py` 中 `amp` 的注释容易让人误解为“梯度是放大模型的真实梯度除以 α”。实际上当前实现是：

- 前向值放大；
- 额外项 `detach`；
- 参数梯度走未放大项。

建议改成明确表述：

> forward-only amplification with detached increment; gradients follow the unamplified product.

### 6.3 增加 `topk_sparsify` 输入检查

建议增加：

```python
if not (0.0 < fraction <= 1.0):
    raise ValueError("fraction must be in (0, 1]")
```

并测试边界情况。

### 6.4 FLASC 诊断聚合字段

`weak_upload="topk"` 时，当前 `d0.py` 会把 `weak_sets` 改成全分量集合，导致 `owner` / `hajek` 诊断量把零元素也当作“已贡献”。最终 `zerofill` 不受影响，但如果以后用这些诊断字段解释机制，会得到错误结论。

建议：

- 单独保存一个 `uploaded_mask`；
- 或明确在 FLASC run 中禁用 `owner`/`hajek` 诊断；
- 或把 `weak_sets` 保持为实际非零上传位置。

### 6.5 增加 cold-start equivalence 测试

新增测试：

```text
B=0 时：
  amp 与 fslora_st 的一步更新近似一致
  amp 与 fslora_hajek 的梯度范数比接近 R/r
```

这不是为了把 FSLoRA 改成 amp，而是为了在论文中准确说明 A1 到底比较了什么。

---

## 7. 建议修改后的实验清单

当前 15 个 run 之外，建议至少增加：

| Run | 数量 | 用途 |
|---|---:|---|
| `fslora_zerofill_s2040/2041/2042` | 3 | 同 allocation、同 forward，隔离 zerofill vs Hájek |
| `fslora_st_s2040` | 1（可选 3） | 只保留 state sketch 差异，隔离梯度尺度 |
| `flasc_samebudget_s2040` | 1（可选 3） | 如果论文要声称同等通信预算 |

如果只做最小修改，至少完成：

1. 3 个 `fslora_zerofill`；
2. 1 个 `fslora_st` 诊断；
3. 报告改为 per-seed + 区间；
4. README 写明 FLASC 预算与 FSLoRA 优化器差异。

修改后 plan 数会增加，需要同步更新：

- `make_batch2.py` 的 `METHODS`；
- `run_batch2.sbatch` 的 `--array=0-N%2`；
- `scra/batch2.py` 的 `METHODS`、`LABELS`、判据；
- README 中的“15 次运行”数字。

---

## 8. 提交前检查清单

```text
[ ] 新增 fslora_zerofill 的 3 个 plan
[ ] 新增 fslora_st 的 1 个诊断 plan（或至少 eps 敏感性诊断）
[ ] batch2.py 支持新方法并输出 per-seed 差值
[ ] FLASC 强端预算明确写入 README / identity
[ ] P3–P5 标为 external baseline
[ ] identity.json 写入 code hash
[ ] model.py 中 amp 注释修正
[ ] topk_sparsify 输入检查
[ ] pytest 全绿
[ ] 先跑 s2040 一行 smoke test，再跑完整 array
```

本地测试命令：

```bash
cd /Users/liming/Documents/FYP/federated_finetuning/experiments/scra_diagnostics_20261017
../../.venv/bin/python -m pytest -q tests -p no:cacheprovider
```

当前基线测试结果：

```text
26 passed
```

---

## 9. 不要从当前代码直接写出的结论

在完成上述修改前，不应写：

- “A1 已经证明 state sketch 有害”（还有梯度尺度和 epsilon 混淆）；
- “P3 已经证明我们的方法优于 FSLoRA 原论文”（当前是 AdamW + unified FedAvg 的 FSLoRA-style 实现）；
- “FLASC 是同等通信预算对照”（当前强端全量上传）；
- “P4 证明 amplification 机制有效”（P4 是 random + owner 的混合外部基线）；
- “3 seed 均值超过 0.3pp 就是显著”（需要 per-seed 差值和区间）。


---

## 10. 处理记录（2026-10-18）

| 条目 | 处理 | 位置 |
|---|---|---|
| P0 §1 fslora_zerofill | **采纳**，3 个 seed | `make_batch2.py`、`scra/batch2.py`（A1c） |
| P0 §2 梯度尺度 / eps 混淆 | **采纳并复现**：小模型 B = 0 走一步，eps = 1e-8 时相差 12.4%，1e-10 时 1.1%，1e-12 时 0.01%，与梯度裁剪无关。加入 `weak_forward="fslora_st"`（前向同 FSLoRA，梯度同 amp），3 个 seed。单元测试确认冷启动时 fslora_st 与 amp 的整个本地训练一致 | `scra/model.py`、`scra/core.py`、`scra/d0.py`、测试 |
| P1 §3 FLASC 预算 | **部分采纳**：采用方案 A，定义为异构预算的 FLASC 式对照。各客户端预算与 amp 一致，这正是正确的比较；不增加同预算运行。标明全局 Top-K，索引开销未计入 | README、`d0.py` 文档字符串、`core.topk_sparsify` |
| P1 §4 P3–P5 为外部定位 | **采纳**：报告分为"机制对照"和"外部定位"两节；fslora_paper 标为"我们统一的 AdamW 协议下的 FSLoRA 式" | `scra/batch2.py` |
| P1 §5 统计 | **采纳**：逐 seed 配对差值、均值 ± 标准差、95% t 区间、同向计数；判读用"均值阈值 + 所有 seed 同向" | `scra/batch2.py` |
| P2 §6.1 代码哈希 | **采纳**：写入 `identity.json` 的 `code_hashes`，不参与续跑的身份校验 | `scra/d0.py` |
| P2 §6.2 amp 注释 | **补充措辞**：原注释在数学上正确（detach 后的梯度精确等于放大模型真实梯度的 1/α），补了"只在前向放大、梯度走未放大的乘积"这一更直白的说法 | `scra/model.py` |
| P2 §6.3 Top-K 输入检查 | **采纳** | `scra/core.py`、测试 |
| P2 §6.4 FLASC 诊断字段 | **采纳为说明**：不影响训练和结果，在文档字符串和 README 中注明 | `scra/d0.py`、README |
| §8 冒烟测试 | **采纳**：README 中先提交 `--array=0`，确认后再提交 `--array=1-20%2` | README |

运行数从 15 变为 21；`run_batch2.sbatch` 为 `--array=0-20%2`。测试：27 passed。
