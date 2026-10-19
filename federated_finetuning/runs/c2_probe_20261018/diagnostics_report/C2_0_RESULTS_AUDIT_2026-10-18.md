# C2-0 解码器大模型探针结果审查

日期：2026-10-18
数据目录：`federated_finetuning/runs/c2_probe_20261018/`
审查对象：`c2check`、`c2snap`、3 个 `c2probe` 快照结果、`REPORT_C2_0.md`、`report_c2_0.json`

状态：**结果已完整生成；本文件是对结果和解释的独立审查，不替代运行日志或原始 HPC 调度记录。**

---

## 0. 结论摘要

这次 C2-0 的结果是**混合结果**，不能简单写成“方法在解码器大模型上验证成功”。

| 判据 | 结果 | 含义 |
|---|---|---|
| M0：受限训练出现明显过冲 | **False** | 按预注册的 aggregate product/function step 口径，3 个快照里只有 1 个满足 |
| M1：放大修正更新尺度 | **False** | aggregate amp product step 接近 1，但 per-client amp product step / cos 没有达到预注册阈值 |
| M2：一步准确率信号 | **True** | 第 5、30 轮快照上 L2/hajek 明显掉点，amp 接近 vref；第 0 轮不满足 |
| 可学习性 gate | 3/3 通过 | snapshot round 0/5/30 的 core accuracy 都 ≥ 45% |

因此最准确的表述是：

> 在 Qwen2.5-0.5B 上，受限训练在中后期快照上造成了可观测的一步聚合准确率损失，amp 在这些快照上接近 vref；但预注册的 per-client 更新尺度和过冲机制没有被确认。
> **任务层面的信号存在，但方法声称的单客户端机制没有在这个大模型设定下得到验证。**

---

## 1. 完整性检查

已检查：

- `c2check/check.json`
- `c2check/identity.json`
- `c2snap/identity.json`
- `c2snap/snapshot_run.json`
- `c2probe/round_0000.json`
- `c2probe/round_0005.json`
- `c2probe/round_0030.json`
- `diagnostics_report/REPORT_C2_0.md`
- `diagnostics_report/report_c2_0.json`

结果：

- label words：` yes` / ` maybe` / ` no`，三个都是单 token；
- label token ids：`[9834, 7196, 902]`；
- adapters：**168**；
- LoRA 参数：**35.19M**；
- train / validation 行数：**372557 / 19681**；
- `max_len`：256；
- 3 个 probe 文件均为 **12 draws**；
- 没有发现 NaN / 非有限指标；
- 所有 probe 的 identity 中记录 plan hash、model revision、cache hash 和 source hashes；
- 当前源码 hash 与 c2check identity 一致；
- 本地测试：**32 passed**。

### snapshot run

| round | all acc | core acc | NLL |
|---:|---:|---:|---:|
| 0 | 58.25% | 57.75% | 4.1033 |
| 10 | 81.45% | 80.08% | 0.5087 |
| 20 | 81.80% | 80.25% | 0.4906 |
| 30 | 83.65% | 82.50% | 0.4747 |

可学习性 gate 通过，snapshot 不是随机模型。

### 运行成本

- `c2check` peak：约 2087 MiB（明确注明不是资源证据）；
- snapshot：约 725 秒，peak 约 5727 MiB；
- 三个 probe：约 1091 / 1140 / 1173 秒，peak 约 5935 / 5945 / 5956 MiB。

这些 peak 是包含多个状态/路径的诊断运行，不等于 C1 的折叠实现，因此不能用于声称 10–14% 显存节省。

---

## 2. 三个快照的关键数值

### 2.1 M0：过冲判据

M0 定义为：

> `hajek` aggregate product step 或 function step ≥ 1.2，3 个快照中至少 2 个满足。

实际：

| snapshot | hajek product step | hajek function step | M0 单项 |
|---:|---:|---:|---|
| 0 | 0.83 | 1.02 | 否 |
| 5 | 0.72 | **2.53** | 是（function） |
| 30 | 0.79 | 1.12 | 否 |

所以 M0 **False**：只有第 5 轮快照的函数空间出现明显大于 1 的步长；第 0、30 轮没有达到预注册条件。

这不是“完全没有异常”，而是：

> **预注册的“沿 r8 方向的过冲”判据没有稳定抓住这个模型上的失败模式。**

---

### 2.2 M1：放大修正

M1 要求：

- aggregate amp product step ∈ [0.9, 1.1]；
- per-client amp product step ∈ [0.95, 1.05]；
- per-client amp product cos ≥ 0.95。

实际 aggregate amp product step：

| snapshot | aggregate amp product step |
|---:|---:|
| 0 | 0.96 |
| 5 | 0.93 |
| 30 | 0.96 |

这部分看起来接近 1。

但 per-client 中位数：

| snapshot | restricted product step | restricted cos | amp product step | amp cos |
|---:|---:|---:|---:|---:|
| 0 | 0.41 | 0.41 | 0.79 | 0.79 |
| 5 | 0.66 | 0.62 | 0.90 | 0.90 |
| 30 | 0.79 | 0.74 | 0.94 | 0.94 |

因此 M1 **False**。

这一点很关键：

> **聚合层面 amp 接近 r8，但单个弱端的 LoRA 更新并没有恢复到完整训练的方向和尺度。**

这不是简单的“服务器平均后看起来对了”，而是聚合平均掩盖了 per-client 的方向差异。

---

### 2.3 M2：一步准确率信号

M2 要求：

- `L2` 或 `hajek` 的 core diff ≤ −0.5pp；
- `|amp − vref|` ≤ 0.2pp。

实际：

| snapshot | L2 core diff | hajek core diff | amp core diff | amp − vref |
|---:|---:|---:|---:|---:|
| 0 | −17.97 | −21.01 | −6.30 | **−5.83** |
| 5 | −6.90 | −12.26 | −0.14 | +0.03 |
| 30 | −3.42 | −4.45 | −0.33 | +0.01 |

第 5、30 轮满足，第 0 轮因 amp 与 vref 相差 −5.83pp 不满足；3 个快照中 2 个满足，因此 M2 **True**。

解释：

- 中后期快照上，受限训练确实造成一步准确率损失；
- 同一快照上 amp 的一步准确率接近 vref；
- 但第 0 轮不成立，说明冷启动/早期状态下修复并不稳定。

---

## 3. 结果真正说明了什么

### 3.1 任务层面：有信号，但不是完整机制验证

M2 为 True 说明：

> 在 Qwen2.5-0.5B 的中后期 snapshot 上，受限训练的一步聚合结果确实比 r8 差，而 amp 的一步 core accuracy 接近 vref。

这是一个值得保留的的正向信号。

但它不能说明：

- per-client 更新方向/尺度已经匹配；
- α=8 的机制在解码器模型上成立；
- 多轮训练会保持这个收益；
- 方法可以泛化到其他模型或任务。

### 3.2 机制层面：更新方向差异比“过冲”更突出

在这个模型上，per-client restricted product step 和 cosine 都远低于 D3 中 RoBERTa 的数值：

- RoBERTa D3：restricted cos 约 0.89–0.93，amp 约 0.99–1.00；
- Qwen C2-0：restricted cos 约 0.41 / 0.62 / 0.74，amp 约 0.79 / 0.90 / 0.94。

这意味着：

> **Qwen 上的差异不只是“更新太大/太小”，而是受限训练的方向本身偏离完整训练。**

amp 改善了方向（cos 从 0.41→0.79、0.62→0.90、0.74→0.94），但没有达到预注册的 0.95。

### 3.3 M0 False 与 M2 True 并不矛盾，但暴露了指标问题

M0 只看 aggregate hajek 的 product/function step 是否 ≥1.2。
M2 看 core accuracy 是否下降、amp 是否接近 vref。

实际出现了：

- M0 False：幅度型“过冲”指标没有稳定超过 1.2；
- M2 True：任务准确率却明显下降，amp 在中后期接近 vref。

可能的解释：

1. 受限训练的失败主要是**方向偏差**，而不是沿 r8 方向的幅度过冲；
2. per-client 方向和聚合平均之间的抵消/放大，使 aggregate step 接近 1，但任务损失仍然存在；
3. 函数空间 probe 的 64 条样本对方向变化不够敏感；
4. 没有新分类头时，压力分布与 RoBERTa 不同。

这些都需要进一步分析，不能只凭 M2 就宣称机制成立。

---

## 4. 对当前报告解读的修正

现有 `REPORT_C2_0.md` 的数值是正确的，但它的通用决策文本没有覆盖当前这个组合：

```text
M0 = False
M1 = False
M2 = True
```

旧文本大致写：

- M0 成立、M1 成立 → 进入 C2-1；
- M0 成立、M1 不成立 → 检查 per-client 假设；
- M0 不成立 → 论文缩小适用范围。

但当前结果是“**任务信号存在，但 per-client 机制不成立**”。因此建议新增一个明确分支：

> **M0 False + M1 False + M2 True：**
> 只能写“聚合层面出现了一步任务损失，amp 在中后期接近 vref”；
> 不能写“放大机制在解码器大模型上成立”。
> 下一步应先做 per-client 方向和 α 的诊断，再决定是否进入 C2-1。

---

## 5. 下一步建议

### 5.1 先做 α sweep，不要直接上 C2-2

在 Qwen 的第 5、30 轮 snapshot 上测试：

```text
α ∈ {4, 8, 16, 32}
```

每个设置测：

- per-client product step / cos；
- aggregate product step；
- core / all accuracy；
- 函数空间 step；
- amp − vref。

目的：判断 per-client cos 0.79–0.94 是因为：

- α=8 不够；
- 还是前向放大无法修复方向偏差；
- 还是无分类头设置本身改变了机制。

### 5.2 分解强端/弱端贡献

当前 core accuracy 是 30 个核心弱端；建议额外报告：

- 强端 core/all；
- 被降级弱端；
- 每个客户端自己的 amp − r8 / L2 − r8；
- 聚合更新中强端和弱端的加权贡献。

否则无法判断 amp 的聚合收益是否由强端主导。

### 5.3 不要直接进入 C2-2 资源实验

C2-0 的 peak memory 不是资源证据，且 M0/M1 未确认。
在机制没有澄清前，先不要投入 C2-2 的折叠实现和大规模资源测量。

### 5.4 C2-1 可以作为探索性短跑

如果时间允许，可以跑短 C2-1，但应标记为：

> exploratory multi-round check

而不是“机制已确认后的正式验证”。只有在 α sweep 和 per-client 分析找到更接近完整训练的配置后，才把 C2-1 解释为方法验证。

---

## 6. 可写的结论与不可写的结论

### 可以写

> 在 Qwen2.5-0.5B 的 C2-0 单步探针中，受限训练在中期和后期快照上造成 core-weak 一步准确率下降，放大受限训练在这些快照上接近 vref；但 per-client 更新方向和尺度没有恢复到预注册阈值，且早期快照的 amp 未匹配 vref。

> 解码器模型上的失败模式不能只用“过冲”描述；per-client 方向偏差更突出。

### 不能写

- “C2-0 证明方法在解码器大模型上有效”；
- “α=R/r 已在大模型上验证”；
- “amp 修复了 per-client 更新”；
- “C2-0 证明了 10–14% 显存节省”；
- “M2 True 就代表方法成功”。

---

## 7. 最终判断

**C2-0 完成度很好，结果有信息量，但它是一个 mixed/partially negative mechanism result，不是 clean success。**

- 任务层面：有一步准确率信号（M2 True，2/3 gated snapshots）。
- per-client 机制：没有确认（M1 False）。
- 过冲判据：没有确认（M0 False）。
- 资源节省：C2-0 不测，不能声称。
- 下一步：先做 α sweep、per-client 方向分析和 strong/weak 分解；在机制澄清前，不要直接把 C2-2 大规模资源实验当作水到渠成的下一步。
