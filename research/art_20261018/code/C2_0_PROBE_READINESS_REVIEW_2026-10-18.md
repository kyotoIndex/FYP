# C2-0 解码器大模型探针：代码与实验就绪度评估

日期：2026-10-18  
评估对象：`federated_finetuning/experiments/scra_diagnostics_20261017/` 下的 C2-0 probe  
上游设计：`docs/designs/c2_0_probe_design_2026-10-18.md`  
资源核算：`docs/designs/resource_case_2026-10-18.md`  
交接文档：`HANDOFF_C2_0_ASSETS.md`

状态：**代码与设计审查报告；本地没有 Qwen2.5-0.5B 的真实模型权重和 C2 结果，因此本报告不评价任务效果，只评价实现、实验设计和 HPC 就绪度。**

---

## 0. 结论

### 总体判断

**C2-0 的代码可以进入 HPC 的 asset check 和小规模 smoke test，但还不能直接当成“大模型验证结果”或“资源效率证据”。**

本地证据：

```text
31 passed
```

实现层面没有发现会直接阻塞运行的硬 bug；主要问题集中在四类：

1. **C2-0 不能验证显存节省**：当前代码仍保留全部 R 个 LoRA 分量和 Adam 状态，C1 的 10–14% 节省只是折叠实现下的估计；
2. **缺少 identity / hash**：模型、tokenizer、数据、prompt cache、源码和 snapshot 的版本没有完整绑定；
3. **缺少 resume/checkpoint**：snapshot 或 probe 中断会丢进度；
4. **缺少 snapshot 可学习性门槛**：如果 30 轮后模型仍接近随机，M2 的一步准确率判读没有意义。

### 建议状态

| 项目 | 状态 |
|---|---|
| 本地 CPU 单测 | 通过（31/31） |
| 代码结构 | 可运行 |
| 单步机制设计 | 合理，与 D2/D3 口径一致 |
| 机制判读 M0/M1 | 可作为 C2-0 的主要终点 |
| 一步准确率 M2 | 需要先验证 snapshot 是否学到任务 |
| 显存/计算节省 | **当前 C2-0 不测，不能声称** |
| 论文级复现性 | 还缺 identity/hash/resume |
| HPC 提交建议 | 先 `c2check`，再 `snap`，再单个 `probe` smoke test |

---

## 1. 审查范围

本次检查的文件：

- `scra/llm.py`
- `scra/c2probe.py`
- `plan_c2.json`
- `run_c2.sbatch`
- `tests/test_c2.py`
- `HANDOFF_C2_0_ASSETS.md`
- `docs/designs/c2_0_probe_design_2026-10-18.md`
- `docs/designs/resource_case_2026-10-18.md`
- `docs/designs/checks/c1_resource.py`

检查内容：

- Qwen2.5-0.5B 包装与 LoRA 安装
- MNLI → label-word next-token 数据路径
- snapshot → probe → report 流程
- 分组 block、乘积空间 Gram、函数空间 probe
- 预注册判据 M0/M1/M2
- C1 资源核算口径
- 本地测试覆盖

没有执行的内容：

- 真实 Qwen2.5-0.5B 前向/训练
- CUDA / 4090 显存与时间测试
- HPC 上的 snapshot / probe
- 真实 MNLI tokenizer 的 label token 检查

因此，本报告是**就绪度评估**，不是效果评估。

---

## 2. 设计层面：做得好的部分

### 2.1 任务与模型选择合理

- 使用 Qwen2.5-0.5B Base；
- 没有从零训练的新分类头；
- 输出层使用冻结的词嵌入；
- 将 MNLI 组织成标签词的下一个 token 预测；
- 这样能直接检验“没有新分类头时，受限训练是否仍然过冲”。

### 2.2 R=64 / r=8 的设定符合 C1 的资源核算

C1 的结论是：

- 通信量和 LoRA 训练状态严格按 r/R 缩放；
- 当 R=64、r=8 时，折叠实现下的峰值显存估计节省约 10–14%；
- 计算量主要受冻结主干影响，节省有限。

选择 R=64 / r=8 可以让资源差异更明显，方向正确。

### 2.3 单步探针口径清楚

`scra/c2probe.py` 中包含：

- 单客户端乘积空间：受限/放大 vs 同一客户端完整训练限制到同一组分量；
- 聚合后乘积空间：与同一 draw 的 r8 比较；
- 函数空间：固定 probe batch 上三个标签词的 log-prob 投影；
- 一步后的 core-weak accuracy / NLL；
- 规则覆盖 r8、vref、L2、hajek、amp。

这和 RoBERTa 的 D2/D3 分解是同一类分析，便于跨模型对照。

### 2.4 预注册判据已经写好

M0、M1、M2 在运行前写出，且有明确的成立/不成立后续动作。这是正确的实验纪律。

### 2.5 关键代数与流程有单测

`tests/test_c2.py` 覆盖了：

- 标签词单 token；
- prompt cache；
- 分组 block 的 roundtrip；
- `product_gram` 与显式矩阵构造一致；
- α=1 退化；
- snapshot → probe → report 全流程；
- 小 Qwen CPU 端到端。

本地结果：

```text
31 passed
```

---

## 3. 必须在 HPC 前修正的问题

### P0-1：C2-0 不能测显存节省

当前 `LLMTaskModel` 和 `LoRALinearFP32` 的实现是：

- 所有 R 个 LoRA 分量仍然是可训练参数；
- 未分配分量的梯度被清零，但参数和 Adam 状态仍然存在；
- 因此训练 r 个分量和训练 R 个分量的峰值显存不会体现 C1 的 10–14% 节省；
- amp 还需要保存一份起点副本，显存可能略高。

C2-0 只能回答：

> 受限训练在解码器模型上是否仍然过冲，放大是否修正位移尺度。

它不能回答：

> 真实折叠实现后显存是否节省 10–14%。

建议：

- 在 C2-0 报告和论文表格中明确写“C2-0 不测资源节省”；
- C2-2 再实现“未训练分量折进冻结权重”的版本，并实测峰值显存、每步时间和通信量；
- 不要用 C2-0 的 `peak_memory_mib` 去支撑 C1 的数字。

### P0-2：缺少 identity / hash

现在 C2 的 snapshot/probe 输出没有完整记录：

- 源码 hash；
- 模型 revision；
- tokenizer hash；
- manifest hash；
- prompt cache hash；
- label token ids；
- snapshot 文件 hash；
- plan hash。

建议新增 `identity.json`，至少包含：

```json
{
  "plan_id": "...",
  "model_name": "models/qwen2.5-0.5b",
  "model_revision": "...",
  "tokenizer_hash": "...",
  "manifest_hash": "...",
  "cache_hashes": {"train.pt": "...", "validation.pt": "..."},
  "label_token_ids": [..],
  "code_hashes": {"scra/llm.py": "...", "scra/c2probe.py": "..."},
  "snapshot_sha256": "..."
}
```

否则无法证明 snapshot、probe 和 report 来自同一版本。

### P0-3：缺少 checkpoint / resume

当前：

- `run_c2_snapshots` 只在 30 轮结束后写 `snapshot_run.json`；
- `run_c2_probe` 只在 12 个 draw 全部结束后写 `round_*.json`。

后果：

- snapshot 中断会从第 0 轮重跑；
- probe 中断会从第 1 个 draw 重跑；
- 1000 次本地训练的成本不能接受这种损失。

建议：

- snapshot 每 5 轮保存 checkpoint（至少已有 snapshot 文件也要有 round/stream 位置）；
- probe 每完成 1 个 draw 就写 partial JSON，并记录已完成 draw；
- 最终完成后写 `complete.json`。

### P0-4：先验证 snapshot 是否真的学到了任务

M2 检查“L2/hajek 一步后准确率下降、amp 接近 vref”。  
但如果 30 轮全秩 FedAvg 后模型仍接近随机，M2 的准确率差就没有解释力。

建议在 `c2check` 和 `snapshot_run` 中记录：

- round 0 的 core / all accuracy；
- round 30 的 core / all accuracy；
- 随机基线（3 类约 33.3%）；
- 若 round 30 仍接近随机，先增加 snapshot 轮数、检查 prompt/label 或改动任务，不要直接解释 M2。

建议在正式运行前冻结一个门槛，例如：

> round 30 的 core accuracy 必须明显高于 33.3%；否则 C2-0 只报告 M0/M1 机制，不报告 M2 任务收益。

### P1-1：prompt cache 复用前必须校验

`build_prompt_cache` 现在只要 `meta.json` 存在就直接返回，不检查：

- tokenizer 是否改变；
- 模型 revision 是否改变；
- manifest 是否改变；
- train.pt / validation.pt 是否还存在或损坏。

建议在 meta 中写 hash，并在启动时校验；不匹配时报错或重建。

### P1-2：`LLMData` 需要验证 train cache 与 manifest 的对应关系

`Dataset` 会检查：

- train/validation 行数；
- 每个 train index 属于 manifest 的 client；
- train/validation group 不重叠。

`LLMData` 现在只检查 validation 行数，没有验证 train cache 的顺序和 client 归属。  
如果 prompt cache 与 manifest 行顺序错位，代码会静默训练错数据。

建议把 `Dataset` 中的检查迁移或复制到 `LLMData`。

### P1-3：`c2check` 结果要落盘

现在 `run_c2_check` 只打印 JSON。建议写到：

```text
runs/c2_probe_20261018/c2check/check.json
```

至少包含：

- `label_token_ids`；
- 三个标签词是否单 token；
- `adapters` 数量（Qwen2.5-0.5B 应为 168）；
- LoRA 参数数量（约 35M）；
- train / validation 行数；
- `max_len`；
- 一次前向 shape；
- 一步训练 loss 是否 finite；
- peak memory。

### P1-4：成本估计需要先实测

当前估计：

- snapshot 30–45 分钟；
- 每个 probe 40–60 分钟；
- 合计 2–2.5 GPU 小时。

建议先只提交：

```bash
sbatch --time=00:20:00 run_c2.sbatch check
sbatch --time=01:30:00 run_c2.sbatch snap
sbatch --time=02:00:00 --array=0 run_c2.sbatch probe
```

观察一个 probe 任务的实际耗时和显存后，再提交 array=1-2。不要一开始就直接 `--array=0-2`。

---

## 4. 统计与解释注意事项

### 4.1 M2 需要看 per-draw 分布

12 个 draw 的样本量不算大。M2 建议同时报告：

- 每个 draw 的 paired diff；
- 均值 ± 标准差；
- 正负方向数量；
- 是否所有 draw 同向。

如果 M2 边缘通过，不要写成强结论。

### 4.2 M1 验证的是位移尺度，不是最终收益

M1 检查：

- amp aggregate product step ∈ [0.9, 1.1]；
- 单客户端 amp product step ∈ [0.95, 1.05]；
- cos ≥ 0.95。

这只能说明“放大后位移尺度与完整训练匹配”，不能单独证明任务收益。任务收益由 M2 和多轮 C2-1 决定。

### 4.3 “没有分类头”的表述要严谨

Qwen2.5 的输出词嵌入是冻结的，但标签词概率仍然通过输出层计算。  
论文里建议写：

> 没有从零训练的新分类头，但冻结的输出词嵌入仍参与前向和损失计算。

不要写“完全没有头部”。

### 4.4 C1 数字是纸面估计

C1 的峰值显存和 FLOPs 是粗估：

- 激活和 checkpoint 公式是近似；
- 当前代码没有折叠未训练分量；
- 实际显存应通过 C2-2 实测。

C1 可以作为设计依据，不能当成实验测量。

---

## 5. HPC 提交前检查清单

```text
[ ] 同步 scra/llm.py、scra/c2probe.py、plan_c2.json、run_c2.sbatch 到 HPC
[ ] 下载 Qwen/Qwen2.5-0.5B，并记录 revision
[ ] 确认 data/mnli_r1_seed2026/{train,validation}.jsonl 存在
[ ] 先跑 c2check，确认 label_token_ids 是三个单 token
[ ] 确认 adapters == 168
[ ] 确认 lora 参数约 35M
[ ] 确认 train/validation 行数和 manifest 一致
[ ] 确认 max_len <= 256
[ ] c2check 结果落盘
[ ] 跑 snapshot，检查 round 0 / round 30 准确率
[ ] round 30 不得接近随机，否则先修任务/增加轮数
[ ] 只跑一个 probe 任务，记录实际耗时和显存
[ ] 再跑其余 probe 任务
[ ] 生成 REPORT_C2_0.md，若 M2 不成立要区分“机制存在但一步看不见”和“任务没学会”
```

---

## 6. 建议的最终表述

### 可以写

> C2-0 是解码器大模型上的单步机制探针，用于检验受限训练是否仍产生过冲、放大是否修正单客户端和聚合后的位移尺度。它使用预训练 Qwen2.5-0.5B、R=64/r=8、冻结主干与输出嵌入，不训练新的分类头。

### 不能写

- “C2-0 证明了大模型上省 10–14% 显存”；
- “C2-0 已经证明方法在大模型上提升准确率”；
- “没有分类头”被解释为“没有任何输出层计算”；
- “M2 失败”被解释为“方法失败”，除非先确认 snapshot 已学会任务。

---

## 7. 最终建议

**建议按以下顺序推进：**

1. 先补 identity/hash、c2check 落盘、cache/manifest 校验、resume/checkpoint；
2. 运行 `c2check`；
3. 运行 `snap`，检查 round 30 是否脱离随机；
4. 只跑一个 `probe` smoke test；
5. 确认耗时、显存、M0/M1 方向后，再跑完整 3 个 probe；
6. 结果只作为机制诊断；资源节省留给 C2-2 的折叠实现实测。

**一句话：C2-0 代码可跑、设计方向正确，但它是机制探针，不是资源效率或最终性能验证。先做 smoke test 和可学习性门槛，再投入完整 HPC 资源。**

---

## 8. 处理记录（2026-10-18）

| 条目 | 处理 | 位置 |
|---|---|---|
| P0-1 不测显存 | 不改代码（设计文档 §7 第 4 条和资源核算 §5 已写明），在报告、check.json、交接文档和模块文档字符串中明确写出"本探针不测资源节省" | `scra/c2probe.py`、`HANDOFF_C2_0_ASSETS.md` §7 |
| P0-2 身份与哈希 | 每个阶段写 `identity.json`：plan 和它的哈希、源码哈希、模型目录、`REVISION.txt`、`config.json` 哈希、缓存指纹与缓存文件哈希、标签词 token id、版本、主机、GPU。快照哈希写入 `snapshots/hashes.json`，探针读取前核对，不一致就拒绝 | `scra/c2probe.py` `identity`、`run_c2_snapshots`、`run_c2_probe` |
| P0-3 续跑 | 快照每 `checkpoint_every`（5）轮保存检查点（全局状态、数据流位置、评测记录），探针每轮模拟后写部分结果；重启时继续。测试确认：中断后续跑与一次跑完的结果逐位相同 | 同上；`tests/test_c2.py` |
| P0-4 可学习性门槛 | 采纳，门槛定为核心弱端准确率 ≥ 45%，写进 `plan_c2.json`（`learnable_min_core_acc`）。M2 只在通过门槛的快照上判读，没有快照通过时记为"无法解读"。快照运行和 c2check 都记录核心弱端准确率 | `scra/c2probe.py` 报告部分、设计文档 §5 |
| P1-1 缓存校验 | 缓存元数据中记录 manifest、train / validation jsonl、tokenizer 文件的哈希，以及 `max_len`、模板和标签词；复用时全部核对，并核对缓存文件本身的哈希 | `scra/llm.py` `cache_fingerprint`、`build_prompt_cache` |
| P1-2 数据校验 | LLMData 检查训练和验证行数、训练行的客户端归属、前提组不重叠、缓存标签与 manifest 行一致 | `scra/llm.py` `LLMData` |
| P1-3 c2check 落盘 | 写入 `c2check/check.json`，并增加原始模型准确率和"一步损失有限" | `scra/c2probe.py` `run_c2_check` |
| P1-4 冒烟测试顺序 | check → snap（1.5 小时上限）→ 只跑 `--array=0` → `--array=1-2` | `run_c2.sbatch`、交接文档 §4 |
| 4.1 M2 逐轮分布 | 报告每条规则的配对差值均值 ± 标准差和正负计数，以及 amp − vref 的配对差值 | `scra/c2probe.py` |
| 4.2–4.4 表述 | 报告、设计文档和交接文档改为"没有从零训练的新分类头，冻结的输出词嵌入仍参与计算"，M1 只检验更新尺度，C1 是纸面估计 | 同上 |

测试：32 passed（新增：c2check 落盘与身份记录、缓存指纹不一致时拒绝、客户端归属错位时拒绝、快照与探针中断续跑逐位一致、快照被篡改时拒绝、可学习性门槛不通过时 M2 记为无法解读）。
