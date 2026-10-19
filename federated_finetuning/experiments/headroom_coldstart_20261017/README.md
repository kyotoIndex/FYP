# 冷启动"提升空间"实验 `headroom_coldstart_20261017_v1`

2026-10-17 创建。状态：**代码完成、CPU 测试通过，尚未提交 HPC。**

## 要回答的问题

在设计任何新的自适应 PEFT 机制之前，先测出这个设置下"可以被改进的空间"有多大：

1. **秩容量**：全员 rank8 比全员 rank2 好多少？（如果很小，任何"在 rank2 里挑子空间"的方法都没有空间）
2. **异构代价**：强端 rank8、弱端 rank2 混合时，弱端相对全员 rank8 损失多少？
3. **联邦价值**：同样的资源与训练步数下，联邦比完全本地训练好多少？加上本地微调（FedAvg+FT）后还剩多少？

之前 HetSA 与 whitespace 的问题都出在没有先测这个：HetSA 的 +11.88pp 几乎全部可由 local-only 解释（联邦只贡献约 1pp），whitespace 里 rank8 只比 rank2 好 0.0008 NLL。

## 设置

| 项目 | 值 |
|---|---|
| 模型 / 任务 | RoBERTa-base（固定 revision，离线）/ MultiNLI，Q/V LoRA，FP32，dropout 0 |
| 数据 | 已有 `data/mnli_r1_seed2026`：50 个纯 genre 客户端，R1 资源角色（government、travel 为强端，共 20 强 30 弱）；复用已有 token cache |
| 起点 | **冷启动**：预训练主干 + 随机分类头 + LoRA（A 随机、B=0），分类头参与训练与聚合 |
| 联邦 | 150 轮，每轮随机 10 个客户端（同 seed 下各方法完全相同），每端 20 步、batch 8、fresh AdamW lr 2e-4、梯度裁剪 1.0 |
| LoRA | 所有方法都构造同一个 rank-8 模型；客户端只训练前 r 个分量，其余冻结。每分量缩放固定为 2（=16/8），所以全员 rank2 恰好等价于缩放相同的 rank-2 LoRA，与 rank8 共用同一初始化 |
| 聚合 | 按样本数加权平均 A、B 因子（FedAvg-LoRA）；**分量 k 只在训练过它的参与者之间平均**，无人训练则保留服务器值；分类头全体平均 |
| 种子 | 2040、2041（训练种子；数据划分固定为 seed2026 manifest） |

### 四个方法

| 方法 | 强端训练秩 | 弱端训练秩 | 聚合 |
|---|---:|---:|---|
| `fedavg_r8` | 8 | 8 | 有 — 上限参照 |
| `fedavg_r2` | 2 | 2 | 有 — 下限参照 |
| `hetero_r8r2` | 8 | 2 | 有 — 题目中的真实异构场景 |
| `local_only` | 8 | 2 | 无 — 每端只在自己被抽中时训练，步数与联邦一致 |

三个联邦方法在最后一轮之后，另外计算 **FedAvg+FT 参照**：每个客户端从最终全局模型出发，在自己的数据上训练 20 步，再在自己的验证行上评测。这是 HetSA 实验缺失的个性化基线。它不影响训练轨迹。

### 评测

- 验证集为 manifest 的 `validation`（训练集按 premise 组留出，按客户端画像分配）；**官方 dev matched/mismatched 不读取**，保留作最终锁定集。
- 主指标：最终轮的**客户端宏平均准确率**；同时报告宏平均 NLL、样本平均、按强/弱角色、按域、按训练秩分组。
- 联邦方法用一个全局模型评测所有客户端；local-only 用各客户端自己的模型评测自己的行。
- 每 10 轮评测一次，保存完整曲线。

## 预先登记的判定规则（看结果前写定）

逐 seed 计算配对差（百分点），**两个 seed 都 ≥ 1.0pp 才算"是"**：

| 判定 | 对比 | 结论"是"意味着 |
|---|---|---|
| rank_capacity_headroom | fedavg_r8 − fedavg_r2 | 秩容量有意义，自适应分配秩/子空间值得做 |
| weak_clients_lose_under_heterogeneity | fedavg_r8 − hetero（弱端宏平均） | 异构设置下弱端确实吃亏，有可以修补的对象 |
| federation_beats_local_only | hetero − local_only | 联邦本身有价值 |
| federation_beats_local_after_personalization | hetero+FT − local_only | 个性化公平对比后联邦仍有价值 |

若"秩容量"为"否"：停止子空间/秩选择方向，论文转为提升空间与代理错位分析。只有两个 seed，区间仅作描述，不做显著性声明。

## 成本估算

依据 bridge_controls 在 4090D 上实测的每轮约 13 秒（10 端 × 20 步）：每格训练约 33 分钟，加上 16 次评测与 FedAvg+FT 约 40–45 分钟。8 格合计约 **5–6 GPU 小时**；`%2` 并发时墙钟约 2.5–3 小时，不含排队。每格申请 3 小时上限。

## 文件

```text
plan.json            全部参数（路径相对 project_root）
train.py             单格入口：--task-index（数组）或 --method/--seed
run.sbatch           Slurm 数组 0-7%2，默认可续跑
summarize.py         汇总、配对差与判定 → run_root/SUMMARY.md, summary.json
headroom/model.py    rank-8 LoRA 模型、按秩屏蔽梯度
headroom/data.py     token cache、manifest、确定性客户端数据流
headroom/fed.py      联邦轮次、分量聚合、评测、FedAvg+FT、checkpoint
tests/               微型随机 RoBERTa + 合成数据的 CPU 测试
```

任务序号 = seed 序号 × 4 + 方法序号：0–3 为 seed 2040 的 r8、r2、hetero、local；4–7 为 seed 2041。`%2` 下最先跑完的是 seed 2040 的 r8 与 r2，即最关键的容量对比。

## 本地验证（已通过）

```bash
cd /Users/liming/Documents/FYP/federated_finetuning/experiments/headroom_coldstart_20261017
../../.venv/bin/python -m pytest -q tests -p no:cacheprovider
```

6 项测试：分量聚合数值；数据流续读一致；四方法均可运行且同 seed 初始模型一致；全员 rank2 时分量 2–7 的 B 始终为 0；弱端冻结分量不被更新；中断后续跑与连续执行的最终权重**逐位相同**；数组入口跑满 2×4 格并生成汇总；合成数据上确实能学习。只用随机微型模型和合成 token，**不是 MNLI、CUDA 或效果证据**。

## HPC 提交步骤

1. 同步本目录到 HPC 的同一相对位置（`<hpc-login>` 换成你平时的登录地址）：

```bash
rsync -av --exclude '__pycache__' --exclude '.pytest_cache' /Users/liming/Documents/FYP/federated_finetuning/experiments/headroom_coldstart_20261017/ <hpc-login>:/home/student/t330025090/FYP/federated_finetuning/experiments/headroom_coldstart_20261017/
```

2. 在 HPC 上确认依赖资产存在（模型、manifest、token cache、环境）：

```bash
cd /home/student/t330025090/FYP/federated_finetuning && ls models/roberta-base data/mnli_r1_seed2026/manifest.json runs/client_influence_20261015_sgd_v1/shared_token_cache/{train,validation}.pt .venv-hpc/bin/python
```

3. 查看队列，确认 optimizer_matched 作业链（210704–210707）的状态，避免占用冲突：

```bash
squeue -u t330025090
```

4. 提交：

```bash
cd /home/student/t330025090/FYP/federated_finetuning/experiments/headroom_coldstart_20261017 && mkdir -p logs && sbatch run.sbatch
```

5. 中途超时或节点故障：用同样的命令对失败的序号重新提交，会从最近的 checkpoint（每 5 轮）续跑，并校验方法、seed 与客户端秩一致：

```bash
sbatch --array=3,6 run.sbatch
```

6. 全部完成后汇总：

```bash
/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python summarize.py
```

结果在 `runs/headroom_coldstart_20261017_v1/`：每格的 `result.json`（最终指标、逐客户端结果、曲线、成本）、`final_state.pt`，以及总的 `SUMMARY.md`。

## 已知局限

- 所有方法共用 lr 2e-4，未逐方法调参；rank2 与 rank8 的最佳学习率可能不同。若秩容量差接近阈值，应补做小范围 lr 检查后再下结论。
- 150 轮可能未收敛；曲线会显示差距是在扩大还是收窄。
- 因子平均（分别平均 A 与 B）不等于乘积平均，这是常用 FedAvg-LoRA 的已知近似，对所有联邦方法一致。
- 资源角色与域耦合（R1：government/travel 为强端），按域结果需结合这一点解读。
- 单卡模拟，耗时不代表真实联邦网络时间；上传量按实际训练的参数数计算。
