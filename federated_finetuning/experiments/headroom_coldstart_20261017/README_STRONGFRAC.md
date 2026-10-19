# 强端比例扫描 `headroom_strongfrac_20261017_v1`

2026-10-17 创建。状态：**代码完成、CPU 测试通过（8 项），尚未提交 HPC。** 代码与第一轮冷启动实验共用，见 [README.md](README.md)。

## 为什么做

第一轮（`headroom_coldstart_20261017_v1`，2 个 seed）的结论：

- 秩容量是真实的：全员 rank8 比全员 rank2 高约 1.5–1.75pp，达到 80% 准确率少用约 60% 的轮数。
- 但在 **40% 强端**时，朴素异构（强 r8 / 弱 r2）已经追回了大部分差距，弱端只比全员 rank8 少约 1pp，没有通过"弱端吃亏 ≥1pp"的判定。

猜测：强端越少，弱端越接近全员 rank2，留给自适应方法的空间就越大。本扫描用来验证这一点，并找出后续设计方法用的实验设置。

## 设置

除强端比例外，所有设置与第一轮相同（MNLI、50 客户端、冷启动、150 轮、每轮 10 端 × 20 步、lr 2e-4、seed2026 的数据划分）。

| 强端比例 | 强端客户端 | 弱端数 |
|---|---|---:|
| 10% | 10、11、12、40、41 | 45 |
| 20% | 10–14、40–44 | 40 |
| 40% | 10–19、40–49（即原 manifest 的 R1 角色） | 30 |

- 强端只从原来的强端（government 10–19、travel 40–49）里取，按两个域交替排列后取前 k 个。所以 10% ⊂ 20% ⊂ 40%，资源和域的耦合方式与第一轮一致，强端集合不随训练 seed 变化。
- 15 格 = 3 个 seed（2040、2041、2042）×（`fedavg_r8`、`fedavg_r2`、`hetero` 在 10% / 20% / 40%）。
- `fedavg_r8`、`fedavg_r2` 与强端比例无关，每个 seed 只跑一次。汇总时，把它们的逐客户端曲线按各比例下的弱端集合重新切分。这次的评测历史会保存每个客户端的结果，第一轮没有，所以 rank8 和 rank2 需要重跑。
- 40% 那格与第一轮的 hetero 设置相同，seed 2040 和 2041 可以顺便检查结果能否复现。GPU 本身存在非确定性，复现结果不会逐位相同。

## 预先登记（结果出来之前写定）

- **主指标：** **固定弱端集合（core weak）** 的客户端宏平均准确率，取 **第 110–150 轮的 5 个评测点平均**。固定弱端集合是在所有比例下都从未当过强端的 30 个客户端（fiction / slate / telephone）。
  - 为什么用固定集合：强端比例变小时，被降级的 government / travel 客户端和剩下的强端属于同一个域，它们会直接受益于强端训练的尾部分量。如果把它们混进"弱端平均"，不同比例之间就不再是同一批客户端，变量没有控制住。改用固定集合后，三个比例比较的是同一批客户端。
  - 为什么用最后 5 个点平均：第一轮显示单个评测点会波动约 1pp。最后一轮作为次指标一并报告。
- **判定：** 某个比例下，若 `fedavg_r8 − hetero` 在固定弱端集合上的差距（主指标）**在 3 个 seed 里都 ≥ 1.0pp**，就判定该比例"有提升空间"。
- **同时报告（次要）：**
  - 全体客户端的差距；
  - 该比例下全部弱端的差距；
  - 被降级的 government / travel 客户端单独的差距；
  - hetero 追回了 r8 与 r2 之间差距的几成（按固定弱端集合计算）；
  - 达到 78% 和 80% 准确率各需多少轮。
- **参与诊断：** 第 2–8 个 LoRA 分量只在本轮被抽中的强端之间平均。汇总时会按实际的参与日程，统计每轮抽到 0 个、1 个强端的轮次占比。理论值如下：

  | 强端比例 | 抽到 0 个强端 | 抽到 1 个强端 |
  |---|---:|---:|
  | 10% | 31% | 43% |
  | 20% | 8% | 27% |
  | 40% | 0.3% | 3% |

  也就是说，强端很少时，尾部分量要么整轮不更新，要么只由一个强端决定。这是朴素异构做法本身的特性，不是代码错误，记录下来用于解释结果，也可能成为后续方法的改进点。
- 3 个 seed 的结果只做描述，不做显著性检验。

## 成本

第一轮实测每格约 12 分钟（RTX 4090D）。15 格约 **3 GPU 小时**；`%3` 并发时墙钟约 1 小时，不含排队。每格申请 1 小时上限。

## HPC 提交步骤

1. 同步整个实验目录。这次 `headroom/`、`train.py` 都改过，需要整目录覆盖：

```bash
rsync -av --exclude '__pycache__' --exclude 'logs' /Users/liming/Documents/FYP/federated_finetuning/experiments/headroom_coldstart_20261017/ <hpc-login>:/home/student/t330025090/FYP/federated_finetuning/experiments/headroom_coldstart_20261017/
```

2. 提交：

```bash
cd /home/student/t330025090/FYP/federated_finetuning/experiments/headroom_coldstart_20261017 && mkdir -p logs && sbatch run_strongfrac.sbatch
```

3. 有任务失败时，只重新提交失败的序号，会从 checkpoint 续跑：

```bash
sbatch --array=7,12 run_strongfrac.sbatch
```

4. 全部完成后汇总：

```bash
/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python summarize_strongfrac.py
```

结果目录是 `runs/headroom_strongfrac_20261017_v1/`，里面有每格的 `result.json` 以及 `SUMMARY_STRONGFRAC.md` / `summary_strongfrac.json`。下载**整个目录**回本地即可，不需要 `final_state.pt` 时可以排除它。

## 任务序号

| 序号 | seed 2040 | 序号 | seed 2041 | 序号 | seed 2042 |
|---:|---|---:|---|---:|---|
| 0 | fedavg_r8 | 5 | fedavg_r8 | 10 | fedavg_r8 |
| 1 | fedavg_r2 | 6 | fedavg_r2 | 11 | fedavg_r2 |
| 2 | hetero 10% | 7 | hetero 10% | 12 | hetero 10% |
| 3 | hetero 20% | 8 | hetero 20% | 13 | hetero 20% |
| 4 | hetero 40% | 9 | hetero 40% | 14 | hetero 40% |

## 代码改动（与第一轮的差异）

- `headroom/data.py`：新增 `nested_strong_clients` / `override_roles`，按比例生成嵌套的强端集合。
- `headroom/fed.py`：`run()` 新增 `strong_fraction` 参数；每个评测点保存逐客户端的准确率和 NLL；identity 记录强端比例与强端名单。
- `train.py`：plan 可以显式列出 `cells`（方法、seed、强端比例），任务序号就是列表位置；不写 `cells` 时仍按原来的方法 × seed 网格，第一轮的 `plan.json` 不受影响（已用第一轮真实结果复核汇总不变）。输出目录带 `_sfXX` 后缀。
- 新增 `plan_strongfrac.json`、`run_strongfrac.sbatch`、`summarize_strongfrac.py`（主指标为固定弱端集合，另报被降级客户端与强端参与统计）；测试新增强端嵌套与扫描全流程 2 项。

## 已知局限

- 强端只来自 government 和 travel 两个域，比例与"哪些域是强端"仍然耦合。若需要"资源与域独立"的版本，要另做按域分层抽取强端的设置。
- 其余局限（lr 未逐方法调参、150 轮、因子平均）同第一轮。
