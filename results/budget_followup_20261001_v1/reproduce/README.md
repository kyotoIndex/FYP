# 复现材料

## 复算与核对

在结果目录下运行：

```bash
python3 reproduce/verify_tables.py
python3 reproduce/verify_public.py
```

- `verify_tables.py` 从 `data/` 里的文件重新计算 README 中引用的全部数字，包括各方法均值、配对差值、落后的种子数、诊断读数、成本和返回轮次，并逐一和 README 中的取值比较。
- `verify_public.py` 按 `PUBLIC_MANIFEST.json` 核对每个发布文件的 SHA-256，并检查目录里没有清单之外的文件。

两个脚本只用 Python 标准库。

## 实际运行的源码

`executed_sources/` 按三部分保存每次冻结的源码快照，每个子目录是当时实际上传到服务器运行的完整代码。每个快照的 `SOURCE_MANIFEST.json` 记录逐文件哈希；第一部分里的 `*_vs_base.patch` 是相对仓库提交 `a3f72f4`（PR #3）的完整差分。

| 目录 | 用途 |
|---|---|
| `1_budget_selection/core01`、`core02` | 训练代码，开发阶段与固定方法的验证使用 |
| `1_budget_selection/native03`、`native04` | Fed-SB、FedSVD、Fed-PLoRA 的移植；两版差别见 `native04_changes_from_native03.json` |
| `1_budget_selection/validation01`、`validation02` | 固定方法验证的调度版本 |
| `1_budget_selection/conditional01`、`conditional02` | 依赖开发结果的验证方法，以及同名额随机编号对照 |
| `1_budget_selection/cost01`、`cost02` | 独立计时：`cost01` 是技术检查版本，正式计时用 `cost02` |
| `1_budget_selection/diagnostic_preflight01` | 诊断的 CPU 预检 |
| `1_budget_selection/diagnostic02`、`diagnostic03` | 诊断训练；`diagnostic03` 加入了配置生成 |
| `1_budget_selection/scheduler_20260928` | GPU 调度与审计控制，不含训练代码 |
| `1_budget_selection/analysis_scripts` | 导出、汇总和统计脚本 |
| `2_server_feedback/technical01`、`technical02` | 正式运行前的技术检查 |
| `2_server_feedback/core01`、`core02` | 训练与服务器控制代码 |
| `2_server_feedback/native01`–`native03` | Fed-SB 等原生方法；后两版加入状态审计 |
| `2_server_feedback/diagnostic_preflight01`、`diagnostic_preflight02` | 诊断 |
| `2_server_feedback/workspace` | 服务器运行目录在收尾时的快照 |
| `2_server_feedback/analysis_scripts` | 导出、汇总和统计脚本 |
| `3_controller_switch_off/policy01`、`policy02` | 续训代码；正式结果来自 `policy02`，它只修改了断点恢复和计费汇总，训练代码与 `policy01` 相同 |
| `3_controller_switch_off/analysis_scripts` | 配对分析 |

源码里的绝对路径已替换为 `$FYP_ROOT`、`$HPC_HOME` 等变量，原始字节的哈希保存在 `PUBLIC_MANIFEST.json`。

## 配置与数据清单

- `configs/`：每条运行的完整配置文件（开发、验证、计时、诊断、续训）。
- `data_manifest/1_budget_selection/`：第一部分的客户端划分与角色清单（`train_roles.json.gz`，只有样本编号）、分词设置、参与顺序和通信额度。
- `data_manifest/2_server_feedback/`：第二部分追加新种子后的角色清单，原有种子的记录与第一部分相同。
