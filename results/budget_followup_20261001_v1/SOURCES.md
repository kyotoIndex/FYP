# 数据来源

本目录的每个文件都由实际运行的导出文件复制而来，没有手工修改数字。[reproduce/PUBLIC_MANIFEST.json](reproduce/PUBLIC_MANIFEST.json) 逐个文件记录了来源路径、原文件 SHA-256、发布后内容的 SHA-256，以及是否压缩、是否替换了路径。

## 发布时做的两种处理

- **路径替换：** 文件中的本机和服务器绝对路径替换为 `$FYP_ROOT`、`$LOCAL_HOME`、`$HPC_HOME`，服务器账号替换为 `<hpc_account>`，其余内容不变。替换过的文件，原文件哈希和发布哈希都记在清单里。
- **压缩：** 超过 1 MB 的数据文件用 gzip 压缩，文件名加 `.gz`。可以直接用 `gzip -dc` 读取，或者在 Python 里用 `gzip.open`。

## 各目录对应的运行

| 本目录 | 来自 | 内容 |
|---|---|---|
| `data/1_budget_selection/development/` | 第一部分开发阶段导出 | 117 种配方 × 3 个种子的终点、逐轮记录、通信分项、计时、配方参数，以及可离线打开的曲线图 |
| `data/1_budget_selection/validation/` | 第一部分验证阶段导出与汇总 | 16 种方法 × 12 个种子的终点、逐轮曲线（原参数与滑动平均）、全部配对差值、运行来源、失败尝试记录 |
| `data/1_budget_selection/costs/` | 第一部分独立计时运行 | 117 条计时作业的终点、尝试记录与回执，以及质量与成本的并列表 |
| `data/1_budget_selection/diagnostics/` | 第一部分诊断导出与汇总 | `selection_bank/` 为 96 个挑法比较对象，`coordinates/` 为 48 个拆法比较对象，`raw/` 为两者的全部原值 |
| `data/2_server_feedback/development/` | 第二部分开发阶段导出 | 32 种配方 × 4 个种子的终点、逐轮曲线、必要时间 |
| `data/2_server_feedback/validation/` | 第二部分验证阶段导出与汇总 | 9 种方法 × 12 个种子的终点、逐轮曲线、必要时间；`summary/` 为全部配对差值与时间比；`controller/` 为每轮的控制器选择与质量读数 |
| `data/2_server_feedback/diagnostics/` | 第二部分诊断导出与汇总 | 120 个对象的规则读数、客户端读数、四角交互，以及按种子和按规则的汇总 |
| `data/2_server_feedback/costs/` | 第二部分成本导出 | 每次计时尝试、计时分类、诊断成本、运行退出记录 |
| `data/3_controller_switch_off/` | 第三部分分析与收尾 | 24 条续训的逐种子配对值、按轮汇总、正式与技术成本 |

## 没有放进来的文件

| 文件 | 大小 | SHA-256 | 原因 |
|---|---:|---|---|
| 第二部分验证阶段每条运行的原始记录合集（`accepted_originals.jsonl`） | 503 MB | `254210610440b6b681a7e1d9a3ac0aea3ed385aee5f09030fd075c2240384322` | 体积过大；本目录的终点、曲线和成本表都由它导出，导出清单中有对应哈希 |
| 模型检查点、客户端更新张量、预测 logits | — | 见各导出文件里的哈希字段 | 体积过大 |
| SST-2 原始文本和分词缓存 | — | 见 `reproduce/data_manifest/*/train_tokens.json` | 用公开数据集和清单里的样本编号可以重建 |

这些文件都保存在服务器的运行目录和本地归档中，需要时可以按哈希核对。
