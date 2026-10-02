# 数据来源

`data/` 下的每个文件都直接复制自实际运行的导出，没有手工改动数字。[PUBLIC_MANIFEST.json](reproduce/PUBLIC_MANIFEST.json) 逐个记录每个文件的：

- 来源；
- 原文件 SHA-256 和发布后的 SHA-256；
- 是否压缩；
- 是否替换了路径。

另外两类文件由这些数据生成：

- **按方法名称整理的表和浏览页（`reader/`）：** 输入和输出哈希记在 [BUILD_MANIFEST.json](reader/BUILD_MANIFEST.json)。
- **图（`figures/`）：** 数值核对记录在 [figure_numeric_audit.json](figures/figure_numeric_audit.json)。

## 发布时做的两种处理

- **路径替换：** 本机和服务器的绝对路径换成 `$FYP_ROOT`、`$LOCAL_HOME`、`$HPC_HOME`，服务器账号换成 `<hpc_account>`，其余内容不变。替换过的文件，原文件哈希和发布哈希都在清单里。
- **压缩：** 超过 1 MB 的数据文件用 gzip 压缩，文件名加 `.gz`。用 `gzip -dc` 或 Python 的 `gzip.open` 可以直接读取。

源码清单另有一项公开处理：运行时 `SOURCE_MANIFEST.json` 的逐文件哈希按实际公开源码更新，原清单以 `ORIGINAL_SOURCE_MANIFEST.json` 原样保留。公开总清单中的 `source_sha256` 仍是原件身份，`content_sha256` 是更新后的公开文件身份；这类条目标记 `redacted: true` 并注明 `transformation`。实验配置、结果、原清单和历史父版本的哈希不变。

## 各目录对应的运行

| 目录 | 内容 |
|---|---|
| `data/1_budget_selection/development/` | 117 种配方 × 3 个种子的结果、逐轮记录、通信分项、计时、配方参数，以及可离线打开的曲线图 |
| `data/1_budget_selection/validation/` | 16 种方法 × 12 个种子的结果、逐轮曲线（原参数与参数滑动平均）、全部配对差、运行来源、失败尝试记录 |
| `data/1_budget_selection/costs/` | 117 条计时运行的结果、尝试记录与回执、质量与成本并列表、历史执行索引 |
| `data/1_budget_selection/diagnostics/` | `selection_bank/`：96 个挑法诊断；`coordinates/`：48 个拆法诊断；`raw/`：两者的全部原始读数 |
| `data/2_server_feedback/development/` | 32 种配方 × 4 个种子的结果、逐轮曲线、必要时间和完整原始记录 |
| `data/2_server_feedback/validation/` | 9 种方法 × 12 个种子的结果、逐轮曲线、必要时间和完整原始记录；`summary/`：全部配对差与时间比；`controller/`：控制器每轮的选择与质量读数 |
| `data/2_server_feedback/diagnostics/` | 120 个诊断的各规则读数、客户端读数、LoRA 与分类头可加性，以及按种子和按规则的汇总 |
| `data/2_server_feedback/costs/` | 每次计时尝试、计时分类、诊断成本、运行退出记录 |
| `data/3_controller_switch_off/` | 24 条续训的逐种子配对值、按轮汇总、正式与技术成本 |

## 完整原始记录与实际配置

- **第二部分验证的完整原始记录：** 未压缩时 503 MB，[gzip 后约 23 MB](data/2_server_feedback/validation/accepted_originals.jsonl.gz)，原始 SHA-256 为 `254210610440b6b681a7e1d9a3ac0aea3ed385aee5f09030fd075c2240384322`。它包含 108 条运行的配置、结果、逐轮指标、审计和接受记录；开发阶段的原始记录在对应的 `development/` 目录。
- **实际执行配置：** [配置索引](reproduce/actual_configs/INDEX.csv) 共 428 条，覆盖第一部分 192 条验证、第二部分 128 条开发和 108 条验证。
  - 第一部分有 28 份依赖开发结果的配置，是从归档中找回、再逐一按配置哈希核对的。
  - 第二部分的配置直接从原始记录里提取。提取时调整了排版并替换了路径，所以索引同时保留执行时的配置哈希和公开文件的哈希。

## 没有放进来的内容

| 内容 | 哈希在哪里 | 原因 |
|---|---|---|
| 模型检查点、客户端更新张量、预测输出 | 各导出文件里的哈希字段 | 体积太大 |
| SST-2 原文和分词缓存 | `reproduce/data_manifest/*/train_tokens.json` | 用公开数据集和清单里的样本编号可以重建 |

这些文件都保存在服务器运行目录和本地归档里，需要时可以按哈希核对。
