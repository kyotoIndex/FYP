# 数据来源

原值归档由实际运行的导出文件复制而来。[reproduce/PUBLIC_MANIFEST.json](reproduce/PUBLIC_MANIFEST.json) 逐个文件记录来源、原文件SHA-256、发布后内容SHA-256、压缩与路径替换。新增的名称表、阅读页和图表由公开原值生成，输入与输出哈希分别记在 [阅读表清单](reader/BUILD_MANIFEST.json) 和 [图表数值核验](figures/figure_numeric_audit.json) 中。说明文档是本PR的交接文本，与原值文件分开管理。

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

## 实际执行配置与完整原值

第二组完整验证原值合集已以 [gzip文件](data/2_server_feedback/validation/accepted_originals.jsonl.gz) 纳入，原始未替换文本503MB，发布gzip约23MB。其原始SHA-256为 `254210610440b6b681a7e1d9a3ac0aea3ed385aee5f09030fd075c2240384322`，包含108条配置、结果、metrics、审计和接受记录。

[428条实际配置索引](reproduce/actual_configs/INDEX.csv) 覆盖第一组192条验证，以及第二组128条开发和108条验证。第一组28份条件配置从归档找回，逐一匹配原值记录的配置哈希；第二组执行配置从已接受原值的`config_original`提取。提取文件的排版与路径替换会改变字节哈希，因此索引同时保留执行时原配置哈希与容器哈希，公开清单记录提取后文件哈希。选择前计划与实际参数由索引区分。

## 大型输入与状态的取得

| 文件 | 大小 | SHA-256 | 原因 |
|---|---:|---|---|
| 模型检查点、客户端更新张量、预测 logits | — | 见各导出文件里的哈希字段 | 体积过大 |
| SST-2 原始文本和分词缓存 | — | 见 `reproduce/data_manifest/*/train_tokens.json` | 用公开数据集和清单里的样本编号可以重建 |

这些文件都保存在服务器的运行目录和本地归档中，需要时可以按哈希核对。
