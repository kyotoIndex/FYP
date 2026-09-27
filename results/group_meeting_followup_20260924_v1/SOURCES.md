# 两批实验的结果来源

本目录汇总两批已完成的实验。第一批检查基础 LoRA、不同聚合方式、客户端划分和前 30 轮训练；第二批延长到 100 轮，并检查集中式组件选择、SVD 重拆分、本地适配和第二个数据集。原始执行归档仍按运行时目录保存，本目录提供队友阅读和复算所需的轻量材料。

| 批次 | 原始证据包 | SHA256 |
|---|---|
| 聚合与基础 LoRA 诊断 | `PCLA_Aggregation_Evidence_20260923_v1.zip` | `fe3df5bd7c74ddb9c6207f7407be0f7d7b8f3b91628f081b8e943cee767343df` |
| 组会后续实验 | `PCLA_V5_Evidence_20260924_v1.zip` | `f820f36e8ff360e9dac440f2fe658e8890108162f951c20f95dd27a519347d78` |

原先交接包中的端点和比较 CSV 保留原字节；从执行归档补入的曲线、成本、补充批次和源码另有发布版。发布版只将个人设备及 HPC 的绝对根路径替换为项目标识，保留原归档 SHA-256、归档内成员路径与原始执行文件 SHA-256。原始字节留在本地执行记录中，发布版及原件的对应关系列在各自的审计文件。逐文件来源和发布版校验值见 [data_sources.json](reproduce/data_sources.json)。

| 内容 | 公开材料 | 来源核验 |
|---|---|---|
| 聚合与基础 LoRA 诊断 | [逐 seed 端点](data/v41_per_seed_endpoints.csv)、[全部聚合差值](data/aggregation_all_five.json)、[固定状态指标](data/fixed_state_metrics.csv) | 上表第一批原始证据包；逐文件 SHA-256 在数据清单中。 |
| IID 集中式与冻结 A 的联邦 LoRA | [10 条逐 seed 端点](data/s1/S1_PER_SEED_ENDPOINTS.csv)、[300 轮读数](data/s1/S1_ROUND_TRAJECTORIES.csv) | [10 个原归档复核](data/s1/ARCHIVE_HASH_CHECK.json)及[逐值核验](data/s1/S1_VERIFICATION.json)。 |
| 两批完整可得逐轮曲线 | [11,600 行 CSV](data/curves/round_curves_public.csv)、[完整图](data/curves/complete_round_curves.pdf) | [归档与成员哈希](data/curves/README.md)、[覆盖与端点复算](data/curves/audit.json)。 |
| 执行开销 | [115 条逐作业记录](data/costs/PER_JOB_COSTS.csv)、[五 seed 汇总](data/costs/COST_SUMMARY_BY_CONDITION.csv) | [字段来源与缺失原因](data/costs/FIELD_COVERAGE.csv)、[核验](data/costs/COST_VERIFICATION.json)。 |
| 实际执行程序、配置与队列 | [源码索引](reproduce/executed_sources/SOURCE_README.md)、[逐文件原 SHA 与公开副本 SHA](reproduce/executed_sources/PUBLIC_SOURCE_MANIFEST.json) | [脱敏副本离线验证](reproduce/executed_sources/verify_public.py)；第三方上游代码按固定 commit 引用。 |

前期八方法比较与评分诊断已在仓库的 `results/core_shared_hh_20260918_v3/` 和 `results/rank_signal_20260918_v1/` 保存。当前主线的个性化指客户端特定的组件分配。

组会后续实验的表格覆盖主队列、匹配随机编号对照、参数指数平均与 SVD 的补充读取。补充读取用于取得已有轨迹上的读数，不增加独立随机种子。完整模型状态和原训练归档继续由实验档案保存，仓库只保留复算需要的轻量材料。

实际方法差异见 [METHODS.md](METHODS.md)。
