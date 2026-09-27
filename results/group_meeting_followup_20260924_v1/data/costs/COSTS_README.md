# PCLA 发布成本附表（2026-09-24）

这里整理**已有正式运行**，没有新训练。`PER_JOB_COSTS.csv` 共 115 条逐作业记录：v3 PCLA / replay / full_alt、V4.1 `fed_both_mean`、V5a 两种 SVD、V5b `pcla_svd`、V5d `replay_svd`、V5 的 100 轮 PCLA / full_alt / fed_both_mean、AG_NEWS 的 PCLA / replay / full_alt / fed_both_mean，以及 V5c 开源实现 IID/非 IID 八格（A100）；每格五 seed。除 `v5_long_sst2` 为 100 轮外，均为 30 轮；每轮 10 客户端。`COST_SUMMARY_BY_CONDITION.csv` 只在**相同 cohort、硬件、条件**内取五 seed 均值；4090 和 A100 没有混合平均。V5c 虽同为 BERT/SST-2，但接入另一套开源训练/聚合代码，不能用与 4090 条件的耗时差解释算法差。

## 字段来源和口径

`FIELD_COVERAGE.csv` 逐条件、逐字段列出五条作业的非 NA 数量、原字段位置和缺失原因；例如 V5b `pcla_svd` 的 `svd_server_seconds` 是 0/5，原因是没有独立 SVD 计时，不能从整轮墙钟推算。

| CSV 字段 | v3 与部分 V5 旧式 `metrics.json` | V4.1 与部分 V5 逐轮 `record.json` |
|---|---|---|
| `client_train_seconds` | 各轮 `train_seconds` 之和；v3 与逐轮 `clients_round_*.json` 客户端和核对 | 逐轮 `client_details[].train_seconds` 之和 |
| `scoring_seconds` | `probe_seconds` 之和；0 是实际记录值 | NA：无独立计时字段 |
| `training_and_server_seconds` | NA：无此合计字段 | `training_and_server_seconds` 之和；包括训练、聚合等服务器处理，不能只称训练时间 |
| `round_wall_seconds` | `simulator_round_wall_seconds` 之和，包含本轮评价与其他开销 | NA：未记录整轮墙钟 |
| `job_elapsed_seconds` | v3 NA；V5d 有队列状态起止时间 | 队列状态 `ended-started`，含加载、归档等作业开销，不能与整轮墙钟同口径相减 |
| `evaluation_seconds` | `eval_seconds` 之和 | NA：无独立普通评价计时字段 |
| `svd_server_seconds` | NA：无独立 SVD 计时 | NA：V5a 的 SVD 包在 `training_and_server_seconds` 内，不能用总时间减客户端训练就命名为 SVD |
| `ema_evaluation_seconds` | NA：无单独 EMA 计时 | NA：V5e 补了 EMA 评价**结果**，没有单独 EMA 评价耗时 |
| `upload_bytes` / `download_bytes` | 逐客户端 `clients_round_*.json` 序列化 payload 字节求和 | 各轮 `record.json` 相应字段求和；V5c 上游开源实现未记录，故 NA |
| `serialized_traffic_bytes` | 各轮原字段之和，并与上传+下载核对 | NA：无此原字段；另给 `payload_total_bytes_computed`（上传+下载的算术和） |
| `peak_cuda_allocated_bytes` | 各轮 `cuda_peak_allocated` 最大值 | 各轮 `record.json` 同字段最大值 |

这些字节是**序列化 payload 的模拟计数**，不是网卡实测流量。v3 的源码对每个客户端重新序列化并计入一次全局 adapter 下发；V5 SVD 记录中的 17.7 GB/作业下载则包含每客户端重复 base 下发。重复 base 下发是这些实现的成本，**不应当写成 SVD 或 PCLA 理论上必需的通信成本**。`payload_total_bytes_computed` 只做记录字段相加，不另推算优化后的实现。

## 可比性与读数

同一 v3 实现和 4090 上，PCLA 五 seed 平均整轮墙钟 302.21 s，其中客户端训练 259.17 s、评分 19.59 s；同层同配额的随机 rank replay 为 263.57 s、客户端训练 245.26 s、记录评分 0 s；full_alt 为 253.22 s。v3 三组各自平均序列化 payload 为 825.84 MB、825.84 MB、1074.40 MB。这是该模拟器、30 轮作业的读数。V4.1 `fed_both_mean` 的“训练+服务器”平均 297.41 s；V5a `fed_svd_refactor` 为 345.21 s、`alt_svd_refactor` 为 307.88 s；两版计时边界不同，不能据此直接报跨版速度比。V5b `pcla_svd` 与 V5d `replay_svd` 沿用分项计时；V5 的 100 轮轨迹单列，不与 30 轮总成本直接比。全部逐 job 数值在 CSV 中，A100 另组可查。

## 复核和来源

`COST_VERIFICATION.json` 记录 115 作业、4500 轮、4090/A100 分组、缺失原因及抽取程序 SHA-256。v3 原件位于 `runs/20260918_core_shared_hh_v3/results/seed_*/{pcla,replay_random_rank,full_alt}/`；V4.1 和 V5 原件路径逐行在 CSV 的 `source_records` 中。`record_chain_sha256` 对逐轮文件按轮次把“相对路径、该文件 SHA-256”串接再哈希；单一 `metrics.json` 则直接存文件 SHA-256。`source_status_sha256` 对状态文件直接哈希。远端只读抽取及本地合并程序保留在 FYP staging；发布 CSV 使用 `fyp_pcla/` 与 `hpc:` 归档 ID 和归档内相对路径，不含个人主机目录。哈希指向未改动的原件。
