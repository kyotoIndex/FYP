# S1 补充批原始逐 seed 记录（2026-09-24 导出）

`S1_PER_SEED_ENDPOINTS.csv` 保留 `central_iid_both`（全体 12800 条训练样本中每轮 IID 抽样的集中式 LoRA）与 `fed_ffa_b`（A 固定、只训练 B 的联邦机制对照）各五个 seed：7、13、21、42、87。每条正式作业 30 轮；主端点是第 26–30 轮 872 条 SST-2 benchmark dev 的平均 accuracy，CSV 用 0–1 比例。`S1_ROUND_TRAJECTORIES.csv` 保留 10×30=300 行逐轮 accuracy、balanced accuracy、NLL、预测类别计数和单类预测标记。两个表都只来自 S1 正式队列的审计原始行，未纳入三条五轮 smoke。

| 条件 | seed 7 | seed 13 | seed 21 | seed 42 | seed 87 | 五 seed 平均 |
|---|---:|---:|---:|---:|---:|---:|
| IID 集中式 `central_iid_both` | 88.05% | 50.18% | 88.10% | 89.13% | 88.49% | 80.79% |
| FFA 式 `fed_ffa_b` | 79.75% | 82.89% | 81.93% | 75.94% | 78.51% | 79.80% |

IID 集中式 seed 13 曾达到 90.37%，但从第 21 轮起至第 30 轮，**每轮 872 条评价样本全部预测为同一个类别**，且预测类别在 0/1 之间切换。第 26–30 轮主端点为 50.1835%，仍完整计入五 seed 均值；现有记录没有裁决塌缩机制。具体第 21–30 轮的类别计数和 accuracy 见 `S1_VERIFICATION.json` 与逐轮 CSV。

每条端点行给出评估身份、样本划分、标签顺序、最终 checkpoint 的原审计 SHA-256，并给出审计 JSON、队列 receipt、原始归档的 SHA-256。`source_audit`、`source_receipt`、`source_archive` 在发布版中使用归档 ID 和相对路径；完整主机路径仅保留在本地 staging。`ARCHIVE_HASH_CHECK.json` 逐件重新读取远端 10 个原始 ZIP 并核对与 receipt 中的 archive SHA-256 一致。`S1_VERIFICATION.json` 确认所有状态为 completed、receipt 哈希与状态一致、每 job 正好 30 轮，并把 10 个逐 seed 端点与既有 `analysis_extra_main_s1/trajectory_readouts.csv` **逐值精确比较**；两组五 seed 均值与旧 `RESULT_TABLES.md` 的 80.79% / 79.80% 舍入一致。

实验配方和旧结论仍以 V4.1 的 `EXECUTION.md`、`RESULT_TABLES.md` 为准。S1 的 FFA 是机制式对照，并非 FFA-LoRA 论文的完整调参复现。
