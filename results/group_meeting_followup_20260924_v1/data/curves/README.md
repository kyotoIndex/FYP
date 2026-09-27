# PCLA 两批完整逐轮曲线

**可入仓库的文件**：`round_curves_public.csv` 是主数据；`complete_round_curves.pdf` 是从这份 CSV 直接绘出的 24 页图，首页解释方法代码；`method_legend.csv`、`extrema.csv`、`coverage.csv`、`field_coverage.csv`、`archive_manifest_public.csv`、`member_sha256.csv`、`audit.json` 和 `figure_provenance.json` 分别给出方法解释、峰谷、覆盖、字段检查、归档与成员哈希、复算审计及图件来源。图件保留每一轮原始观测，没有平滑或截去后期低谷。

范围是聚合与基础 LoRA 诊断（V4.1 主批和 S1）以及组会后续实验（V5a/b/c/d）。五个 seed 均为 7、13、21、42、87。**225 条正式训练轨迹**覆盖 45 个方法，每个方法五个 seed：200 条 30 轮、25 条 100 轮，共 **8,500 条原参数记录**。另有 **2,600 条训练时记录的 EMA**，以及 **500 条 V5e 对原 L2 训练轨迹的确定性只读 EMA 重放**，总计 **11,600 行、270 条完整曲线**。V5e 行通过 `training_job` 关联原训练；它不是新增训练作业。

每行包含 accuracy、评价 loss、样本数及预测类别计数。SST-2 的 class 2/3 列留空，因为任务只有两类；AG_NEWS 四类计数齐全。预测类别没有出现时，其计数是从完整预测列表算出的真实零。没有把缺失值填零。`variant=raw`、`ema`、`ema_replay` 分别代表原参数、训练时 EMA 只读评价、V5e EMA 只读重放。类别计数按原归档保存值或逐轮预测计算；`count_source` 标识来源。V4 的 loss 对应原记录中的 GPU NLL，其他批次使用各自逐轮评价 loss。

`coverage.csv` 列明另 **40 个没有全局逐轮轨迹的作业**：V4 固定状态评估 10 个、V5 本地独立六 epoch 训练 5 个、V5 个性化终点评价 25 个。它们的逐轮 accuracy/loss/类别计数明确标为 unavailable。技术性 smoke/parity 作业 38 个和 V5e 只读因子诊断 5 个另列为排除项。V5e EMA 重放单独列为只读评价，未算作训练。V5 汇总端点表中还有 25 个从更早的 V3 或 V4 复用的 accuracy 窗口，不属于这两批的新训练轨迹；`audit.json` 逐项列出，**其余 240 个窗口**由本表完整复算吻合。

主数据的 `source_archive` 已从原始 HPC 绝对路径改为 `batch/archive.zip` 形式，`archive_manifest_public.csv` 同样只给归档 ID 和回执 ID，保留回执报告的归档 SHA-256；`member_sha256.csv` 是抽取时对实际使用的 ZIP 成员字节独立计算的 SHA-256。ZIP 读取同时检查成员 CRC。原始 `raw_extract.jsonl`、`round_curves.csv` 和 `archive_manifest.csv` 留在此 staging 目录，不属于入仓库材料。公开主 CSV 的 SHA-256 写在 `audit.json` 中，图件及输入哈希写在 `figure_provenance.json` 中。

抽取脚本是 `extract_rounds.py`（在远端只读运行），`build_curves.py` 生成公开数据与审计，`plot_curves.py` 仅从公开 CSV 生成图。V4 主批 1,950 行还与本地既有 `attempt2/analysis/full_curves.csv` 对照，轮次、accuracy、GPU loss 和类别计数逐项完全相同。
