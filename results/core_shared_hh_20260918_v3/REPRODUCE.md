# 配置、来源与复现

## 实际执行版本

训练时的仓库基准是 `a88f65d437fbb893690aadcade194b1f221d46b9`，本次新增实现当时尚未提交。执行时逐文件 SHA256（文件内容校验值）保存在 [execution_source_identity.json](reproduce/execution_source_identity.json)，源码新增内容见 [execution_source.patch](reproduce/execution_source.patch)。本次整理后的代码提交记录在 [publication_identity.json](reproduce/publication_identity.json)。执行身份仍以历史配置及文件校验值为准。

实际配置是 [core_shared_hh_20260918_v3.json](../../configs/pcla_repair/core_shared_hh_20260918_v3.json)，读取原有基础配置并覆盖本批训练设置。核心运行代码为 [runtime.py](../../pcla_repair/runtime.py)，层与组件分配为 [core.py](../../pcla_repair/core.py)。八种方法的实际命令见 [executed_jobs.jsonl](reproduce/executed_jobs.jsonl)。

模型为 `google-bert/bert-base-uncased`，版本 `86b5e0934494bd15c9632b12f734a8a67f723594`。五个模型与分词器文件的校验值见 [model_assets.json](reproduce/model_assets.json)。执行环境为 Python 3.11、PyTorch 2.5.1+cu121、Transformers 4.57.1、NumPy 2.1.2，使用 RTX 4090。汇总使用 SciPy 1.17.1。

SST-2 原始文件按原路径 `data/SST-2/train.tsv`、`data/SST-2/dev.tsv` 准备，校验值记录在执行身份中。[五份数据划分记录](reproduce/partitions/)保存源文件行号、各客户端样本和资源档位。`train:0` 表示训练文件去掉表头后的第 0 条有效记录。每个种子的各方法共用同一划分。

## 评分和更新

LoRA 的调整写为 `B × A`。一个 rank 分量由 A 的一行和 B 的一列构成，同编号的 query/value 分量共同作为一个组件。

更新 B 时，归一化评分为 `||梯度_B[:,j]||₂ / ||A[j,:]||₂`；更新 A 时交换两者。分母为零时该投影分数记零。Q、V 两个投影分别计算后相加。旧评分（legacy）使用乘法，未用于本批核心选择。

PCLA 将分数归一化并保留历史移动平均，再加入根据选择次数计算的探索项和距上次更新时间计算的新鲜度项，随后分配层与组件。参数见实际配置中的 `ema`、`beta`、`gamma`，对应 [UtilityState 和 allocate](../../pcla_repair/core.py)。评分来自当前客户端的训练样本，每次 16 条；客户端和 A/B 因子的评分状态分别保存。

## 复算已发布结果

在仓库根目录，使用已有 Python 环境中的 NumPy、SciPy：

```bash
python scripts/summarize_issue1_results.py --check
```

该命令检查原始逐轮准确率与已发布 CSV 一致，复算末五轮均值、七项配对差值与 95% t 区间，并核对全部客户端预算与随机编号对照的逐层数量。移除 `--check` 可重新生成结果及成本表。

重新生成图表：

```bash
python results/core_shared_hh_20260918_v3/reproduce/build_figures.py \
  --analysis results/core_shared_hh_20260918_v3/data \
  --out /tmp/pcla-core-figures
```

原有完整检查程序 [audit_original_results.py](reproduce/audit_original_results.py)还会读取预测和模型状态。运行它需要取得受控保存的完整原件；本目录保留的是已完成检查的[结果](data/verification.json)。检查覆盖全部 1,200 轮预测指标、160 份分轮模型状态和 40 份最终模型状态。NLL 是执行时记录的交叉熵损失；准确率、各类召回率和平衡准确率曾从预测独立复算。

[冻结参数检查](reproduce/frozen_parameter_verification.json)确认未选参数与基础模型保持不变；[暂停恢复检查](reproduce/integration_verification.json)确认恢复与连续运行的参数、选择记录和预测一致。本次整理复用这些已完成检查。

## 重新运行实验

基础模型和分词器须预先放在仓库根目录 `bert-base-uncased/`，并与模型清单的版本及文件校验值一致。沿用已安装的 CUDA 版 PyTorch；额外统计和测试依赖见 [requirements_pcla_repair_extra.txt](../../requirements_pcla_repair_extra.txt)。

以下为单个种子的示例。使用新的输出目录，完成 PCLA 后再运行依赖其选择记录的对照：

```bash
python scripts/run_pcla_repaired.py \
  --config configs/pcla_repair/core_shared_hh_20260918_v3.json \
  --seed 7 --policy pcla --device cuda \
  --out-dir results/core_shared_hh_reproduction/seed_7/pcla

python scripts/run_pcla_repaired.py \
  --config configs/pcla_repair/core_shared_hh_20260918_v3.json \
  --seed 7 --policy replay_random_rank --device cuda \
  --donor results/core_shared_hh_reproduction/seed_7/pcla/checkpoint.pt \
  --out-dir results/core_shared_hh_reproduction/seed_7/replay_random_rank
```

其余方法替换 `--policy`，全部种子与方法见命令清单。随机编号对照只读取 PCLA 的选择记录，自身参数仍从相同预训练初始化开始。

## 原件与文件说明

[原件清单及校验值](reproduce/original_file_hashes.json)对应直接保留的执行产物。[完整归档信息](reproduce/archive_reference.json)包含 200 份模型状态的归档哈希；需要模型状态或完整预测记录的同学可联系 PR 作者获取项目内共享副本。原始材料保留在实验档案中，仓库保存便于阅读和复算的文件。

`runs/` 保存 40 次训练的原始 `metrics.json`、`identity.json`、`status.json`。`data/client_updates.csv` 是逐客户端原始记录的紧凑表示，位置编码见[成本说明](COSTS.md)。它可以重新核对更新预算、费用和固定层对照；模型状态的实际张量一致性采用已有完整检查记录。

[返回实验说明](README.md)
