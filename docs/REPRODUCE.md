# AmpLoRA 复现指南

本仓库收录最终的 RoBERTa 与 Qwen 实验、逐轮结果，以及重新生成报告的代码。实验目录保留原名，便于逐一对应代码、配置和已完成的运行。

## 1. 运行环境与已有结果复算

原实验使用 Python 3.12、PyTorch **2.8.0+cu128**、Transformers **4.57.1**、NumPy **2.2.6**，以及 24 GB 显存的 NVIDIA RTX 4090 D。训练在 Linux 上使用 CUDA；数据准备与报告生成使用 CPU。请按显卡驱动安装相应的 CUDA 版 PyTorch。

在仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[test,figures]'
```

阅读和复算仓库已有的结果只需安装环境。下面的命令直接读取已收录的逐轮记录，无需下载模型和数据：

```bash
python scripts/reproduce_results.py
```

输出位于 `generated/results/`：包括两批 RoBERTa 报告、Qwen 报告，以及汇总各种子数值和样本标准差的 `MAIN_RESULTS.md`。原始实验记录保留在发布目录中，方便对照。

## 2. 准备重新训练所需的模型与数据

重新训练时使用独立的工作目录。脚本以 `--project-root` 为基准查找数据、模型和缓存，并保存新结果；代码仍从当前仓库运行。这样可以直接对照已发布的实验记录。

```bash
export AMP_PROJECT_ROOT="$PWD/.reproduction"
python scripts/prepare_assets.py all --project-root "$AMP_PROJECT_ROOT"
```

这条命令下载两个固定版本的模型和官方 MultiNLI 1.0，构建固定的客户端划分，再生成分词缓存。该阶段只准备资产。使用 `--family roberta` 或 `--family qwen` 可只准备一个模型及其缓存；两个模型共用同一份客户端划分。

| 资产 | 固定来源或设置 |
|---|---|
| RoBERTa-base | `FacebookAI/roberta-base`, revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b` |
| Qwen2.5-0.5B Base | `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| MultiNLI | [官方 MultiNLI 1.0 项目](https://cims.nyu.edu/~sbowman/multinli/)，原始 JSONL 文件 |
| 客户端划分 | 50 个纯语域客户端，数据种子 2026，R1，初始强端比例 0.4，每端至少 8 条样本 |
| 验证集 | 按规范化前提句分组留出约 5%；官方开发集单独保留 |
| 跨语域前提句 | 划分前明确排除，并记入生成的 manifest |
| 预期样本数 | 训练集 372,557 条，验证集 19,681 条 |
| 主实验资源角色 | 5 个强端：10、11、12、40、41；训练时将初始 R1 角色改为 10% 强端 |
| RoBERTa 分词 | 前提句与假设句成对输入，优先截断较长序列，最长 256 tokens |
| Qwen 任务 | 预测下一个词 ` yes`、` maybe`、` no`，提示词最长 256 tokens |

语料压缩包约 227 MB。准备脚本提取所需的三个 JSONL 文件后会清理临时 ZIP。模型、原始数据、分词缓存和新训练的检查点均保存在本地。

如果已有官方语料目录，可分阶段准备：

```bash
python scripts/prepare_assets.py models --project-root "$AMP_PROJECT_ROOT"
python scripts/prepare_assets.py data --project-root "$AMP_PROJECT_ROOT" \
  --mnli-source /absolute/path/to/multinli_1.0
python scripts/prepare_assets.py cache --project-root "$AMP_PROJECT_ROOT"
```

完全离线时，先将固定版本的模型目录复制到 `$AMP_PROJECT_ROOT/models/roberta-base` 和 `models/qwen2.5-0.5b`，包含 tokenizer 文件和 `REVISION.txt`，再运行数据划分与缓存阶段。为兼容原配置，缓存沿用历史目录名；其中 `client_influence_20261015_sgd_v1/shared_token_cache` 只是共用的 RoBERTa 分词缓存，可由当前准备脚本直接生成。

## 3. 按实验组重新训练

`scripts/run_experiments.py` 默认列出完整命令，加入 `--execute` 后按顺序**逐个执行**。已完成的运行会跳过，中断的运行从检查点继续。准备与训练默认使用已被 Git 忽略的 `.reproduction/` 目录。同一项目目录一次启动一个调度脚本即可。

```bash
python scripts/run_experiments.py roberta-main --project-root "$AMP_PROJECT_ROOT"
python scripts/run_experiments.py roberta-main --project-root "$AMP_PROJECT_ROOT" --execute
```

用 `--seeds 2040` 选择训练种子；重复使用 `--name` 可从列表中选择指定运行。所有主比较使用训练种子 **2040、2041、2042**，共用数据种子 **2026**。客户端批次、每轮参与者和组件分配均由这些种子确定。

| 实验组 | 运行内容与用途 |
|---|---|
| `roberta-main` | 12 次：r8、vref、L2、amp，各 3 个种子 |
| `roberta-ablation` | 13 次：拥有者平均、前缀分配、α=2/8、r=1、混合秩、20% 强端，以及 r=1 的 vref |
| `roberta-controls` | 21 次：FSLoRA 梯度路径、状态与聚合的分解，学习率补偿，以及 Fed-PLoRA/FSLoRA/FLASC 风格比较 |
| `roberta-baselines` | 原 headroom 扫描的 15 次运行：r8/r2，以及 10%/20%/40% 强端下的前缀异构；已完成的 r8 会跳过 |
| `roberta-diagnostics` | RCS、弱端小学习率、固定收缩，以及 D2/D3 固定快照诊断；在 `roberta-main` 后运行 |
| `qwen-lr` | 4 次 10 轮开发运行，种子 2041，用于选择学习率 |
| `qwen-main` | 种子 2040 的 9 次运行：r8/vref/L2/amp/FSLoRA 对照，以及 α=4/16/32 |
| `qwen-seeds` | 种子 2041/2042 的 8 次 r8/vref/amp/FSLoRA 补充运行 |
| `qwen-external` | 种子 2040 的 2 次 FSLoRA 风格/Fed-PLoRA 风格比较 |
| `qwen-aggregation` | 12 次：零填充收缩、服务器步长 0.25，以及其余 FSLoRA 风格种子 |
| `qwen-replay` | 原实验中用于核对不同源码版本的 3 次完整 60 轮重放 |
| `reports` | 从当前结果文件重新生成 RoBERTa 第一批、第二批和 Qwen 报告 |

RoBERTa 主实验使用 R=8、r=2，训练 150 轮，每轮 10 个客户端；每端做 20 步 AdamW，批大小 8，学习率 2e-4，缩放系数 2。Qwen 使用 R=64、r=8，训练 60 轮，其余本地优化器设置相同。两者均将梯度裁剪阈值设为 1.0，权重衰减设为 0。Qwen 主干使用 bfloat16、LoRA 因子使用 FP32；RoBERTa 使用 FP32。

要生成完整的 RoBERTa 结果表，依次运行 `roberta-main`、`roberta-ablation`、`roberta-controls`、`roberta-baselines`，最后运行 `reports`。D2/D3 机制图还需要 `roberta-diagnostics`，其报告命令见下文。

Qwen 原实验顺序为 `qwen-lr` → `qwen-main` → `qwen-seeds` → `qwen-external` → `qwen-aggregation` → `qwen-replay` → `reports`。仓库配置已填入选出的学习率 **2e-4**。事先写定的选择规则是：比较第 10 轮核心弱端准确率，在与最高值相差不超过 0.5 个百分点的候选中取最大学习率。完整记录包含 **38** 次 Qwen 运行，其中 3 次为源码版本重放。

在集群上，请先通过调度器申请 GPU，再在分配到的计算资源中执行，例如：

```bash
python scripts/run_experiments.py qwen-aggregation \
  --project-root "$AMP_PROJECT_ROOT" --execute
```

历史 `.sbatch` 文件记录了原集群的资源设置。可按自己的集群修改分区、GPU 和账号字段，也可在 GPU 作业内直接使用上述便携入口。

## 4. 代码入口与结果阅读

| 组件 | 源码或输出 |
|---|---|
| 分配、本地 AdamW 与聚合 | `federated_finetuning/experiments/scra_diagnostics_20261017/scra/core.py` |
| 放大前向与状态草图 | `scra/model.py` |
| RoBERTa 训练 | `run.py d0 --plan <plan> --project-root <root> --device cuda --resume` |
| r8/r2/前缀基线 | `experiments/headroom_coldstart_20261017/train.py` |
| Qwen 模型、提示词与缓存 | `scra/llm.py` |
| Qwen 联邦训练与报告 | `run.py c2fed` / `run.py c2fedreport` |
| RoBERTa 原始曲线 | `runs/scra_batch_20261018/*/d0/d0_rounds.json`、第二批同类文件，以及 diagnostics/headroom 下的参照曲线 |
| Qwen 原始曲线 | `runs/c2_fed_20261018/*/fed.json` |
| 配置与软件版本 | 各结果旁的 `identity.json` |
| 方法名称 | `amp`：放大 + Hájek；`amp_zf`：放大 + 零填充收缩；`vref`/`vref_zf`：完整本地训练、采用相同的部分上传 |

表中第一行以后的路径，相对于对应的实验目录或 `federated_finetuning`。Qwen 中沿用历史名称的 `r8` 实际表示同构 **rank 64**，这一名称继承自编码器实验。

主指标是 fiction、slate、telephone 三个语域中 30 个弱客户端的宏平均准确率，即先分别计算每端准确率，再对客户端等权平均。RoBERTa 取第 110、120、130、140、150 轮的均值；Qwen 取第 40、45、50、55、60 轮的均值。标准差与配对区间均以 3 个训练种子为单位计算。Qwen 每端最多抽取 40 条验证样本，所有运行共用种子 2040 确定的评测子集。

主结果参考值，单位为百分比：

| 模型 | r8 | 完整训练、部分上传 | AmpLoRA | 不放大的受限训练 L2 |
|---|---:|---:|---:|---:|
| RoBERTa | 80.19 | 80.17 | 80.12 | 77.85 |
| Qwen | 82.19 | 80.48 | 80.41 | 33.70（种子 2040） |

表中 RoBERTa 的 vref 与 AmpLoRA 使用 Hájek 聚合，Qwen 的两者使用零填充收缩；r8 使用完整上传的普通平均，L2 使用拥有者平均。Qwen 的 Hájek 变体保留为聚合对照。FSLoRA/Fed-PLoRA/FLASC 标签指在本研究统一 AdamW 协议下的相应风格实现，报告逐项列出了其分配、本地训练规则与聚合方式。

完成 D2/D3 运行后，用以下命令重新生成报告：

```bash
python federated_finetuning/experiments/scra_diagnostics_20261017/run.py d2report \
  --plan federated_finetuning/experiments/scra_diagnostics_20261017/plan_d2.json \
  --project-root "$AMP_PROJECT_ROOT" --device cpu
python federated_finetuning/experiments/scra_diagnostics_20261017/run.py d3report \
  --plan federated_finetuning/experiments/scra_diagnostics_20261017/plan_d3.json \
  --project-root "$AMP_PROJECT_ROOT" --device cpu
```

当前训练代码在完整 rank-R 参数结构上使用梯度掩码。资源分析分别说明了上传因子数量、可训练状态的估算，以及端到端时间和显存的实测结果，详见[资源分析](designs/resource_case_2026-10-18.md)与[最终 Qwen 设计及结果](designs/c2_1_design_2026-10-18.md)。

## 5. 图表生成与代码检查

```bash
pytest -q
python federated_finetuning/experiments/scra_diagnostics_20261017/figures/make_figures.py
python docs/thesis/figures/method_v2/build_figures.py --output-dir generated/method-figures
```

五张 RoBERTa 图读取仓库收录的曲线，输出到绘图脚本所在目录；数值核对记录在 `figures/checks.md`。方法图使用 Python 标准库重新生成原生 SVG 和可编辑 Draw.io 文件。导出 PDF/PNG 需要安装 Draw.io CLI，详见[图表编辑说明](thesis/figures/method_v2/README.md)。
