# PCLA Experiments

面向联邦参数高效微调（Federated PEFT）的可复现实验项目。项目在异构客户端和非独立同分布数据下，统一实现并比较 LoRA 层/组件分配方法，重点记录 Fed-HeLLo 机制复现、经典 baseline、LEGEND 对照，以及组件级 PCLA 的正式实验。

> 研究边界：这里的 Fed-HeLLo 是受论文启发的受控机制复现，不声称与论文的完整数值设置或结果完全一致。PCLA 结果也只在对应配置、数据划分、随机种子和成本模型下成立。

## 先看什么

| 目的 | 入口 |
|---|---|
| 了解研究问题和方法 | [docs/PCLA.md](docs/PCLA.md) |
| 复现实验 | [docs/REPRODUCE.md](docs/REPRODUCE.md) |
| 查看结果状态和证据边界 | [results/README.md](results/README.md) 和 [docs/RESULTS.md](docs/RESULTS.md) |
| 查看完整目录 | [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md) |
| 生成教师汇报材料 | [reports/README.md](reports/README.md) |
| 与同学协作 | [CONTRIBUTING.md](CONTRIBUTING.md) |

## 当前项目包含什么

- fedlora：全层、统一 rank 的基线。
- fedhello：RGD warm start + 周期性 FIM 分配的机制复现。
- resource_only、fedra_random_partial、fim_hla、rgd_hla：资源、随机和重要性相关对照。
- legend：论文启发的 LCD / arithmetic-rank 对照实现。
- personalized_cost_aware_lora：早期层级 PCLA 原型，保留作历史对照。
- random_component_alternating 与 pcla_component_alternating：严格匹配组件预算的随机控制和组件级 PCLA 主线。

正式结果采用多随机种子、固定协议和版本化输出目录保存。开发运行、诊断运行和被拒绝的假设不与正式结果混在同一个目录中。

## 目录结构

~~~text
pcla_experiments/
├── configs/       实验协议；每个配置对应一个可追踪的运行设置
├── data/          已准备的数据集和数据清单
├── docs/          方法说明、复现笔记、结果解释和项目地图
├── models/        小型配置/tokenizer 文件；大模型权重不进 Git
├── reports/       从结果生成的教师汇报材料和图表
├── results/       版本化原始结果；正式、开发、诊断分区
├── scripts/       运行、验证、聚合、绘图和数据准备脚本
├── src/           训练与分配实现
├── tests/         分配约束和调度不变量测试
├── requirements.txt
└── README.md
~~~

目录内的详细说明见 docs/PROJECT_MAP.md。不直接移动已有结果目录，是为了保持配置、绘图脚本和论文材料中的路径有效，并保留实验 provenance。

## 环境要求

- Python 3.12（已验证环境为 3.12.6）。
- PyTorch、Transformers、NumPy、pandas、Matplotlib，版本范围见 requirements.txt。
- 运行配置使用 local_files_only=true；BERT-base 权重需要预先存在于本机 Hugging Face 缓存。
- models/roberta-base/ 只保留配置和 tokenizer 文件；约 499 MB 的 model.safetensors 被 .gitignore 排除，需在本机另外准备。

## 快速开始

PowerShell：

~~~powershell
cd F:\FYP\pcla_experiments
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
~~~

先运行不依赖训练权重的分配测试：

~~~powershell
python -m unittest tests\test_allocations.py
~~~

验证主实验配置和已保存结果：

~~~powershell
python scripts\validate_setup.py --config configs\expanded_sst2_bert_base_dirichlet.json
python scripts\validate_results.py --config configs\expanded_sst2_bert_base_dirichlet.json --results results\expanded_sst2_bert_base_dirichlet_v1
~~~

复现时不要覆盖已经审计的 v1 目录，使用新的版本号：

~~~powershell
python scripts\run_full_experiments.py --config configs\expanded_sst2_bert_base_dirichlet.json --out-dir results\expanded_sst2_bert_base_dirichlet_v2
~~~

组件级主线的配置、结果和解释见 docs/PCLA_LARGE_SCALE_CONSTRAINED_V1.md。完整的复现矩阵、数据准备和验证顺序见 docs/REPRODUCE.md。

## 结果状态

- results/expanded_sst2_bert_base_dirichlet_v1：当前经典 baseline 主表，3 个 seed，已保留完整原始输出和图表。
- results/pcla_large_scale_constrained_dirichlet_v1：组件级 PCLA 主线，包含匹配预算的随机组件控制。
- results/legend_large_scale_constrained_dirichlet_v2：确定性成本模型下的 LEGEND 对照；旧的失稳诊断运行仅作诊断证据。
- results/formal_v2：跨数据集/模型矩阵的部分运行；当前不应写成“矩阵已完成”，详情见 results/README.md。

报告时应使用多 seed 的均值和样本标准差，并同时说明数据异质性、通信/时间成本口径和方法覆盖范围。smoke、单 seed、开发和诊断运行不能替代正式证据。

## 生成表格和图表

~~~powershell
python scripts\make_formal_comparison_table.py --results results\expanded_sst2_bert_base_dirichlet_v1
python scripts\plot_expanded_results.py --results results\expanded_sst2_bert_base_dirichlet_v1
python scripts\plot_pcla_large_scale.py --results results\pcla_large_scale_constrained_dirichlet_v1
~~~

## 协作约定

- 每个新假设使用新的配置名和结果目录，不覆盖已验证的 v1。
- 提交代码、配置、文档和可审计的轻量结果；不要提交大模型权重、缓存、虚拟环境或密钥。
- 新实验先运行 allocation tests 和对应的 validate_*，再把结果写入正式报告。
- 使用分支和 Pull Request 协作，具体约定见 CONTRIBUTING.md。
