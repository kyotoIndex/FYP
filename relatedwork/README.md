# FYP 相关论文与原始 LaTeX 源码

检索与下载日期：2026-09-30（香港时间）。已下载 20 篇论文 PDF，其中 11 篇特别接近的论文附有作者在 arXiv 公开的原始 TeX 工程。

范围：联邦 LoRA 微调、客户端资源与数据异构、层/秩/组件预算分配、因子聚合和个性化。结合当前 `pcla_experiments/docs/HETSA_LORA_V1.md` 与 PCLA/PCLA-WL 项目文档筛选；同时覆盖两条研究线。

阅读更新：20 篇均已阅读核心方法与主要实验，11 篇最接近的论文已精读关键公式、协议、消融和相关附录。完整阅读结果及与当前 FYP 的代码/结果对照见 [result.md](result.md)，逐篇实际范围见 [reading_log.json](reading_log.json)。未复现论文训练实验，未逐行重证所有证明，未完整阅读所有附录；相关性不意味着方法等价或实验可以直接排名。

## 建议阅读顺序

1. HetSA：FedSA-LoRA → Fed-HeLLo / LEGEND → FedAS-LoRA → FedSAIL → FDLoRA。先厘清共享 A、本地 B 的既有基础，再检查异构层预算与个性化的贡献边界。
2. PCLA/PCLA-WL：Fed-HeLLo → Fed-pilot → LEGEND → HAFLQ → Fed-PLoRA → AdaLoRA。重点核对分配单位、重要性定义、真实成本与异构聚合。
3. A/B 训练与聚合：RoLoRA → AS-LoRA → FFA-LoRA → FLoRA / FlexLoRA。注意 AS-LoRA 中的 component 指 A/B 因子，不是 PCLA 的 rank-1 组件。

## 文件组织

- `01_closest/`：11 篇特别接近的论文 PDF。
- `02_related/`：5 篇强相关论文 PDF。
- `03_foundations/`：4 篇基础论文 PDF。
- `tex/方法_arXiv编号版本/`：作者原始 `.tex` 及配套图、`.bib`、模板；保留作者文件名。
- `source_archives/`：arXiv 原始 source 响应备份，文件扩展名 `.source`；实际封装可能为 gzip/tar 或单个 TeX。
- `references.bib`：20 篇的 BibTeX，按 arXiv 元数据生成；正式投稿前按目标刊会核对发表信息。
- `paper_index.csv`：可检索索引。`manifest.json`：作者、固定版本、来源 URL、摘要、文件路径、SHA-256 和下载状态。
- `verification.json`：PDF 解析及文件完整性检查。

## 论文索引

| 方法 | 相关程度 | 年份（初次提交） | 与 FYP 的关系 | 本地 PDF | 官网 |
|---|---|---|---|---|---|
| FedSA-LoRA | 特别接近 | 2024 | HetSA 的共享 A、本地 B 直接基础；须对照贡献边界。 | [PDF](01_closest/FedSA-LoRA_2410.01463v3.pdf) | [arXiv](https://arxiv.org/abs/2410.01463v3) |
| Fed-HeLLo | 特别接近 | 2025 | 按资源与层重要性分配 LoRA 层；PCLA-WL 和异构层预算的直接近邻。 | [PDF](01_closest/Fed-HeLLo_2506.12213v1.pdf) | [arXiv](https://arxiv.org/abs/2506.12213v1) |
| LEGEND | 特别接近 | 2024 | 联合配置 LoRA 深度和秩；对照后层优先与真实设备成本。 | [PDF](01_closest/LEGEND_2412.20004v1.pdf) | [arXiv](https://arxiv.org/abs/2412.20004v1) |
| Fed-pilot | 特别接近 | 2024 | 内存约束下的层分配背包优化及异构聚合；对照成本感知选择。 | [PDF](01_closest/Fed-pilot_2410.10200v2.pdf) | [arXiv](https://arxiv.org/abs/2410.10200v2) |
| Fed-PLoRA | 特别接近 | 2026 | 并行 rank-1 模块与异构客户端；与组件级 PCLA 的分配单位直接相关。 | [PDF](01_closest/Fed-PLoRA_2602.16936v1.pdf) | [arXiv](https://arxiv.org/abs/2602.16936v1) |
| FedAS-LoRA | 特别接近 | 2026 | 比较共享 A 或 B 并自适应选择；检查固定 Share-A 的适用条件。 | [PDF](01_closest/FedAS-LoRA_2608.09742v1.pdf) | [arXiv](https://arxiv.org/abs/2608.09742v1) |
| FedSAIL | 特别接近 | 2026 | 输入感知共享子空间和本地系数；检查共享因子解释与个性化边界。 | [PDF](01_closest/FedSAIL_2609.32485v1.pdf) | [arXiv](https://arxiv.org/abs/2609.32485v1) |
| AS-LoRA | 特别接近 | 2026 | 逐层逐轮自适应选择 A/B 更新因子；这里的 component 不是 PCLA 的 rank-1 组件。 | [PDF](01_closest/AS-LoRA_2605.05769v1.pdf) | [arXiv](https://arxiv.org/abs/2605.05769v1) |
| RoLoRA | 特别接近 | 2025 | 交替优化 A/B；对照 PCLA 交替训练机制，不能等同于保留本地 B。 | [PDF](01_closest/RoLoRA_2502.01755v4.pdf) | [arXiv](https://arxiv.org/abs/2502.01755v4) |
| FDLoRA | 特别接近 | 2024 | 全局与本地双 LoRA 个性化及融合；对照 HetSA 的本地知识保留。 | [PDF](01_closest/FDLoRA_2406.07925v2.pdf) | [arXiv](https://arxiv.org/abs/2406.07925v2) |
| HetLoRA | 强相关 | 2024 | 异构秩、自剪枝和聚合；与异构预算、组件选择相关。 | [PDF](02_related/HetLoRA_2401.06432v2.pdf) | [arXiv](https://arxiv.org/abs/2401.06432v2) |
| FlexLoRA | 强相关 | 2024 | 全尺寸更新聚合后 SVD 重分配异构秩；异构聚合对照。 | [PDF](02_related/FlexLoRA_2402.11505v2.pdf) | [arXiv](https://arxiv.org/abs/2402.11505v2) |
| FLoRA | 强相关 | 2024 | 堆叠聚合异构 LoRA，讨论因子分别平均的聚合误差。 | [PDF](02_related/FLoRA_2409.05976v1.pdf) | [arXiv](https://arxiv.org/abs/2409.05976v1) |
| FFA-LoRA | 强相关 | 2024 | 冻结 A、训练并共享 B；与共享 A、本地 B 的设计区别。 | [PDF](02_related/FFA-LoRA_2403.12313v1.pdf) | [arXiv](https://arxiv.org/abs/2403.12313v1) |
| HAFLQ | 特别接近 | 2024 | 重要性感知截断、冻结及仅聚合更新过的 rank-1 矩阵；组件级 PCLA 的直接近邻，另包含异构量化。 | [PDF](01_closest/HAFLQ_2411.06581v2.pdf) | [arXiv](https://arxiv.org/abs/2411.06581v2) |
| SLoRA | 强相关 | 2023 | 非 IID 下联邦 PEFT 与数据驱动初始化。 | [PDF](02_related/SLoRA_2308.06522v1.pdf) | [arXiv](https://arxiv.org/abs/2308.06522v1) |
| LoRA | 基础 | 2021 | 低秩适配的基本矩阵定义与参数效率。 | [PDF](03_foundations/LoRA_2106.09685v2.pdf) | [arXiv](https://arxiv.org/abs/2106.09685v2) |
| AdaLoRA | 基础 | 2023 | 重要性感知的自适应秩预算；非联邦基础工作。 | [PDF](03_foundations/AdaLoRA_2303.10512v2.pdf) | [arXiv](https://arxiv.org/abs/2303.10512v2) |
| FedIT | 基础 | 2023 | 联邦指令微调背景；不要将项目中统一 rank 的 fedlora 控制等同于完整论文复现。 | [PDF](03_foundations/FedIT_2305.05644v2.pdf) | [arXiv](https://arxiv.org/abs/2305.05644v2) |
| FedAvg | 基础 | 2016 | 联邦平均及非 IID 问题的基础。 | [PDF](03_foundations/FedAvg_1602.05629v4.pdf) | [arXiv](https://arxiv.org/abs/1602.05629v4) |

## 特别接近论文的 TeX 主文件

这里提供的是作者公开的原始源码，不是 PDF 转换出的近似文本，也不是我重写的论文。下载版 PDF 和源码固定为同一个 arXiv 版本。

| 方法 | 论文标题 | 主文件入口 | TeX 文件数 |
|---|---|---|---|
| FedSA-LoRA | Selective Aggregation for Low-Rank Adaptation in Federated Learning | [main.tex](tex/FedSA-LoRA_2410.01463v3/main.tex) | 3 |
| Fed-HeLLo | Fed-HeLLo: Efficient Federated Foundation Model Fine-Tuning with Heterogeneous LoRA Allocation | [arxiv.tex](tex/Fed-HeLLo_2506.12213v1/arxiv.tex) | 1 |
| LEGEND | Adaptive Parameter-Efficient Federated Fine-Tuning on Heterogeneous Devices | [main.tex](tex/LEGEND_2412.20004v1/main.tex) | 10 |
| Fed-pilot | Fed-pilot: Optimizing LoRA Allocation for Efficient Federated Fine-Tuning with Heterogeneous Clients | [main.tex](tex/Fed-pilot_2410.10200v2/main.tex) | 1 |
| Fed-PLoRA | Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation | [main_cameraready.tex](tex/Fed-PLoRA_2602.16936v1/main_cameraready.tex) | 2 |
| FedAS-LoRA | Rethinking Factor Sharing in Federated LoRA: A Rank-Aware Adaptive Approach | [main_arxiv.tex](tex/FedAS-LoRA_2608.09742v1/main_arxiv.tex) | 1 |
| FedSAIL | What Should Federated LoRA Share? FedSAIL via Input-aware Subspace Alignment | [main.tex](tex/FedSAIL_2609.32485v1/main.tex) | 1 |
| AS-LoRA | Adaptive Selection of LoRA Components in Privacy-Preserving Federated Learning | [main.tex](tex/AS-LoRA_2605.05769v1/main.tex) | 8 |
| RoLoRA | Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA | [neurips_2025.tex](tex/RoLoRA_2502.01755v4/neurips_2025.tex) | 2 |
| FDLoRA | FDLoRA: Personalized Federated Learning of Large Language Model via Dual LoRA Tuning | [sample-sigconf.tex](tex/FDLoRA_2406.07925v2/sample-sigconf.tex) | 2 |
| HAFLQ | HAFLQ: Heterogeneous Adaptive Federated LoRA Fine-tuned LLM with Quantization | [main.tex](tex/HAFLQ_2411.06581v2/main.tex) | 1 |

主文件入口由 `\documentclass` 和 `\begin{document}` 检查确定。源码仅下载和解包，未编译；是否可直接编译还取决于多文件依赖、作者模板、字体和 TeX 环境。

## 引用时要核对的区别

- FedSA-LoRA 已提出共享 A、保留本地 B；我们不能把这一点作为首次提出的贡献。HetSA 应对照异构层预算下的本地效果和实际成本。
- HAFLQ 包含重要性感知截断/冻结与 rank-1 矩阵聚合；它与组件级 PCLA 很接近，应优先读方法部分，不能只将它视为量化工作。
- 层数、秩数、逻辑 mask、实际计算开销是不同量。对照论文时需区分训练时间、峰值显存与上传/下载字节。
- 本地个性化准确率与单个共享服务器模型准确率需分别报告，不能互换。
- 项目代码里的统一 rank `fedlora` 基线是受控实现名称，不能仅凭名称认定为某篇 FedLoRA/FedIT 论文的完整复现。

## 检索中发现但尚未纳入下载的候选

- **HRALoRA**：Heterogeneous-Resource-Aware Federated Learning with Intelligent LoRA Allocation and Aggregation。检索命中 CCGrid 2026，DOI [10.1109/CCGrid68966.2026.00043](https://doi.org/10.1109/CCGrid68966.2026.00043)。未取得可验证的公开 PDF 下载地址及作者 TeX，因此不计入已下载的 20 篇。

## 旧目录中的引用编号问题

`F:/FYP/references/papers/FedIT_2303.10512v2.pdf` 的文件名把 FedIT 与 AdaLoRA 的 arXiv 编号混在一起。官网确认 `2303.10512` 对应 AdaLoRA，FedIT 对应 `2305.05644`。本目录分别下载两篇正确编号的论文；旧文件未改动，旧 PDF 的实际正文身份未在本次核对。

## 可重复下载

在安装了 requests 的 Python 环境中运行 `python F:/FYP/relatedwork/download_papers.py`。该脚本从官网读取版本并下载，对不同内容的已有文件拒绝覆盖。原始 source 只安全解包普通文件，不执行任何作者代码。再运行 `python F:/FYP/relatedwork/build_index.py` 更新此索引。
