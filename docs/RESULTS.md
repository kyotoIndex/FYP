# 完成结果与实验口径

2026-10-19。结果取自保存的逐轮轨迹。运行 `python scripts/reproduce_results.py` 可离线重建两份 RoBERTa 汇总和一份 Qwen 报告，并与发布的 JSON 逐项比较；统一表输出至 `generated/results/MAIN_RESULTS.md`。

## 主结果

所有主比较使用训练 seed 2040、2041、2042；数据 seed 为 2026。下表为三种子均值 ± **样本标准差（ddof=1）**，单位为百分比。完整配置、轮次窗口和分区见 [REPRODUCE](REPRODUCE.md)。

| 模型 | 方法 | 2040 | 2041 | 2042 | 均值 ± SD |
|---|---|---:|---:|---:|---:|
| RoBERTa | 完整秩 R=8、全量上传（运行名 r8） | 80.13 | 80.26 | 80.19 | 80.19 ± 0.07 |
| RoBERTa | 完整训练、部分上传 vref | 80.15 | 80.32 | 80.02 | 80.17 ± 0.15 |
| RoBERTa | **AmpLoRA / Hájek** | 80.05 | 80.32 | 79.99 | **80.12 ± 0.17** |
| RoBERTa | 不放大的受限训练 L2 | 77.72 | 77.28 | 78.56 | 77.85 ± 0.65 |
| Qwen | 完整秩 R=64、全量上传（运行名 r8） | 82.52 | 82.27 | 81.80 | 82.19 ± 0.36 |
| Qwen | 完整训练、部分上传 vref_zf | 80.93 | 80.48 | 80.02 | 80.48 ± 0.46 |
| Qwen | **AmpLoRA / 零填充收缩 amp_zf** | 80.87 | 80.30 | 80.05 | **80.41 ± 0.42** |
| Qwen | 完整训练、部分上传 vref / Hájek | 79.92 | 80.30 | 79.38 | 79.87 ± 0.46 |
| Qwen | amp / Hájek | 78.45 | 78.35 | 78.27 | 78.36 ± 0.09 |
| Qwen | FSLoRA 风格 | 80.40 | 80.43 | 79.78 | 80.21 ± 0.37 |

RoBERTa 的本地训练差距由 2.32pp 缩至 0.05pp；Qwen 收缩配置下，受限训练与相同上传参照的差距为 −0.07pp。Qwen 完整全量上传到部分上传参照仍有 1.72pp 差距。RoBERTa 的轮间波动为 amp 0.35、vref 0.37、L2 2.69。

主指标是 **30 个核心弱客户端**的宏平均准确率：这些客户端所在领域没有强客户端。RoBERTa 在第 110、120、130、140、150 轮取均值；Qwen 在第 40、45、50、55、60 轮取均值，使用各客户端最多 40 条的固定验证子集。官方 development 数据另行保存为 final splits；当前研究数字使用上述验证口径。

早期 RoBERTa 批次 1 的报告使用总体 SD（ddof=0），因此原表写作 amp ±0.14、L2 ±0.53；批次 2 与本页使用样本 SD。均值与每 seed 结果相同。

## RoBERTa 机制对照与稳健性

| 比较 / 干预 | 后期准确率变化 |
|---|---:|
| amp − 同梯度路径 FSLoRA 状态草图 | +0.42pp，三种子配对 |
| amp − LoRA 学习率补偿 | +0.76pp，三种子配对 |
| FSLoRA 前向下，Hájek − zero-fill | +2.62pp，三种子配对 |
| amp − FSLoRA 风格 | +2.95pp，三种子配对 |
| amp − Fed-PLoRA 风格 | +2.96pp，三种子配对 |
| amp − FLASC 风格 | +0.79pp，三种子配对 |
| α=2 替代 α=4 | −0.58pp，seed 2040 |
| owner 替代 Hájek | −0.45pp，seed 2040 |
| prefix 替代 balanced 分配 | −1.55pp，seed 2040 |

r=1、混合秩 {1,2,4}、20% 强端三组稳健性实验，分别追回 L2→vref 差距的 1.01、0.89、0.92（seed 2040）。FSLoRA、Fed-PLoRA、FLASC 风格对照在本项目统一 AdamW 协议中实现；具体差异在 [batch2 原始说明](../federated_finetuning/experiments/scra_diagnostics_20261017/FIX_BATCH2_CONTROLS_2026-10-18.md)与[文献核对](designs/literature_check_2026-10-18_amplified_restricted.md)。

[批次 1 完整报告](../federated_finetuning/runs/scra_batch_20261018/SUMMARY.md) · [批次 2 完整报告](../federated_finetuning/runs/scra_batch2_20261018/SUMMARY.md) · [机制与实验图](../federated_finetuning/experiments/scra_diagnostics_20261017/figures/INDEX.md)

## Qwen：聚合方式的反转

Qwen 的 38 组记录包含学习率选择、α 扫描、主比较、定位对照、补 seed、聚合比较及三个完整 60 轮重放。学习率按照第 10 轮开发指标选为 2e-4。L2 为 33.70%，Fed-PLoRA 风格为 33.07%（均为 seed 2040）。

| 三种子配对差值 | 均值 | 95% t 区间 |
|---|---:|---|
| amp_zf − vref_zf | −0.07pp | [−0.34, +0.20] |
| amp_zf − amp（Hájek） | +2.05pp | [+1.23, +2.87] |
| FSLoRA 前向下 zero-fill − Hájek | +2.28pp | [+2.09, +2.47] |
| amp_zf − FSLoRA 风格 | +0.20pp | [−0.56, +0.96] |
| r8 − vref_zf | +1.72pp | [+1.43, +2.00] |

同一方法的本地规则下，两个模型对聚合归一化的偏好不同。Qwen 的弱组件覆盖较少，zero-fill 的收缩提高了精度；服务器步长 0.25 的单 seed 对照为 80.92%，与 amp_zf 的 80.87% 接近。算法通过显式配置选择聚合，详细公式见 [METHOD](METHOD.md)。

三个旧版本代表运行的完整重放全部通过，保存结果中各评测轮的准确率差为零；原有报告的版本核对与协议比较逻辑继续保留。[Qwen 完整报告](../federated_finetuning/runs/c2_fed_20261018/REPORT_C2_1.md)与 [C2 最终设计记录](designs/c2_1_design_2026-10-18.md)列出全部配对比较和判据。

## 资源与后续工作

发布实现使用完整 rank 张量与梯度掩码，实验在单 GPU 进程中模拟客户端和部分上传。通信预算以 LoRA 因子元素数计数（RoBERTa 加上分类头后，弱端总上行约为完整方案的 75%，见 [METHOD §5](METHOD.md)）；folded 参数化的内存与计算估算、实测成本以及源码来源集中记录于 [PROVENANCE](PROVENANCE.md)。

后续工作以论文与展示为主：定名、完成 Qwen 结果图、整理算法框与章节。连续自适应聚合、更大模型与更多任务可作为研究展望。
