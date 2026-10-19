# 发布的实验记录

此目录保存小型、可审阅的实验材料：逐轮 JSON、运行身份、汇总、机制探针与 Qwen 源码快照。完整权重、数据、token cache 和 checkpoint 由复现脚本在本地准备。

| 目录 | 内容 |
|---|---|
| `scra_batch_20261018` | RoBERTa 主方法、消融与异构 rank / 强端比例稳健性 |
| `scra_batch2_20261018` | 同梯度路径状态草图、学习率补偿、聚合与外部风格对照 |
| `scra_diagnostics_20261017_v1*` | seed 2040 的受限训练 / 完整训练部分上传等参照，以及 D0/D1/D2/D3 诊断 |
| `headroom_strongfrac_20261017_v1` | 三种子完整秩、低秩与前缀等基础参照 |
| `headroom_coldstart_20261017_v1` | 冷启动与头部相关基础实验 |
| `c2_probe_20261018` | Qwen 的单步机制探针 |
| `c2_fed_20261018` | Qwen 的 38 组学习率、主方法、消融、三种子与版本重放记录 |

从根目录运行 `python scripts/reproduce_results.py`，在 `generated/results/` 生成三份报告与统一主结果表。新训练通过 `scripts/run_experiments.py` 写入独立的 `.reproduction/`。
