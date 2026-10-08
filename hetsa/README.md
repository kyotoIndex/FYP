# HetSA-LoRA：当前实验主线

HetSA 按客户端资源训练最深 6 / 9 / 12 层，每轮训练 LoRA 的 A、B，服务器聚合 A 和分类头，客户端保留私人 B。当前重点是：在异构深度与部分参与下，共享 A 如何帮助本地预测；替换 A 后私人 B 的适配能否带来实际收益。

| 入口 | 内容 |
|---|---|
| [云环境安装与运行](CLOUD_SETUP.md) | 克隆分支、固定依赖、准备模型、运行完整矩阵 |
| [研究交接](docs/RESEARCH_HANDOFF_20261008.md) | 当前问题、证据、方法对照和下一步 |
| [五种子确认说明](docs/HETSA_LORA_CONFIRM_EXTENSION_V1.md) | 已完成 AG_NEWS/BERT-base 配对实验 |
| [新机制冻结协议](docs/HETSA_RESEARCH_PILOT_20261008.md) | 六方法 × 五种子 × 30 轮 |
| [文献阅读报告](../relatedwork/result.md) | 20 篇论文、11 套作者源码、与项目直接对照 |
| [PPT](deliverables/HetSA-LoRA_FYP_汇报_研究逻辑加强.pptx) | 教师汇报；[逐页演讲稿](deliverables/演讲稿_逐页_研究逻辑加强.md) |

已完成的五个确认种子为 202 / 303 / 404 / 505 / 606。本地留出准确率为 94.11%，同深度普通 A+B 聚合为 82.23%，配对差 +11.88 pp；相对全深度 FedSA 为 −0.17 pp。只部署服务器 A 的准确率为 25.48%，因此主指标是客户端个性化预测。

2026-10-08 的新机制矩阵正在本机运行。本次上传只保存了 seed 1201 的原版 HetSA 完整 30 轮快照，位于 [hetsa_research_snapshot_20261008_v1](results/hetsa_research_snapshot_20261008_v1/SNAPSHOT.json)；这是未完成矩阵的快照。云端应按安装说明启动自己的完整运行目录。

## 目录

```text
hetsa/
  src/                已完成实验使用的训练器 + 私人 B 适配模块
  scripts/            历史训练、完整新机制矩阵、结果复算、云准备
  configs/            冻结配置；cloud_*.json 在部署时生成
  results/            已完成确认/深度/SST-2 结果 + 新矩阵完成方法快照
  reproduce/          本机来源清单、打包清单、原执行调度脚本
  docs/               方法、实验与交接说明
  deliverables/       PPT、演讲稿、算法演示
```

`src/fedhello_sst2.py` 与已完成确认实验的冻结 SHA-256 一致。这套训练器作为独立目录运行；仓库根目录保留 PR #2–#4 的共享模型证据。Cloud 配置只更改本地模型路径、输出目录和预先指定的种子集合；六个机制方法及训练预算沿用冻结协议。

## 已完成结果的离线复核

从 `hetsa/` 目录运行：

```bash
python scripts/prepare_cloud.py --data-only
python scripts/summarize_hetsa_lora_confirm_extension.py
python scripts/verify_package.py
```

复算覆盖五个种子全部 750 个方法轮次、参与列表配对、逐客户端留出记录与输入/源码哈希，不需要 GPU 或模型下载。
