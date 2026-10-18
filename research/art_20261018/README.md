# ART / amplified restricted training — curated research package

本目录整理了 2026-10-18 完成的放大受限训练（代码中的 `amp`，暂称 ART）算法设计、实现、实验计划、汇总与原始 JSON 运行记录。

## Main result

在 RoBERTa-base / MNLI、10% strong clients、异构 LoRA rank（R=8, r=2）、AdamW 设定下：

- amp：core weak accuracy **80.12 ± 0.14**
- full-train reference：**80.17 ± 0.12**
- homogeneous r8 upper reference：**80.19 ± 0.06**
- restricted L2 owner baseline：**77.85 ± 0.53**

三个配对 seed 中 amp 均超过 L2，平均差约 +2.27 pp；稳定性为 0.35，对照 vref 为 0.37，L2 为 2.69。

Batch 2 的单因素对照：

- amp - FSLoRA-style state sketch：+0.42 pp
- amp - learning-rate compensation：+0.76 pp
- FSLoRA-style Hájek - zero-fill：+2.62 pp
- gradient-path control：约 -0.11 pp

C2 decoder transfer 作为边界实验保留：Qwen2.5-0.5B 上 amp 为 78.36，full-train reference 为 79.87；该结果用于限定适用范围。

## Layout

- `design/`: 算法设计、路线图、文献关系、C2 设计和资源边界
- `code/`: SCRA/ART 实现、测试、运行入口和 Slurm 入口
- `configs/`: 批量实验与 C2 的实际计划 JSON
- `results/`: 汇总 Markdown、汇总 JSON、identity 和逐轮结果 JSON

## Reproduction entry points

在具备依赖和数据/模型环境后，可从以下文件开始：

- `code/README.md`
- `code/run.py`
- `code/make_batch.py`
- `code/make_batch2.py`
- `code/make_c2fed.py`
- `code/tests/test_scra.py`
- `code/tests/test_c2.py`

运行数据、权重、缓存、Slurm 日志和本地环境目录没有纳入该整理分支。
