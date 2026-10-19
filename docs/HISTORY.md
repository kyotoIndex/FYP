# 研究演进与历史版本

正式仓库当前围绕 AmpLoRA（早期名称 ART / `amp`）组织。研究由“只上传部分组件”和“只训练部分组件”的差异出发，发展为本地增量放大，以及两个覆盖设置下的聚合比较。

| 阶段 | 历史入口 |
|---|---|
| PCLA、组件选择、通信预算与早期对照 | [整理前的 main 快照](https://github.com/kyotoIndex/FYP/tree/c0c3ee509926ad03ff3f3b8d773e9eb59cf46e8b) |
| HetSA 阶段与云环境包 | [历史 PR #5](https://github.com/kyotoIndex/FYP/pull/5) |
| ART 早期归档（RoBERTa 主结果与当时 Qwen/Hájek 结果） | [research/art-20261018-results](https://github.com/kyotoIndex/FYP/tree/research/art-20261018-results/research/art_20261018) |
| 最新方法、Qwen 收缩聚合、三种子与版本重放 | [RESULTS](RESULTS.md)、[方法设计底稿](designs/algorithm_design_2026-10-18_amplified_restricted.md)、[C2 最终实验记录](designs/c2_1_design_2026-10-18.md) |

本次整理以原 main 为父提交，保留完整 Git 历史；旧代码从当前工作树移出。原研究工作区的历次实验与证据仍保存在原位置，本地正式 checkout 单独建立。

阅读早期底稿时，请按文内日期和阶段区分当时的计划与后来完成的结果。当前方法说明以 `METHOD.md` 为准，结果以保存轨迹的复算值为准，资源口径集中在 `PROVENANCE.md`。
