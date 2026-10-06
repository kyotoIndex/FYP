# PCLA v7 D0 全量开发读数

本目录只把 D0 的审计通过导出整理成可核查的开发数据。**不在这里选 V 条件，也不把三个开发 seed 的差值写成方法优越性。** 指标均来自 SST-2 官方训练 split 内的 4,096 条配方开发集；旧 SST-2 dev 和 AG_NEWS test 不在此表。协议依据：`PCLA_Longterm_Research_20260928_v7/SCIENTIFIC_PROTOCOL.md` 第 5.1–5.2 节。

CSV 准确率使用 0–1 比例（例如 0.85 = 85%）；差值窗口列 `gap_from_method_best_pp` 的单位是百分点。

全量静态导出 `full_20260928_01` 已通过7个输入文件 SHA-256 核对；351/351运行均为审计通过，117/117配方各有401、409、419三个seed。生成表包含40,085行逐轮记录（含每运行第0轮）、1,577行通信分项、42行表示×学习率曲面和171行逐seed受控差值。117条seed401时间记录均标记为独占 4090 D，且部署前缀没有缺失的 round-wall 记录。117个配方的B8原始均值、每seed端点及停轮还与独立读取351份审计readout的复算文件逐一吻合。具体输入与14个生成文件哈希见 `validation.json`。

有297条运行没有原生“实际尝试通信”日志，对这些运行只可核对已提交的通信账；54条有实际尝试账。18条在200轮上限停下，18条有单列的原生初始化记录。导出工具的`stage_status`字段仍写作`partial_unreviewed`，因此此处用**审计计数与哈希**描述数据完整性；该标签并不构成 V 释放或科学结论。

## 读数口径

- `recipe_117.csv`：117 个配方各三 seed 的原始参数 B8 预算模型准确率、B8/2 和固定第 96–100 轮原始准确率均值。行按配方 ID 排列，没有胜者排序。
- `seed_endpoints_351.csv`：351 个 seed 的相同读数及 B8 停轮、真实部署字节、诊断字节、审计/配置/源码摘要。`b8_raw_accuracy` 是 D0 主要配方读数；第 96–100 轮均值是长期性次要读数，即使这些轮次已经在 B8 之后也不取代预算终点。
- `ema_seed_endpoints_351.csv` 和 `ema_recipe_117.csv`：EMA 读数单列，**不与原始参数读数混用**。
- `rounds_complete.csv`：从第 0 轮到每条轨迹最后一轮的原始/EMA、成本和实际累计通信；`incremental_committed_bytes` 才是该轮增加的字节。B8 后的轮次标 `diagnostic_after_budget`，不能计入预算模型的部署成本。
- `cost_phases.csv`：部署前缀、预算后诊断和存在时的原生初始化分行列出。原生初始化未并入轮次运行时间；W0 冷启动和失败尝试计算耗时没有一致测量。`wire_events.csv` 可继续按方向/消息种类追溯已提交和未提交尝试通信。`native_and_recipe_settings.csv` 保留所有配方的原生优化器、rank、裁剪、head LR、因子日程及其他配置差异。
- `seed401_timing_components_117.csv`：保留 seed401 的 117 条逐项时间记录与 GPU/共享状态，只在相同独占 4090 D 条件下参考。各项与总 wall 时间可能包含、重叠或口径不同；本目录不自行相加定义“总算法时间”，也不据此把共享 4090 D 或 A100 记录直接做时延比较。
- `hardware_sharing_blocks.csv`：名义硬件块与实际 GPU 型号、独占/共享模式的组合及运行条数。时间解释应回到这些块，不跨块合并成速度结论。
- `representation_lr_surface.csv`：local/cyclic × N/D/S/T × LR/周期的 42 个配方原始预算曲面。`controlled_seed_differences.csv` 逐 seed 列出相同表示、LR、周期的 local−cyclic 以及相同方法、LR 的表示−N；N 无刷新周期，后者比较的是完整表示日程。
- `method_candidate_windows.csv`：在每个方法名内标出与该方法三 seed 最佳开发均值相差不超过 0.2 个百分点的候选。**候选并非已选配方**；特别是时间破平局的组成口径仍待研究裁决。
- `all_recipe_curves.html`：可离线打开的 117 张三 seed 原始准确率曲线。实线到各自 B8 停轮，虚线表示其后的诊断轨迹，圆点为 B8 模型；同一横轴轮次不能代替通信对齐。
- `validation.json`：输入导出和配方清单的 SHA-256、输出文件 SHA-256 及行数。脚本核对 117×3 种子、逐轮范围、B8/半预算停轮映射、通信账及成本分项。

CSV 中空白单元格表示该源字段未记录或不适用，**不表示零**。例如没有原生初始化的配方在该成本行缺席，缺失 EMA、失败尝试通信账或 wall 分项仍保持空白；`rounds_complete.csv` 的第 0 轮无训练成本。

生成命令：

```bash
python3 analyze_d0.py \
  --export ../../checks/summaries/full_20260928_01 \
  --recipes ../../../../handoffs/PCLA_Longterm_Research_20260928_v7/jobs/recipes.csv \
  --independent ../../../../handoffs/PCLA_D0_Completed_20260928/independent_recipe_endpoints.json \
  --out .
```

脚本拒绝不满足 117 配方、351 审计通过运行的导出。
