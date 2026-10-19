# 正式仓库整理验证

日期：2026-10-19（北京时间）。验证对象为本次 AmpLoRA 主线整理。

| 工作 | 结果 |
|---|---|
| 独立环境安装 | 新建 Python 3.12 环境，`pip install -e '.[test,figures]'` 对应依赖安装成功 |
| 模型与算法现有测试 | 46 passed，覆盖 scra 与 headroom |
| 三份核心结果 JSON | 从发布轨迹重新生成，与原文件逐项一致 |
| Qwen 运行覆盖 | 38 个配置均有 fed.json / identity.json，达到各自计划轮数 |
| Qwen 源版本重放 | 保存的三个完整重放全部 pass，三种子比较齐全 |
| 官方 MultiNLI 下载与分区重建 | 实际下载官方 ZIP 并运行完整准备脚本；train 372557、validation 19681、final matched 9815、final mismatched 9832、跨领域排除 96 行 |
| 固定模型 revision | 两个远端快照均可访问，safetensors 权重存在 |
| 实验图重建 | 58 条运行曲线生成 5 张图，图表脚本全部 712 项数值核对通过 |
| 中英文方法图 | 四份 scene / SVG / Draw.io 可自包含重建；中文两份 PDF 已补齐并检查 |
| 文档与提交内容 | 本地 Markdown 文件链接无缺失，提交排除模型、数据、cache、checkpoint、环境和凭证；diff whitespace 检查通过 |

本次重新执行安装、数据构造、代码测试、轨迹复算与制图。完整 GPU 训练使用已完成的实验记录；后续从零训练按 [REPRODUCE](REPRODUCE.md) 在独立工作目录执行。验证用的临时数据与方法图目录已自动清理，`generated/` 为本地可再生成输出。
