# 协作与结果维护

使用独立分支和 Pull Request。开发入口为 `pip install -e '.[test,figures]'`，提交前运行 `pytest -q` 和 `python scripts/reproduce_results.py`。

修改算法时，在 PR 里说明改变了分配、本地规则、梯度路径或聚合中的哪一项，以及对应对照。`federated_finetuning/runs/` 是随仓库发布的实验记录；新训练放在 `.reproduction/` 或通过 `--project-root` 指定的独立目录。保留每次运行的配置、seed、完整轮次记录和身份文件。

模型、原始数据、token cache、checkpoint、虚拟环境、凭证和调度日志不进入 Git。论文图以可编辑 Draw.io / SVG 和生成脚本维护；文档数字从保存轨迹计算。更改方法或结果时同步维护 `docs/METHOD.md`、`docs/RESULTS.md` 和复现命令。

历史研究路线见 `docs/HISTORY.md`。项目名称 AmpLoRA 为写作阶段暂名，引用正式论文时以最终定名和出版信息为准。
