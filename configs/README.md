# 配置目录

配置文件是实验协议的入口。每次新实验应新增或明确版本化配置，并把 output_dir 指向新的结果目录。

- expanded_sst2_bert_base_dirichlet.json：经典 baseline 主表。
- pcla_large_scale_constrained_dirichlet.json：组件级 PCLA 主线。
- legend_large_scale_constrained_dirichlet.json：LEGEND 对照。
- formal_v2/：跨数据集、模型和异质性设置的矩阵配置。
- development/：开发验证配置，不作为正式证据。

配置中的数据和模型路径相对于项目根目录解析。运行前先看 dataset、federated、methods 和成本模型字段。
