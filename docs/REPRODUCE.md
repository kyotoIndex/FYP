# 复现指南

本项目把“能运行”与“能作为正式证据报告”分开。复现前先确认配置、模型、数据、seed 和输出目录，复现后再运行结果验证。

## 1. 安装环境

~~~powershell
cd F:\FYP\pcla_experiments
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
~~~

主配置使用 local_files_only=true。因此需要在本机预先准备 bert-base-uncased；使用 RoBERTa 配置时，也需要补齐 models/roberta-base/model.safetensors，仓库只提供 tokenizer/config 文件。

## 2. 先做确定性检查

~~~powershell
python -m unittest tests\test_allocations.py
python scripts\validate_setup.py --config configs\expanded_sst2_bert_base_dirichlet.json
~~~

如果验证失败，先修复环境或路径，不要直接启动长实验。

## 3. 复现经典 baseline 主表

已保存的审计结果是：

~~~text
results/expanded_sst2_bert_base_dirichlet_v1
~~~

验证它：

~~~powershell
python scripts\validate_results.py --config configs\expanded_sst2_bert_base_dirichlet.json --results results\expanded_sst2_bert_base_dirichlet_v1
~~~

重新运行时使用新目录：

~~~powershell
python scripts\run_full_experiments.py --config configs\expanded_sst2_bert_base_dirichlet.json --out-dir results\expanded_sst2_bert_base_dirichlet_v2
~~~

该协议是 SST-2、BERT-base、20 clients、10 selected clients、30 rounds、3 seeds（7/13/21）的受控比较。它不是 Fed-HeLLo 论文的逐项数值复现。

## 4. 复现组件级 PCLA 主线

配置：configs/pcla_large_scale_constrained_dirichlet.json

正式输出：results/pcla_large_scale_constrained_dirichlet_v1

该实验包含 pcla_component_alternating 和严格匹配组件预算的 random_component_alternating，并保留 FedLoRA/Fed-HeLLo 作为参照。协议和结果解释见 PCLA_LARGE_SCALE_CONSTRAINED_V1.md。

## 5. 外部数据和正式矩阵

QNLI 和 AG News 的准备脚本是：

~~~powershell
python scripts\prepare_external_datasets.py
~~~

跨数据集/模型配置位于 configs/formal_v2/，矩阵入口是：

~~~powershell
python scripts\run_formal_v2_matrix.py
~~~

results/formal_v2/matrix_status.json 当前只代表部分运行状态；在所有配置、seed 和方法完成并通过验证前，不要把它称为完成的正式矩阵。

## 6. 生成结果材料

~~~powershell
python scripts\make_formal_comparison_table.py --results results\expanded_sst2_bert_base_dirichlet_v1
python scripts\plot_expanded_results.py --results results\expanded_sst2_bert_base_dirichlet_v1
python scripts\plot_pcla_large_scale.py --results results\pcla_large_scale_constrained_dirichlet_v1
~~~

所有新实验都应使用新的版本化输出目录；不要通过手工删除 CSV 或修改 checkpoint 来“修复”结果。
