# HetSA-LoRA 补充确认实验

状态：协议于 2026-09-28（HKT）冻结，三组新增种子已完成，分析复核通过。

## 问题与判定

检验在相同客户端、数据划分、初始化、训练预算和客户端参与序列下，资源深度受限的 HetSA-LoRA 是否比同层支持上的普通 FedAvg 提高客户端留出集准确率。

主要比较为 `hetsa_lora - resource_depth`。沿用已有方案的门槛：五个确认种子中，每个种子的第 26–30 轮平均 `local_accuracy` 都至少高 1.0 个百分点。若有任何种子低于门槛，照实报告，不改深度、超参数或判定阈值。种子是统计单位；客户端和轮次不作为独立重复。

`fedsa_lora` 是全深度的本地 B 对照，用于判断 HetSA 是否在保留个人 B 的同时承受了资源深度限制。`fedhello` 和 `fedlora` 作为同批次参考。主要指标是每轮客户端留出集的等权平均准确率，对最后五轮取平均。部署服务器指标 `server_accuracy`（A 平均、B=0）与诊断指标 `diagnostic_global_accuracy`（A 平均加客户端 B 的样本加权均值）分开报告；后者不是可部署服务器模型。

## 冻结协议

- 任务：AG_NEWS，BERT-base，20 个客户端，每客户端分配 512 条，Dirichlet 标签偏斜 α=0.3；每轮无放回抽取 10 个客户端，共 30 轮。
- LoRA：Q/V，rank 16；每个客户端按资源档位训练最深的 6、9 或 12 层；本地训练 1 epoch，batch 16，学习率 0.001。
- 方法：`resource_depth`、`hetsa_lora`、`fedsa_lora`、`fedhello`、`fedlora`，沿用原确认配置。
- 新种子：404、505、606。与已有确认种子 202、303 合并分析为五个确认种子；开发种子 101 不进入确认统计。
- 配对：客户端抽样由 `random.Random(seed * 10000 + round_id)` 决定，与方法无关；训练器在每个方法开始时按 seed 重置初始化随机状态。已检查 seed 202、303 的 `allocation_map.csv`，五种方法 30 轮的客户端集合逐轮一致。新种子运行后重复此检查。
- 数据：客户端留出集在训练前固定，每个客户端样本的 20% 留出；同一 seed 的所有方法共享该划分。
- 统计：报告每个 seed 的第 26–30 轮均值与配对差。五 seed 汇总提供均值、标准差和 df=4 的描述性 95% t 区间；不把轮次或客户端当作独立样本。
- 成本与服务器：逐方法报告末轮累计通信量、运行时间、客户端留出准确率、部署服务器准确率和诊断全局准确率。

## 已有结果背景

运行前复算了已有轮次表中最后五轮的客户端留出准确率：seed 202 的 HetSA 相对 `resource_depth` 为 +13.510 pp，seed 303 为 +15.814 pp。两者超过既有 +1.0 pp 门槛；新种子不用于重新选择方法或阈值。

已有 seed 202、303 保存了每轮平均指标，但没有保存逐客户端 `client_holdout.csv`。新运行会保留逐客户端文件以便复核分层结果。

## 执行与复现

执行命令：

```powershell
python scripts/run_full_experiments.py --config configs/hetsa_lora_confirm_extension_v1.json
```

新结果写入 `results/hetsa_lora_confirm_extension_v1/`，不覆盖早期开发或确认结果。

运行时环境：Python 3.11.5、PyTorch 2.5.1+cu121、Transformers 4.32.1、CUDA 12.1 runtime、NVIDIA GeForce RTX 4070。预训练模型使用 Hugging Face 本地缓存 revision `86b5e0934494bd15c9632b12f734a8a67f723594`，配置限定 `local_files_only=true`。

运行前 SHA-256：

| 文件 / 输入 | SHA-256 |
|---|---|
| `configs/hetsa_lora_confirm_extension_v1.json` | `AA86A7C3B22C8F530B8542A85EA6A2636AC768A9E22AD49A997453FF6449FE19` |
| `scripts/run_full_experiments.py` | `07E4DB30019F054923F5E39EDB56C69ACDED202964F1AE6682EF7F40BC618D84` |
| `src/fedhello_sst2.py` | `D22D5A3E58C6CE38E3484F907038BCC722CD41899B339CDF145919AC287D1CAF` |
| `data/AG_NEWS/train.tsv` | `02EAEA97DDFAA1BA0FA169C33A18BF9C8B280A0209AB8DAB526DFFB80248076F` |
| `data/AG_NEWS/dev.tsv` | `419184C5E817E3A450FCCDE842EA872CD745C0FF0CF146F8F7255B18A6671B02` |

## 实验结果与结论

三个新增 seed（404、505、606）均完整运行五种方法，各 30 轮。主指标按每个 seed 第 26–30 轮 `local_accuracy` 求均值；202、303 是先前冻结的确认 seed，101 开发 seed 不纳入统计。

| Seed | HetSA-LoRA | resource_depth | 差值 | +1 pp 门槛 |
|---:|---:|---:|---:|---|
| 202 | 95.29% | 81.78% | +13.51 pp | 通过 |
| 303 | 94.64% | 78.82% | +15.81 pp | 通过 |
| 404 | 94.72% | 76.75% | +17.96 pp | 通过 |
| 505 | 93.44% | 88.37% | +5.07 pp | 通过 |
| 606 | 92.44% | 85.41% | +7.03 pp | 通过 |

五个配对 seed 的平均差为 **+11.88 pp**（SD 5.59 pp；df=4 的描述性 95% t 区间 **[+4.93, +18.82] pp**）。五个 seed 都达到运行前写下的 +1.0 pp 门槛。该区间按 seed 计算，没有把客户端或轮次当作独立样本。

HetSA-LoRA 与全深度 `fedsa_lora` 的本地精度差为 **−0.17 pp**（HetSA 减 FedSA；描述性 95% t 区间 **[−0.76, +0.42] pp**）。当前结果没有显示 HetSA 的本地精度高于 FedSA，也不能据此宣称两者等效。按五个 seed 的配对累计成本均值，HetSA 比 FedSA 少约 **42.5% 运行时间**和 **37.1% 通信量**；相对同层支持的 `resource_depth`，少约 **41.2% 运行时间**和 **49.6% 通信量**。

服务器结果说明了方法适用范围。五 seed 的第 30 轮完整评估中，HetSA 的服务器 A-only 模型平均准确率为 **25.48%**；带客户端私有 B 的本地留出准确率均值为 **94.11%**。`A_avg + mean(B_k)` 的离线诊断值为 **86.81%**，它不能由服务器单独提供。HetSA 的证据支持客户端个性化，不能支持“单一共享服务器模型优于 Fed-HeLLo / FedLoRA”的结论。

新增三 seed 的能力分层结果为次要描述：HetSA 减 `resource_depth` 的末五轮本地准确率差，low / medium / high 分别为 **+12.50 / +8.38 / +0.10 pp**（每层 12 / 6 / 2 个客户端）。这与收益主要出现在低、中资源层相符；高资源层只有两个客户端，且该分析只有三个新增 seed，作为探索性结果报告，不用于主判定。

### 完整性与复现

- 五个 seed 均有 5 种方法、每种 30 轮；每 seed 的 30 轮客户端参与集合在五种方法间逐轮一致。
- 新增 seed 各有 3,000 条客户端留出记录；按相同窗口重算的留出均值与 `local_accuracy` 最大绝对差小于 `2.3e-16`。
- 配置、训练器、源代码和数据哈希均与冻结记录一致。完整性检查和配对分析由 `scripts/summarize_hetsa_lora_confirm_extension.py` 生成。
- `python -m unittest tests.test_allocations -v`：37 项通过；`git diff --check` 通过。
- 复算命令：`python scripts/summarize_hetsa_lora_confirm_extension.py`。

明细见 `results/hetsa_lora_confirm_extension_v1/analysis_v1/`：`paired_differences.csv`、`paired_statistics.csv`、`seed_method_summary.csv`、`round30_server_summary.csv`、`cost_summary.csv`、`capability_paired_summary.csv`、`integrity.csv` 和 `input_hashes.csv`。

**结论：**在这组 AG_NEWS / BERT-base 的异质资源设置中，保留客户端 B、仅聚合 A 的 HetSA-LoRA 相比相同层支持的普通 FedAvg，稳定提高了客户端留出准确率；相对全深度 FedSA，本地精度没有观察到优势，但训练和通信成本更低。方法依赖客户端保存私有 B，单独部署服务器端 A 的准确率接近四分类随机水平。结果支持“低资源客户端的个性化适配”，不构成对全局服务器模型性能的提升证据。
