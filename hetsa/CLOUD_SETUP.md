# 云环境：安装与全规模运行

当前上传分支：`experiment/hetsa-mainline-20261008`。新云环境选择仓库 `kyotoIndex/FYP` 和这个分支，项目根目录为 `FYP/`，训练工作目录为 `hetsa/`。

## 1. 克隆与环境

在已有 GitHub 访问权限的环境中：

```bash
git clone --branch experiment/hetsa-mainline-20261008 --single-branch https://github.com/kyotoIndex/FYP.git
cd FYP/hetsa
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

使用 Python 3.11 与支持 CUDA 12.1 runtime 的 NVIDIA GPU/驱动。本机已完成运行使用 RTX 4070、PyTorch 2.5.1+cu121、Transformers 4.32.1。此目录固定兼容依赖；仓库根目录的 `requirements.txt` 对应另一套环境。

## 2. 准备数据与固定模型版本

AG_NEWS 两份 TSV 已在仓库 `data/AG_NEWS/`。准备脚本在本目录生成两份输入，恢复历史实验的换行字节并核验原始 SHA-256；跨 Windows/Linux 检出时仍能逐字节匹配。

```bash
python scripts/prepare_cloud.py --download-model
```

这会下载 `bert-base-uncased` 的固定 revision `86b5e0934494bd15c9632b12f734a8a67f723594`，生成两份 `configs/cloud_*.json`，使用绝对本地模型路径，之后训练保持 `local_files_only=true`。已有同版本模型时可运行：

```bash
python scripts/prepare_cloud.py --model-dir /absolute/path/to/bert-base-uncased
```

## 3. 当前主线：直接运行完整机制矩阵

```bash
python -u scripts/run_hetsa_research_matrix.py \
  --config configs/cloud_mechanism_v1.json \
  --out-dir results/hetsa_cloud_mechanism_v1
```

固定 **6 方法 × 5 配对种子 × 30 轮 = 900 方法轮次**。种子为 1201 / 1301 / 1401 / 1501 / 1601；20 客户端、每轮 10 个参与、每客户端 512 条、AG_NEWS/BERT-base、Q/V rank16、最深 6/9/12 层、本地 1 epoch、batch16、lr .001。

方法包括原版 HetSA、私人 B 半量/全量 ridge 适配、低频共享 A、完全本地、固定 A/本地 B/共享头。矩阵按顺序占用单张 GPU；留出数据不参与训练或调参。

`RESULT.md` 是动态汇总，`matrix_status.json` 是当前状态，`seed_*.stdout.log` 和 `seed_*.stderr.log` 保存日志。输出路径必须是新目录。中断后只有**已完成的方法**可以保留；如果当时方法只跑了一部分，重新运行该方法必须改用新目录。调度器不会将半段训练当成完整方法：

```bash
python -u scripts/run_hetsa_research_matrix.py \
  --config configs/cloud_mechanism_v1.json \
  --out-dir results/hetsa_cloud_mechanism_v1 --resume
```

上传的本机快照不带训练状态文件，云端从头运行新的完整矩阵。

## 4. 复现已完成的五种子确认

```bash
python -u scripts/run_full_experiments.py \
  --config configs/cloud_confirm_all_v1.json \
  --out-dir results/hetsa_cloud_confirm_all_v1
```

这将运行 202 / 303 / 404 / 505 / 606 的五方法 × 30 轮；和仓库中已完成证据使用不同输出目录。历史统计复算命令见 [README](README.md)。

## 5. 作为云端编码工作环境

优先阅读 `hetsa/AGENTS.md`、`hetsa/README.md`、`hetsa/docs/RESEARCH_HANDOFF_20261008.md` 与 `relatedwork/result.md`。本次操作已提供远端分支和复现实验包；云端 GPU 训练由上述命令启动。
