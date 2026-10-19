# C2-0 交接：资源准备与提交

日期：2026-10-18。对象：在 HPC 上准备模型和数据、提交 C2-0 探针的同学。
设计：[docs/designs/c2_0_probe_design_2026-10-18.md](../../../docs/designs/c2_0_probe_design_2026-10-18.md)。代码：本目录下的 `scra/llm.py`、`scra/c2probe.py`，配置 `plan_c2.json`，提交脚本 `run_c2.sbatch`。本地 CPU 测试：31 passed（`tests/test_c2.py` 用的是随机初始化的小 Qwen2）。

下面的路径都相对于项目根目录 `/home/student/t330025090/FYP/federated_finetuning`，与 `plan_c2.json` 一致。

## 1. 需要准备的东西

| # | 内容 | 位置（必须与 plan_c2.json 一致） | 说明 |
|---|---|---|---|
| 1 | **Qwen2.5-0.5B Base 的权重和 tokenizer** | `models/qwen2.5-0.5b/` | 用 Base 模型，不要用 Instruct。需要 `config.json`、`generation_config.json`、`model.safetensors`、`tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt` |
| 2 | MNLI 划分数据（已有） | `data/mnli_r1_seed2026/manifest.json`、`train.jsonl`、`validation.jsonl` | RoBERTa 实验用的就是这一份。jsonl 每行要有 `text`、`text_pair`、`label`、`client` 字段 |
| 3 | Python 环境 | `.venv-hpc` | 需要 `transformers` ≥ 4.37（支持 Qwen2；本地测试用的是 4.57.1）和 `tokenizers` |

计算节点通常不能联网，**第 1 项请在有网络的登录节点上下载**。

## 2. 下载模型（在有网络的节点上）

任选一种方式：

```bash
cd /home/student/t330025090/FYP/federated_finetuning
.venv-hpc/bin/huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir models/qwen2.5-0.5b
```

或者：

```bash
cd /home/student/t330025090/FYP/federated_finetuning
.venv-hpc/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B', local_dir='models/qwen2.5-0.5b'))"
```

下载后**记录版本**，供论文复现用：把 Hugging Face 上 `Qwen/Qwen2.5-0.5B` 当前的 commit hash 写进 `models/qwen2.5-0.5b/REVISION.txt`（页面 "Files and versions" 里可以看到，或者用 `huggingface_hub.model_info('Qwen/Qwen2.5-0.5B').sha`）。

## 3. 快速自检（不需要 GPU 也能跑前两条）

```bash
cd /home/student/t330025090/FYP/federated_finetuning
.venv-hpc/bin/python -c "import transformers, tokenizers; print(transformers.__version__, tokenizers.__version__)"
.venv-hpc/bin/python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('models/qwen2.5-0.5b', local_files_only=True); print([t.encode(w, add_special_tokens=False) for w in (' yes',' maybe',' no')])"
head -c 400 data/mnli_r1_seed2026/train.jsonl
```

第二条命令应当输出三个**只含一个数字**的列表（三个标签词各是一个 token）。如果不是，先告诉我，不要改代码。

## 4. 同步代码并提交（在实验目录下）

先把本地的实验目录同步到 HPC（代码有更新）：

```bash
rsync -av --exclude '__pycache__' --exclude 'logs' /Users/liming/Documents/FYP/federated_finetuning/experiments/scra_diagnostics_20261017/ <hpc-login>:/home/student/t330025090/FYP/federated_finetuning/experiments/scra_diagnostics_20261017/
```

然后**一步一步**提交，每一步确认后再提交下一步（按 C2_0_PROBE_READINESS_REVIEW_2026-10-18.md 的建议）：

**第 1 步：资源自检**

```bash
cd /home/student/t330025090/FYP/federated_finetuning/experiments/scra_diagnostics_20261017
sbatch --time=00:30:00 run_c2.sbatch check
```

`check` 会生成 Qwen 的 MNLI 提示词缓存（`runs/c2_llm_cache_qwen2.5-0.5b/`，只生成一次；以后如果 tokenizer、manifest 或 `max_len` 变了，会报错并要求删除重建），校验缓存与 manifest 的对应关系，加载模型，评测原始模型，跑一步本地训练。结果写在 `runs/c2_probe_20261018/c2check/check.json`，版本信息写在同目录的 `identity.json`。请确认：
- `label_words_single_token` 为 true，`label_token_ids` 是三个数；
- `adapters` 是 168（24 层 × 7 个线性层）；
- `lora_params_millions` 约为 35；
- `cache_splits` 里 train / validation 的行数与 manifest 一致，`max_len` ≤ 256；
- `one_step_loss_finite` 为 true；
- `base_model_accuracy` 记录了原始模型的准确率（三分类随机水平为 33.3%）。

**第 2 步：快照**

```bash
sbatch --time=01:30:00 run_c2.sbatch snap
```

30 轮同质 rank-64 FedAvg，保存第 0、5、30 轮的快照，约 30–45 分钟。每 5 轮写一次检查点，中断后重新提交会从检查点继续。完成后打开 `runs/c2_probe_20261018/c2snap/snapshot_run.json`，看 `eval` 中第 30 轮的 `core_accuracy`：

- **≥ 0.45**：通过可学习性门槛，继续第 3 步；
- **< 0.45**：任务没有学会，先停下来告诉我，不要继续。这时一步后的准确率比较没有意义。

**第 3 步：先只跑一个探针**

```bash
sbatch --time=02:00:00 --array=0 run_c2.sbatch probe
```

记录实际耗时（日志中每轮模拟都会打印秒数）和显存。每完成一轮模拟就写一次部分结果，中断后重新提交会从下一轮继续。

**第 4 步：其余两个探针**

```bash
sbatch --time=02:00:00 --array=1-2 run_c2.sbatch probe
```

## 5. 生成报告并下载

```bash
.venv-hpc/bin/python run.py c2report --plan plan_c2.json     # 在实验目录下，CPU，几秒；可学习性门槛读自 plan 的 learnable_min_core_acc
```

只下载小文件（快照每个约 140 MB，不需要下载）：

```bash
rsync -av --include '*/' --include 'snapshot_run.json' --include 'round_*.json' --include 'REPORT_C2_0.md' --include 'report_c2_0.json' --include 'meta.json' --include 'check.json' --include 'identity.json' --exclude '*' <hpc-login>:/home/student/t330025090/FYP/federated_finetuning/runs/c2_probe_20261018/ /Users/liming/Documents/FYP/federated_finetuning/runs/c2_probe_20261018/
```

## 6. 遇到问题时

| 现象 | 处理 |
|---|---|
| `label words must be single tokens` | 告诉我 tokenizer 的输出，我来换标签词；不要自己改 |
| 找不到模型 / `local_files_only` 报错 | 检查 `models/qwen2.5-0.5b/` 里是否有 `config.json` 和 `model.safetensors` |
| 显存不足 | 不太可能（0.5B，bf16）；把 `logs` 里的报错发给我 |
| `prompt cache ... was built from different inputs` | tokenizer、manifest 或 `max_len` 变了。确认是有意为之后，删除 `runs/c2_llm_cache_qwen2.5-0.5b/` 再重跑 check |
| `does not match snapshots/hashes.json` | 快照文件被改动或损坏。不要手工修复，告诉我 |
| `sdpa` 相关报错 | 把 `plan_c2.json` 中 `"attn_implementation"` 改为 `"eager"` 后重新提交，并告诉我 |

## 7. 这个探针不能说明什么

- **不测资源节省。** 未训练的分量没有折进冻结权重，所以这里记录的峰值显存不能用来支撑资源核算（C1）的 10–14%。那要等 C2-2 的折叠实现。
- **"没有新分类头"的准确说法是：** 没有从零训练的新分类头，但冻结的输出词嵌入仍然参与前向和损失计算。
- **M1 只检验更新的尺度**，不代表任务收益；任务收益要看 C2-1 / C2-2。

---

## 8. C2-1：多轮联邦验证（2026-10-18 追加）

设计：`docs/designs/c2_1_design_2026-10-18.md`。不需要新的资源，模型和提示词缓存都沿用 C2-0。先同步实验目录（代码有更新，新增 `scra/c2fed.py`、`make_c2fed.py`、`run_c2fed.sbatch`、`plans_c2fed/`）。

**阶段 A：学习率检查**（`plans_c2fed/lr_*.json` 已经生成好）

```bash
cd /home/student/t330025090/FYP/federated_finetuning/experiments/scra_diagnostics_20261017
PLANS='plans_c2fed/lr_*.json' sbatch --time=00:40:00 --array=0-3%2 run_c2fed.sbatch
```

跑完后生成报告，里面会按事先写定的规则（第 10 轮核心弱端准确率）给出选中的学习率，并列出诊断项"完整训练与受限训练的本地损失"：

```bash
/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python run.py c2fedreport --plan plans_c2fed/lr_1.0e-04.json
cat ../../runs/c2_fed_20261018/REPORT_C2_1.md
```

**阶段 B：主比较**（把 `<lr>` 换成报告里的 Chosen lr；共 9 次运行：r8、vref、L2、amp、fslora_st、fslora，以及 α = 4 / 16 / 32 的 amp）

先生成配置，再**只跑 amp 一个**做冒烟测试：

```bash
python3 make_c2fed.py main --lr <lr>
PLANS='plans_c2fed/main_amp.json' sbatch --array=0 run_c2fed.sbatch
```

跑完后生成一次报告，确认 `REPORT_C2_1.md` 的 "Mechanism diagnostics" 表里 amp 有第 1、10、20…60 轮的记录，而且数值不是空的（client step、aggregate step 等都有数）。确认无误后再提交全部 9 组（amp 已经完成，会自动跳过）：

```bash
PLANS='plans_c2fed/main_*.json' sbatch --array=0-8%2 run_c2fed.sbatch
```

全部跑完后再运行一次 `c2fedreport`（同上），然后下载小文件：

```bash
rsync -av --include '*/' --include 'fed.json' --include 'identity.json' --include 'REPORT_C2_1.md' --include 'report_c2_1.json' --exclude '*' <hpc-login>:/home/student/t330025090/FYP/federated_finetuning/runs/c2_fed_20261018/ /Users/liming/Documents/FYP/federated_finetuning/runs/c2_fed_20261018/
```

**注意：提交之后不要再改 `scra/` 下的代码。** 检查点和部分结果都记录了源代码哈希；代码一变，中断的任务就会拒绝续跑（这是有意的，避免一次运行混用两版代码）。如果确实需要改代码，先告诉我。

中断的任务重新提交即可（每 5 轮保存一次检查点）。`final_state.pt` 和 `checkpoint.pt` 不需要下载。

---

## 9. 阶段 C：补 seed（2026-10-19 追加）

设计：`docs/designs/c2_1_design_2026-10-18.md` §9。先同步实验目录（`make_c2fed.py`、`scra/c2fed.py`、`tests/` 有更新，`plans_c2fed/seed*_*.json` 已经生成好）。

**注意：** 这次同步会更新 `scra/c2fed.py`，所以如果阶段 B 还有没跑完、需要续跑的任务，会因为代码哈希不一致而拒绝续跑。阶段 B 已经全部完成，不受影响。

```bash
cd /home/student/t330025090/FYP/federated_finetuning/experiments/scra_diagnostics_20261017
PLANS='plans_c2fed/seed*_*.json' sbatch --array=0-7%2 run_c2fed.sbatch
```

8 次运行（seed 2041、2042 各 4 组：r8、vref、amp、fslora），同时跑 2 个，约 2.5 小时。跑完后生成报告，报告末尾会多出 "Stage C: three seeds" 一节：

```bash
/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python run.py c2fedreport --plan plans_c2fed/seed2041_amp.json
```

下载命令与第 8 节相同（会一并下载新的 `seed2041_*`、`seed2042_*` 目录）。

---

## 10. 阶段 D：大模型上的外部定位（2026-10-19 追加）

设计：`docs/designs/c2_1_design_2026-10-18.md` §11。先同步实验目录（`scra/c2fed.py`、`make_c2fed.py`、`tests/` 有更新，`plans_c2fed/main_fslora_paper.json`、`main_fedplora.json` 已经生成好）。之前的阶段都已完成，代码哈希变化不影响它们。

直接列出这两个配置提交（不要用通配符，`main_f*` 会把已经跑完的 fslora、fslora_st 也匹配进来）：

```bash
cd /home/student/t330025090/FYP/federated_finetuning/experiments/scra_diagnostics_20261017
PLANS='plans_c2fed/main_fedplora.json plans_c2fed/main_fslora_paper.json' sbatch --array=0-1%2 run_c2fed.sbatch
```

2 次运行，各约 35 分钟。跑完后重新生成报告（会包含之前所有阶段），报告的判读表里会多出 E1、E2 两行：

```bash
/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python run.py c2fedreport --plan plans_c2fed/main_fslora_paper.json
```

下载命令与第 8 节相同。如果 E1 显示 amp − fslora_paper 小于 1.5pp，按设计 §11 再为 fslora_paper 补 seed 2041、2042（告诉我，我来生成配置）。

---

## 11. 阶段 E：聚合方式的对照、补 seed 与重放检查（2026-10-19，已按两轮审查意见扩充）

设计：`docs/designs/c2_1_design_2026-10-18.md` §13 及其两次修订。先同步实验目录（`scra/c2fed.py`、`make_c2fed.py`、`tests/` 有更新，15 个配置已经生成好）。之前的阶段都已完成，代码哈希变化不影响它们。

```bash
cd /home/student/t330025090/FYP/federated_finetuning/experiments/scra_diagnostics_20261017
PLANS='plans_c2fed/main_*_zf.json plans_c2fed/main_amp_step025.json plans_c2fed/replay_*.json plans_c2fed/seed204[12]_*_zf.json plans_c2fed/seed204[12]_fslora_paper.json' sbatch --array=0-14%2 run_c2fed.sbatch
```

这条命令正好匹配 15 个配置：
- 12 次正式运行：`main_amp_step025`、`main_amp_zf`、`main_fslora_zf`、`main_vref_zf`，以及 seed 2041、2042 的 `amp_zf`、`fslora_paper`、`fslora_zf`、`vref_zf`；
- 3 次完整 60 轮的重放检查：`replay_main_amp`、`replay_main_fslora_paper`、`replay_seed2041_vref`。它们用当前代码重跑三个旧版本中各一个代表性运行。只有重放通过，那个旧版本的结果才会被用于多 seed 比较；不通过时报告会自动阻止相关比较，或改用重放结果。

约 4.5 小时。跑完后重新生成报告：

```bash
/home/student/t330025090/FYP/federated_finetuning/.venv-hpc/bin/python run.py c2fedreport --plan plans_c2fed/main_amp_zf.json
```

报告会多出：重放检查一节（每个旧版本是 pass / fail / incomplete）、版本门控说明（哪些运行被采用、被替代或被阻止）、F1–F3、G1，以及三 seed 的 MS5–MS12。如果有比较显示 "blocked"，说明对应的旧版本没有通过验证，请告诉我。如果生成报告时报"differs from ... in [...]"，说明某个运行的协议字段或模型 / 数据内容与其他运行不一致，请把报错发给我。

下载命令与第 8 节相同。每个新运行目录里多了 `source_snapshot/`（几百 KB 的源码副本），一并下载：

```bash
rsync -av --include '*/' --include 'fed.json' --include 'identity.json' --include 'REPORT_C2_1.md' --include 'report_c2_1.json' --include '*.py' --exclude '*' <hpc-login>:/home/student/t330025090/FYP/federated_finetuning/runs/c2_fed_20261018/ /Users/liming/Documents/FYP/federated_finetuning/runs/c2_fed_20261018/
```
