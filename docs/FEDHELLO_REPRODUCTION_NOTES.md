# Fed-HeLLo Controlled Reproduction

This directory keeps the Fed-HeLLo reproduction protocol separate from the
project's LEGEND reproduction and proposed PCAL experiments.

## What Was Verified From The Paper

The paper evaluates Fed-HeLLo on five datasets:

| Modality | Dataset | Model | Metric |
|---|---|---|---|
| Vision | CIFAR-100 | ViT-base | top-1 accuracy |
| Vision | DomainNet-121 | ViT-base | top-1 accuracy |
| Language classification | LEDGAR | BERT-base | macro-F1 |
| Instruction tuning | Natural Instruction | DataJuicer-1B | Rouge-L |
| Instruction tuning | Dolly-15K | OPT-1.3B | Rouge-L |

The paper protocol is:

| Setting | Classification | Instruction tuning |
|---|---:|---:|
| Total clients | 100 | 50 |
| Selected clients per round | 10 | 10 |
| Local epochs | 1 | 1 |
| Communication rounds | 500 | 25 |
| LoRA rank | 16 | 8 |
| LoRA target modules | query, value | query, value |
| LoRA alpha | 16 | 16 |
| LoRA dropout | 0.1 | 0.1 |

Resource heterogeneity is represented by three layer-capability levels:

- low-resource clients can train one half of the LoRA layers;
- medium-resource clients can train three quarters of the LoRA layers;
- high-resource clients can train all LoRA layers.

The paper evaluates both resource ratios `6:3:1` and `1:1:1`. The first ratio
is the practical setting with more low-resource clients. The server selects
clients uniformly at random each round.

## HLA Variants

The reproduction must keep these methods distinct:

- `straggler`: every selected client trains all LoRA layers; round cost is determined by the slowest client.
- `exclusive_roc_6_3_1` / `exclusive_roc_1_1_1`: only clients that can train all layers participate.
- `fedra`: random partial-layer allocation with the same resource capacities.
- `fim_hla`: Fisher/gradient-norm allocation using a held-out proxy set.
- `gd_hla_triangle`: deterministic shallow-layer geometric allocation.
- `gd_hla_inverted_triangle`: deterministic deep-layer geometric allocation.
- `gd_hla_bottleneck`: deterministic shallow-plus-deep allocation.
- `gd_hla_uniform`: uniform random geometric allocation.
- `rgd_hla`: randomized geometric allocation.
- `fedhello`: `RGD-HLA` warm start followed by periodic `FIM-HLA` updates.

The paper's combined Fed-HeLLo schedule is:

| Task type | `T_RGD` | `T_FIM` |
|---|---:|---:|
| Classification | 50 | 50 |
| Instruction tuning | 1 | 2 |

For the classification `fedhello` run, use RGD-HLA for rounds 1--50 and refresh
the FIM allocation every 50 rounds afterward. For instruction tuning, use
RGD-HLA in round 1 and refresh the FIM allocation every 2 rounds.

## Local Controlled Protocol

The original protocol is too large for the current local workspace and uses
models/datasets that are not currently installed here. The locally reproducible
protocol is therefore a mechanism reproduction, not an exact numerical
reproduction of the paper:

```powershell
cd F:\FYP
python scripts\validate_setup.py --config configs\controlled_sst2.json
```

The controlled protocol uses the existing real SST-2 files and `prajjwal1/bert-mini`,
with 5 clients, 12 rounds, 3 seeds, and the paper's three capability levels
mapped to 1/2, 3/4, and all four BERT encoder LoRA layers. It is intended to
verify the Fed-HeLLo mechanism before attempting the original 100-client
classification or 50-client instruction-tuning runs.

The paper LoRA hyperparameters are retained: query/value targets, rank 16,
alpha 16, dropout 0.1, and one local epoch. To prevent proxy leakage, the 872
SST-2 development examples are deterministically split per seed into 320 server
proxy examples and 552 disjoint evaluation examples.

The short controlled run deliberately adapts the classification schedule to
one RGD-HLA warm-start round followed by FIM refresh every round. Using the
paper's `T_RGD=50` in a 12-round run would never enter the FIM phase and would
not test the combined Fed-HeLLo mechanism.

## Complete Runs

Validate code, allocation invariants, and inputs:

```powershell
cd F:\FYP
python -m unittest tests\test_allocations.py
python scripts\validate_setup.py --config configs\controlled_sst2.json
```

The intended BERT-mini matrix is:

```powershell
python scripts\run_full_experiments.py --config configs\controlled_sst2.json
```

On the currently inspected machine, the cached `prajjwal1/bert-mini` checkpoint
contains only `pytorch_model.bin`. Transformers refuses that pickle-based
format with the installed PyTorch 2.5.1 because of CVE-2025-32434. Do not bypass
that safety check. This exact mini configuration requires PyTorch 2.6 or later,
or an equivalent trusted safetensors checkpoint.

The same command is resumable at completed-method boundaries. Re-running it
skips methods already recorded in each seed's `checkpoint.json`. A changed
configuration must use a new output directory to prevent incompatible results
from being combined.

The immediately runnable local protocol is therefore the controlled BERT-base
configuration. It preserves
the same 5 clients, 12 rounds, 3 seeds and uses capacities 6/9/12 layers:

```powershell
python scripts\run_full_experiments.py --config configs\controlled_sst2_bert_base.json
```

The local RTX 4070 Ti SUPER test with sequence length 64 and batch size 16 used
approximately 0.91 GiB peak allocated CUDA memory for one full-layer
forward/backward step. BERT-base is therefore VRAM-feasible; elapsed time for
the full 10-method by 3-seed matrix is the practical constraint.

The complete controlled BERT-base matrix was run on 2026-07-25 with Python
3.12.6, PyTorch 2.5.1+cu121, Transformers 4.57.1 and an RTX 4070 Ti SUPER. Its
audited output is `results/controlled_sst2_bert_base_v1`. Reproduce into an
independent directory and audit it with:

```powershell
python scripts\run_full_experiments.py `
  --config configs\controlled_sst2_bert_base.json `
  --out-dir results\controlled_sst2_bert_base_v2

python scripts\validate_results.py `
  --config configs\controlled_sst2_bert_base.json `
  --results results\controlled_sst2_bert_base_v2
```

## Required Output

Every complete run must write:

- `run_config.json`
- per-round global metric rows
- per-client allocation maps
- per-client importance scores
- trainable parameter count and communication volume
- round time and average waiting time
- final metric and metric-to-target summary

Per-seed files are stored below the configured output directory. The final
cross-seed table is `aggregate_summary.csv`, reporting mean and sample standard
deviation. Communication is logged as upload plus download of trainable
parameters; `one_way_payload_mb` is also kept in the allocation map.

## Interpretation Boundary

Passing unit tests means every allocation path and periodic schedule is
executable and structurally valid. It is not experimental evidence. Only the
completed 12-round, 3-seed matrix should be used for controlled comparisons.

This implementation is paper-derived because the official repository source
was not available for line-by-line verification in the current environment.
Any source-level behavior that the paper does not specify exactly must be
reported as a reproduction assumption, especially the deterministic
`bottleneck` RGD prior, fixed-rank layer-wise aggregation, and the controlled
one-round RGD warm start.

Do not report a controlled SST-2 result as the original Fed-HeLLo number. In the
thesis, label it as `Fed-HeLLo-core` or `Fed-HeLLo-like controlled reproduction`
unless the original paper protocol is used.

## Formal SST-2 Comparison

The directly comparable BERT-base experiment is defined by
`configs/formal_comparison_sst2_bert_base.json`. It runs all methods inside the
same runner with the same SST-2 split, model initialization policy, client data,
batch order, 5 clients, 12 rounds, and seeds 7/13/21. The 320 proxy examples and
552 evaluation examples are disjoint.

FedLoRA, Fed-HeLLo, LEGEND-like, importance-only, importance-resource LoRA and
personalized cost-aware LoRA use the same total LoRA rank budget of 192 across
12 BERT layers. LEGEND and importance-aware methods redistribute that budget;
resource-aware methods additionally restrict each client to its 6/9/12-layer
capacity. Communication is upload plus download of the actual active LoRA and
classifier tensors.

Global importance-aware methods estimate gradient-norm importance on the server
proxy set. Personalized cost-aware LoRA estimates each client's importance from
two local training batches. Neither source includes the 552 evaluation samples.
Importance-estimation elapsed time is included in cumulative time, while PyTorch
RNG state is restored afterward so allocation-only passes do not change formal
training dropout sequences.

Run and audit the formal experiment:

```powershell
python scripts\run_full_experiments.py `
  --config configs\formal_comparison_sst2_bert_base.json

python scripts\validate_results.py `
  --config configs\formal_comparison_sst2_bert_base.json `
  --results results\formal_comparison_sst2_bert_base_v1
```

The completed audited result is in
`results/formal_comparison_sst2_bert_base_v1`. It contains 504 round rows and
2,520 client-allocation rows. `formal_comparison_table.csv` reports the main
mean/std results and relative savings against FedLoRA.

The present three-seed result is descriptive evidence, not a statistical
significance claim. The original PCAL absolute-threshold cluster assignment
placed all five clients in cluster 1 for every seed. PCAL still makes
client-specific decisions through resource capacities and personalized layer
costs, but cluster-specific importance collapsed to a shared cluster mean. This
must be disclosed; do not claim that this run demonstrated multiple learned
client clusters.

