# HetSA-LoRA v1

Status: depth analysis and SST-2 completed. Local gain is largest on
low-resource clients; SST-2 reproduces the personalization, with resource_depth
collapsing on low/medium seeds 202 and 303. Date: 2026-09-21.

Development seed 101 is kept. The complete claim “beat Fed-HeLLo globally and
personalize locally” is withdrawn. SplitSA-LoRA is a closed ablation, not a
replacement: splitting `B` kept local accuracy but dropped the server to 0.830.

## Claim

Under a heterogeneous LoRA depth budget, keeping `B` local while averaging `A`
personalizes on the same layers as resource-depth FedAvg, at the cost of not
having a Fed-HeLLo-level server model.

Primary metric: client hold-out `local_accuracy`. Global accuracy is a cost,
not a target to beat Fed-HeLLo.

## Decision

Personalization happens in aggregation membership, not in a score mask.

- Resource budget chooses a contiguous deepest suffix, same as `resource_depth`.
- On those layers both LoRA factors train.
- Only `A` is uploaded and FedAvg'd. `B` stays on the client across rounds.
- The classifier head is still globally averaged.

`hetsa_lora` is FedSA-LoRA under a heterogeneous layer budget. Novelty is the
question, not the factor split: when a device can train only some LoRA layers,
does keeping `B` local beat averaging both factors on the same support?

## Methods

| Name | Trainable layers | Aggregate |
|---|---|---|
| `resource_depth` | deepest `q_k` ∈ {6,9,12} | `A` and `B` |
| `hetsa_lora` | same suffix | `A` only; persist `B` |
| `fedsa_lora` | all 12 layers | `A` only; persist `B` |
| `fedhello` | HLA | `A` and `B` of selected layers |
| `fedlora` | all 12 layers | `A` and `B` |

No FIM, no PCLA scores, no residual adapter.

## Metrics

Report both. Do not substitute one for the other.

1. **Local `local_accuracy` (primary):** mean per-client hold-out. Share-A uses
   `A_avg + B_k`. The other methods use the aggregated global model on the same
   hold-out slices.
2. **Global `accuracy` (cost):** held-out AG News eval. For Share-A methods this
   is `A_avg` plus the sample-weighted mean of stored `B_k` (a diagnostic, not
   a serving model). `server_accuracy` is `A_avg` with `B=0`.

## Serving models

Do not mix these three numbers.

| Name | Adapter | Where it is used |
|---|---|---|
| Personalized client | `A_avg + B_k` | Primary. Each device after download. |
| Deployable server | `A_avg`, `B=0` | What the cloud can serve alone. Chance-level on AG News. |
| Diagnostic global | `A_avg + mean_k B_k` | Analysis only. Not a model anyone deploys. |

`accuracy` in Share-A CSV rows is the diagnostic. `server_accuracy` /
`deployable_server_accuracy` is the cloud model. `local_accuracy` is the
client. Non-Share-A methods use the same aggregated adapter for client and
server.

## Depth analysis

Old CSVs only store the mean over all 20 clients. Per-capability hold-out is
in `client_holdout.csv` under `results/hetsa_lora_depth_v1/` and
`results/hetsa_lora_sst2_v1/`. Tables: `results/hetsa_lora_analysis_v1/`.

Three-seed last-round local, AG News:

| Method | low (q=6, n=12) | medium (q=9, n=6) | high (q=12, n=2) |
|---|---|---|---|
| resource_depth | 86.9 | 89.3 | 91.3 |
| HetSA | 94.9 | 94.3 | 95.8 |
| FedSA | 95.2 | 94.3 | 95.1 |
| HetSA − resource | **+8.0** | **+5.0** | **+4.4** |

The gain is not carried by high-resource clients. Low-budget devices, whose
entire adapter *is* the shared suffix, still pick up +8 pp from keeping `B`
local.

SST-2, same federated protocol, last-round local mean: HetSA 94.8%, FedSA
94.7%, resource_depth 65.5%. resource_depth falls apart on seeds 202/303 for
low/medium clients (hold-out near chance); HetSA stays in the low–mid 90s on
every tier. Deployable server for Share-A on SST-2 is 50.9% (majority/chance).

- Development: seed 101, `results/hetsa_lora_development_v1/seed_101`
- Confirmatory: seeds 202 and 303, `results/hetsa_lora_confirmatory_v1/`
- Capability split: `results/hetsa_lora_depth_v1/`
- SST-2 transfer: `results/hetsa_lora_sst2_v1/`

## Reporting bar

- Local mean over rounds 26–30: HetSA − resource_depth should stay ≥ +1.0 pp
  across confirmatory seeds. If a seed falls under, report it; do not retune.
- Do not retune the suffix, do not move residual onto shallow layers, do not
  revive score-based masks, do not split `B` ranks to chase the server.

## Protocol

AG News, BERT-base, 20 clients, 10 selected, 512 examples, 30 rounds,
constrained Dirichlet α=0.3, LoRA rank 16 on query/value, max length 96.

SST-2 uses the same federated/LoRA/resource protocol with `max_length` 64 and
two labels.
