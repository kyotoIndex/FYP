# PCLA Online-Layer Screening v1

## Status

This screening run is complete and audited. It is an algorithm-development
result, not a replacement for the canonical
`expanded_sst2_bert_base_dirichlet_v1` evidence.

Configuration:

```text
configs/pcla_online_layer_screening.json
```

Results:

```text
results/pcla_online_layer_screening_v1
```

Protocol:

- SST-2 and `bert-base-uncased`
- five clients with capacities 6, 6, 6, 9, and 12 layers
- Dirichlet label skew with alpha 0.3
- all five clients selected in each round
- 12 communication rounds
- seeds 7, 13, and 21
- equal fixed LoRA rank 16 for all four methods

Compared methods:

- `fedra_random_partial`
- `fedhello`
- `personalized_cost_aware_lora` (frozen PCLA-v1)
- `pcla_online_layer`

## Online-Layer Mechanism

`pcla_online_layer` removes explicit resource-based client clustering. Each
client maintains its own layer-importance EMA, refreshed every three rounds
from one local batch. Layer scores combine:

```text
standardized EMA importance
+ UCB-style exploration bonus
+ layer-age freshness bonus
+ previous-selection stability bonus
```

The resource capability controls only the number of selected layers. Online
importance overhead is included in simulated cumulative time.

## Validation

- 15 allocation and scheduling tests pass.
- All three seeds contain 48 metric rows and 240 allocation rows.
- Each seed contains 864 importance rows, including 720 online per-client,
  per-layer diagnostic rows.
- The result validator reports all allocation invariants as valid.
- PCLA-v1 clustering collapses to one cluster in the round-one selected clients
  for all three seeds, independently confirming the known v1 limitation.

## Results

| Method | Final accuracy mean | Last-5-round accuracy mean | Accuracy AUC mean | Time mean (s) | Importance overhead / time | Mask stability |
|---|---:|---:|---:|---:|---:|---:|
| FedRA random partial | 0.7548 | 0.7528 | 0.6822 | 70.90 | 0.00% | 0.524 |
| Fed-HeLLo | 0.7983 | 0.7476 | 0.6805 | 74.23 | 4.30% | 0.555 |
| PCLA-v1 | 0.7283 | 0.7153 | 0.6699 | 72.97 | 0.95% | 1.000 |
| PCLA online-layer v1 | 0.7325 | 0.7179 | 0.6578 | 72.83 | 2.69% | 0.948 |

Per-seed final accuracy:

| Seed | Random partial | Fed-HeLLo | PCLA-v1 | Online-layer |
|---:|---:|---:|---:|---:|
| 7 | 0.8297 | 0.8170 | 0.8098 | 0.8098 |
| 13 | 0.8804 | 0.8877 | 0.8822 | 0.8859 |
| 21 | 0.5543 | 0.6902 | 0.4928 | 0.5018 |

All methods use the same 176.20 MB communication except PCLA-v1, whose
importance-derived nonuniform ranks increase mean communication to 186.52 MB.

## Decision

The online-layer method does not pass the screening gate. It slightly improves
PCLA-v1 final and late-round mean accuracy, while reducing communication to the
same level as random partial, but it does not outperform random partial.

The main diagnostic is insufficient exploration:

- online mask stability is 0.948, versus 0.524 for random partial;
- low-resource clients use only one to four unique masks over 12 rounds;
- most clients change masks at most three times;
- standardized importance dominates the configured exploration coefficient
  (`beta=0.15`), so under-selected layers do not receive enough trials.

Seed 21 is also a general robustness warning. Random partial degrades to 0.554,
while both PCLA variants remain near chance. Any next screening must retain this
seed rather than reporting only seeds 7 and 13.

## Next Controlled Step

Before implementing component-level PCLA-v2, run a small exploration ablation
without changing the completed v1 output:

```text
beta = {0.15, 0.5, 1.0}
stability_delta = {0.0, 0.05}
```

Use a new configuration and output directory. Select candidates by last-five
round accuracy and AUC, not final-round accuracy alone. If stronger exploration
still cannot beat random partial, stop layer-level score tuning and move to
rank-component masking with alternating A/B training.
