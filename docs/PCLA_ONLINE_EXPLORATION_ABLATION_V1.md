# PCLA Online Exploration Ablation v1

## Scope

This completed ablation tests whether stronger exploration can make online
layer-level PCLA outperform an equal-capacity random partial-layer baseline.

Configuration:

```text
configs/pcla_online_exploration_ablation.json
```

Results:

```text
results/pcla_online_exploration_ablation_v1
```

The experiment uses the same five-client, 12-round, three-seed SST-2/BERT-base
protocol as the preceding online-layer screening.

## Results

| Method | Final mean | Last-5 mean | AUC mean | Seed-21 final | Mask stability |
|---|---:|---:|---:|---:|---:|
| Random partial | 0.7548 | 0.7528 | 0.6822 | 0.5543 | 0.524 |
| beta 0.15, stability 0.00 | 0.7621 | 0.7344 | 0.6719 | 0.5906 | 0.904 |
| beta 0.15, stability 0.05 | 0.7325 | 0.7179 | 0.6578 | 0.5018 | 0.948 |
| beta 0.50, stability 0.00 | 0.7264 | 0.7207 | 0.6666 | 0.4946 | 0.823 |
| beta 0.50, stability 0.05 | 0.7615 | 0.7275 | 0.6677 | 0.6377 | 0.844 |
| beta 1.00, stability 0.00 | 0.7434 | 0.7300 | 0.6713 | 0.5054 | 0.740 |
| beta 1.00, stability 0.05 | 0.7264 | 0.7163 | 0.6624 | 0.5634 | 0.774 |

All methods use 176.20 MB communication. Online importance measurement adds
approximately 2.7% to simulated total time.

## Decision

No online layer-level variant passes the screening gate:

- beta 0.15 without stability has a slightly higher final-round mean than
  random partial, but lower late-round accuracy and lower AUC;
- stronger exploration reduces mask stability as intended, but does not produce
  a consistent accuracy improvement;
- seed 21 remains unstable for every layer-level PCLA variant.

Layer-level score tuning is therefore stopped. The next method should change the
allocation unit from a complete LoRA layer to aligned rank components and use
alternating A/B training to reduce aggregation discordance. A random component
baseline must use the same component, layer, communication, and factor-training
budgets.
