# PCLA Large-Scale Corrected Protocol v1

## Scope

This is the main large-scale controlled experiment for component-level PCLA.
It uses the corrected globally label-preserving heterogeneous client partition.

Configuration:

```text
configs/pcla_large_scale_constrained_dirichlet.json
```

Results:

```text
results/pcla_large_scale_constrained_dirichlet_v1
```

Protocol:

- SST-2 with BERT-base;
- 20 total clients and 10 uniformly selected clients per round;
- 512 local samples per client;
- 30 federated rounds;
- seeds `7, 13, 21`;
- constrained Dirichlet label skew with `alpha = 0.3`;
- 5,712 positive samples out of 10,240 for every seed;
- no source-record reuse during client partition construction.

The four methods are full FedLoRA, FedHeLLo, random component alternating, and
PCLA component alternating.

## Aggregate results

| Method | Final mean | Final std | Last-5 mean | Round-average accuracy | Time | Communication |
|---|---:|---:|---:|---:|---:|---:|
| FedLoRA | 0.8943 | 0.0038 | 0.8798 | 0.8137 | 314.00 s | 1,353.52 MB |
| FedHeLLo | 0.8841 | 0.0161 | 0.8667 | 0.8030 | 183.89 s | 834.15 MB |
| Random component alternating | 0.7899 | 0.0643 | 0.7370 | 0.7037 | 73.15 s | 211.18 MB |
| PCLA component alternating | 0.8702 | 0.0257 | 0.8365 | 0.7800 | 100.19 s | 211.18 MB |

All four methods reach the 0.7 target in all three seeds.

## Matched-random comparison

PCLA and random component alternating have exactly matched selected clients,
resource capacities, active-component counts, trainable parameter counts,
trained A/B factor, and communication in every corresponding client-round.

PCLA improves over this exact random control by:

- 8.03 percentage points in final accuracy;
- 9.95 percentage points in last-five-round accuracy;
- 7.62 percentage points in round-average accuracy.

The paired final gains are positive in every seed:

| Seed | Random component final | PCLA final | PCLA minus random |
|---:|---:|---:|---:|
| 7 | 0.8279 | 0.8750 | +0.0471 |
| 13 | 0.8261 | 0.8931 | +0.0670 |
| 21 | 0.7156 | 0.8424 | +0.1268 |

This is the strongest evidence in the current study that importance-guided
component allocation contributes value beyond sparse component training alone.

## Accuracy-efficiency comparison

Relative to FedHeLLo, PCLA:

- is 1.39 percentage points lower in final accuracy;
- is 3.02 percentage points lower in last-five-round accuracy;
- is 2.30 percentage points lower in round-average accuracy;
- reduces communication by 74.68%;
- reduces simulated time by 45.52%.

Relative to full FedLoRA, PCLA:

- is 2.42 percentage points lower in final accuracy;
- reduces communication by 84.40%;
- reduces simulated time by 68.09%.

The defensible result is an accuracy-efficiency trade-off. PCLA does not exceed
FedLoRA or FedHeLLo in accuracy, but it recovers most of their accuracy at a
substantially lower communication and time budget.

## Partial-participation importance audit

The large-scale setting requires client-specific importance to support clients
that first participate after round 1. The runner now:

- initializes importance when a client is first selected;
- refreshes it when that client's refresh interval expires;
- retains client-specific EMA and selection state between participations.

Every seed contains 300 selected-client-rounds for PCLA. All 20 clients receive
valid importance state. The numbers of refreshed client-rounds are 154, 157,
and 152 for seeds 7, 13, and 21 respectively.

Importance estimation adds 25.61 s on average, or 25.56% of PCLA's simulated
total time. Even with this overhead included, PCLA remains 45.52% faster than
FedHeLLo. Reducing this overhead is a concrete optimization target.

## Reporting decision

This experiment should replace the five-client, 12-round component screening as
the main PCLA component result. The smaller experiment remains a mechanism and
debugging study.

Recommended claim:

> In a 20-client, partial-participation SST-2 experiment with globally
> label-preserving Dirichlet heterogeneity, PCLA component allocation
> consistently outperforms an exactly budget-matched random component policy
> and approaches FedHeLLo and full FedLoRA accuracy while reducing
> communication by approximately 75% and 84%, respectively.

Required qualifications:

- this is a controlled FedHeLLo mechanism comparison, not an exact numerical
  reproduction of the original paper;
- time includes a simulation model for heterogeneous compute and communication;
- communication assumes the frozen LoRA factor is cached on the client;
- evidence is currently limited to SST-2/BERT-base, three seeds, and
  `alpha = 0.3`;
- the method should not be described as more accurate than FedLoRA or FedHeLLo.
