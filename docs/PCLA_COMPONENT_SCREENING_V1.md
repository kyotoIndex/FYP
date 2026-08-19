# PCLA Component Screening v1

## Scope

This completed screening evaluates whether PCLA becomes more effective when the
allocation unit is changed from a complete LoRA layer to aligned LoRA rank
components.

Configuration:

```text
configs/pcla_component_screening.json
```

Results:

```text
results/pcla_component_screening_v1
```

The experiment uses five clients, 12 federated rounds, three seeds
(`7, 13, 21`), SST-2/BERT-base, and Dirichlet label skew with `alpha = 0.3`.
Odd rounds train the selected `B` components and even rounds train the selected
`A` components. The frozen factor is assumed to be cached by each client.

## Budget matching

`random_component_alternating` and `pcla_component_alternating` have identical
budgets in every client-round:

| Layer capacity | Active components | Trainable parameters | Communication per client-round |
|---:|---:|---:|---:|
| 6 | 48 | 75,266 | 0.5742 MB |
| 9 | 72 | 112,130 | 0.8555 MB |
| 12 | 96 | 148,994 | 1.1367 MB |

Both methods select eight components per active-layer-equivalent of capacity.
Aggregation is component-wise and includes only clients that contributed the
corresponding component in the currently trained factor.

## Results

| Method | Final mean | Last-5 mean | Round-average accuracy | Time | Communication |
|---|---:|---:|---:|---:|---:|
| FedHeLLo | 0.7983 | 0.7476 | 0.6805 | 73.88 s | 176.20 MB |
| Random partial layer | 0.7548 | 0.7528 | 0.6822 | 70.78 s | 176.20 MB |
| Random component alternating | 0.6890 | 0.6848 | 0.6137 | 27.39 s | 44.58 MB |
| PCLA component alternating | 0.7192 | 0.7215 | 0.6416 | 30.28 s | 44.58 MB |

At the matched component budget, PCLA improves over random component selection
by:

- 3.02 percentage points in final-round accuracy;
- 3.67 percentage points in last-five-round accuracy;
- 2.79 percentage points in round-average accuracy.

Relative to random partial-layer training, PCLA component alternating reduces
communication by 74.70% and simulated time by 57.22%, while its final accuracy
is 3.56 percentage points lower. Importance measurement accounts for 2.77 s,
or 9.14%, of its simulated total time.

## Per-seed check

| Seed | Random component final | PCLA component final | Final delta | Last-5 delta | Round-average delta |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.7518 | 0.7899 | +0.0380 | +0.0319 | +0.0373 |
| 13 | 0.8225 | 0.8750 | +0.0525 | +0.0783 | +0.0463 |
| 21 | 0.4928 | 0.4928 | 0.0000 | 0.0000 | 0.0000 |

The PCLA component result is better on all three curve summaries for seeds 7
and 13 and tied for seed 21. Seed 21 remains at near-chance accuracy throughout
the run for both component methods. Consequently, the mean gain must not be
presented as robustness to difficult non-IID partitions.

## Selection stability

The mean same-factor, two-round-lag Jaccard similarity is:

| Method | Component Jaccard | Layer Jaccard |
|---|---:|---:|
| Random component alternating | 0.2044 | 0.5318 |
| PCLA component alternating | 0.6842 | 0.7910 |

This confirms that the online PCLA scores create a substantially more stable
component allocation than random sampling. It does not by itself establish
that higher stability causes the accuracy gain.

## Decision

The component-level mechanism passes the internal matched-random screening
gate: it improves final, late-round, and whole-trajectory accuracy without
using additional trainable parameters or communication.

It does not yet pass a full baseline gate:

- mean accuracy remains below random partial-layer training and FedHeLLo;
- only three seeds are available and the standard deviation is large;
- seed 21 fails for both component methods;
- the communication saving relies on the explicit client-side caching
  assumption for the frozen LoRA factor.

The defensible current conclusion is therefore:

> Component-level marginal-utility allocation is a promising efficiency
> mechanism under a fixed sparse LoRA budget, but the present implementation is
> not yet a generally superior federated fine-tuning algorithm.

The next experiment should target the observed failure rather than continue
layer-score tuning: compare component budgets and add a short dense or
layer-level warm start before sparse alternating training. The same-seed random
component control and exact budget audit must be retained.
