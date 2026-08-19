# Corrected Federated Label-Skew Protocol v1

## Problem found

The earlier `dirichlet_label_skew` implementation independently sampled a
positive-label ratio for every client and then constructed a fixed-size client
dataset from that ratio. It did not constrain the aggregate client population
to preserve the source dataset's label distribution.

This produced the following aggregate training partitions:

| Seed | Positive samples | Total samples | Positive ratio |
|---:|---:|---:|---:|
| 7 | 713 | 2,560 | 27.85% |
| 13 | 1,503 | 2,560 | 58.71% |
| 21 | 96 | 2,560 | 3.75% |

The source SST-2 training split is 55.78% positive. Seed 21 therefore tested
methods on an almost all-negative federated training population while using a
near-balanced evaluation set. A classifier that always predicted the negative
class obtained 0.4928 accuracy, exactly matching the failed component runs.

## Correction

The new `constrained_dirichlet_label_skew` mode preserves:

- 512 samples per client;
- 2,560 samples in the federated population;
- exactly 1,428 positive samples, or 55.78%, for every seed;
- seeded client-level label heterogeneity;
- sampling without replacement from the source training pool.

It draws heterogeneous client label odds and applies a common log-odds shift so
that the fixed client-size and global class-count margins are both satisfied.
Integer counts are assigned deterministically by largest remainder.

The resulting positive counts are:

| Seed | Client positive counts | Aggregate positive ratio |
|---:|---|---:|
| 7 | 1, 512, 390, 359, 166 | 55.78% |
| 13 | 61, 466, 361, 476, 64 | 55.78% |
| 21 | 0, 409, 498, 502, 19 | 55.78% |

The clients remain strongly non-IID, but their union no longer changes the
learning task's global label prior.

Configuration:

```text
configs/pcla_component_constrained_dirichlet.json
```

Formal results:

```text
results/pcla_component_constrained_dirichlet_v1
```

## Corrected results

| Method | Final mean | Final std | Last-5 mean | Round-average accuracy | Time | Communication |
|---|---:|---:|---:|---:|---:|---:|
| FedHeLLo | 0.8418 | 0.0499 | 0.8091 | 0.7478 | 78.17 s | 176.20 MB |
| Random partial layer | 0.8321 | 0.0763 | 0.8430 | 0.7677 | 72.37 s | 176.20 MB |
| Random component alternating | 0.7736 | 0.0847 | 0.7691 | 0.6976 | 31.01 s | 44.58 MB |
| PCLA component alternating | 0.8231 | 0.0871 | 0.8336 | 0.7459 | 32.79 s | 44.58 MB |

At exactly the same sparse component and communication budget, PCLA exceeds
random component selection by:

- 4.95 percentage points in final accuracy;
- 6.45 percentage points in last-five-round accuracy;
- 4.84 percentage points in round-average accuracy.

Relative to random partial-layer training, PCLA:

- is 0.91 percentage points lower in final accuracy;
- is 0.94 percentage points lower in last-five-round accuracy;
- is 2.17 percentage points lower in round-average accuracy;
- reduces communication by 74.70%;
- reduces simulated time by 54.70%.

Relative to FedHeLLo, PCLA is 1.87 percentage points lower in final accuracy,
2.45 percentage points higher in last-five-round accuracy, and 0.19 percentage
points lower in round-average accuracy, while using 74.70% less communication.

## Seed-21 confirmation

| Method | Old seed-21 final | Corrected seed-21 final | Change |
|---|---:|---:|---:|
| Random partial layer | 0.5543 | 0.7446 | +0.1902 |
| FedHeLLo | 0.6902 | 0.7844 | +0.0942 |
| Random component alternating | 0.4928 | 0.6757 | +0.1830 |
| PCLA component alternating | 0.4928 | 0.7228 | +0.2301 |

All methods now reach the 0.7 target in all three seeds. This confirms that the
previous constant 0.4928 behavior was primarily caused by aggregate label-prior
collapse in the partition protocol.

## Reporting decision

The old independent-ratio component experiments remain useful for diagnosing
failure under aggregate label-prior shift. They must not be mixed into the
corrected main comparison or presented as the formal PCLA accuracy result.

The defensible corrected conclusion is:

> Under fixed-size, globally label-preserving heterogeneous client partitions,
> importance-guided component allocation recovers most of the accuracy of
> partial-layer baselines while reducing communication by approximately 75%,
> and consistently outperforms a matched random component policy.

This is still a three-seed controlled SST-2 mechanism study, not evidence of
general superiority across datasets, models, client counts, or heterogeneity
levels.
