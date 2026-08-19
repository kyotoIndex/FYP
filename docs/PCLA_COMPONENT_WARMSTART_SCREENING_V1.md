# PCLA 1.2 Component Warm-Start Screening

## Question

PCLA 1.1 showed that importance-guided component selection was better than a
matched random component baseline, but both methods remained at near-chance
accuracy for seed 21. This experiment tests whether a short shared warm start
can remove that cold-start failure.

Configuration:

```text
configs/pcla_component_warmstart_screening.json
```

Formal results:

```text
results/pcla_component_warmstart_screening_v1
```

The protocol uses five clients, 12 rounds, seeds `7, 13, 21`, SST-2/BERT-base,
and Dirichlet label skew with `alpha = 0.3`.

## Schedule and matched control

Both `random_component_warmstart` and `pcla_component_warmstart` use exactly the
same training and communication schedule:

- rounds 1-2: each client trains full-rank LoRA A and B on its resource-sized
  deterministic bottleneck layer mask;
- rounds 3-12: each client trains eight components per active-layer-equivalent,
  alternating B and A;
- only the sparse component selection policy differs between the two methods.

During warm start, the low-, medium-, and high-capability clients train 96, 144,
and 192 rank components respectively. During sparse training those budgets are
48, 72, and 96. The paired methods have identical per-round trainable parameter
counts and communication.

## Aggregate results

| Method | Final mean | Last-5 mean | Round-average accuracy | Time | Communication |
|---|---:|---:|---:|---:|---:|
| FedHeLLo | 0.7983 | 0.7476 | 0.6805 | 74.88 s | 176.20 MB |
| Random partial layer | 0.7548 | 0.7528 | 0.6822 | 72.84 s | 176.20 MB |
| Random component warm-start | 0.7095 | 0.6691 | 0.6157 | 37.37 s | 66.52 MB |
| PCLA component warm-start | 0.7228 | 0.7169 | 0.6617 | 39.74 s | 66.52 MB |

Against its exact matched random control, PCLA 1.2 improves:

- final accuracy by 1.33 percentage points;
- last-five-round accuracy by 4.78 percentage points;
- round-average accuracy by 4.59 percentage points.

Against random partial-layer training, PCLA 1.2 reduces communication by 62.25%
and simulated time by 45.44%, but its final accuracy is 3.20 percentage points
lower.

## Per-seed results

| Seed | Random warm-start final | PCLA 1.2 final | PCLA 1.2 last-5 | PCLA 1.2 round average |
|---:|---:|---:|---:|---:|
| 7 | 0.8043 | 0.8152 | 0.7942 | 0.7221 |
| 13 | 0.8315 | 0.8605 | 0.8638 | 0.7702 |
| 21 | 0.4928 | 0.4928 | 0.4928 | 0.4928 |

The fixed two-round warm start does not solve the targeted failure. For seed 21,
both warm-start rounds remain at 0.4928, and neither sparse component method
subsequently escapes chance accuracy. The same seed's random partial-layer
baseline only begins moving after round 7, which is evidence that two fixed
warm-start rounds are insufficient for this partition. It does not establish
that a longer fixed warm start is the best solution.

## PCLA 1.2 versus PCLA 1.1

| Metric | PCLA 1.1 | PCLA 1.2 | Change |
|---|---:|---:|---:|
| Final mean | 0.7192 | 0.7228 | +0.0036 |
| Last-5 mean | 0.7215 | 0.7169 | -0.0046 |
| Round-average accuracy | 0.6416 | 0.6617 | +0.0201 |
| Communication | 44.58 MB | 66.52 MB | +49.21% |

Warm start improves early and whole-trajectory learning on seeds 7 and 13, but
provides almost no final-accuracy gain, slightly lowers the late-round mean,
adds substantial communication, and leaves seed 21 unchanged.

## Decision

The specific PCLA 1.2 hypothesis is rejected:

> A fixed two-round bottleneck-layer warm start is not an effective solution to
> the observed cold-start failure.

This does not invalidate component-level PCLA. Importance-guided selection still
outperforms its exact random control at the same budget. However, PCLA 1.2
should not replace PCLA 1.1 as the final proposed method because its small final
gain does not justify the additional communication and it fails its primary
seed-21 objective.

The next test should use an adaptive transition criterion based on a training
signal available to the algorithm, such as aggregated local loss improvement
or a held-out server proxy loss. It must not use the final evaluation set to
choose the transition round. A maximum warm-start budget is also required so
the method cannot obtain accuracy by silently approaching full partial-layer
training.
