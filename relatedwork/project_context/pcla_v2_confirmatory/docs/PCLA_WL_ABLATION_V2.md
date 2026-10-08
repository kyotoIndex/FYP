# PCLA-WL mainline ablation (v2)

Status: pre-registered before any run on seeds 601, 631, 661, 691, or 721.
Supersedes the incomplete `pcla_wl_ablation_v1`, which omitted pipeline controls.

This ablation is for the current FYP mainline: `layer_only` as PCLA-WL with
layer EMA. It has two blocks: ingredient ablations, and same-simulator
pipeline controls for FedLoRA and Fed-HeLLo.

## Ingredient ablations

All sparse methods share heterogeneous 3/4/6-layer (48/64/96-credit) budget,
alternating A/B, active-frozen forward, and fixed-denominator aggregation.

| Policy | Role |
|---|---|
| `layer_only` | Full PCLA-WL (reference) |
| `pcla` | Remove whole-layer structure |
| `random_layer` | Remove client-local importance |
| `deep_layer` | Replace the score with deepest layers; no probe |

## Pipeline controls

These are **not** budget-matched ingredient removals. They answer “where does
PCLA-WL sit next to the two pipelines it is usually compared with?”

| Policy | Pipeline analogue | What is matched | What is not |
|---|---|---|---|
| `full_alt` | FedLoRA | BERT-base/SST-2, 30 rounds, A/B alternating, head training, fixed-denominator aggregation, official 872-example dev | Not a paper reproduction. Uses all 12 LoRA layers every round, so traffic and time are larger on purpose. |
| `fim_hla` | Fed-HeLLo layer allocation | Same simulator and whole-layer budget as PCLA-WL; server-proxy FIM; resource-clustered probabilities; sample without replacement | Not a numerical reproduction of Fed-HeLLo. No 50-round RGD warm start, no contributor-only layer aggregation. |

Do not put `pcla_experiments` v1 FedLoRA / Fed-HeLLo numbers in this table.

## Matrix

Five unused seeds × six policies (30 jobs). Endpoint: mean benchmark-dev
accuracy over rounds 26--30. Report `train_seconds` and `probe_seconds`
separately.

Seeds: 601, 631, 661, 691, 721.

## Decision

Descriptive. Does not replace PCLA-WL as the mainline.

- Efficiency versus component PCLA: ≥25% mean local train time cut, mean
  accuracy within 1 pp.
- Score versus `random_layer` / `deep_layer`: report; directional versus
  random if paired mean > 0 and 4/5 seeds agree.
- Versus `fim_hla`: Fed-HeLLo-style control; PCLA-WL may lose.
- Versus `full_alt`: FedLoRA-style quality/cost reference; a lower accuracy
  with a large time cut is the intended trade-off, not a failed ablation.
