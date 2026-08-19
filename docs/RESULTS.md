# Experiment Registry And Current Findings

## Canonical Main Run

Directory: `results/expanded_sst2_bert_base_dirichlet_v1`

Protocol:

- SST-2 with `bert-base-uncased`
- 20 clients; 10 uniformly sampled without replacement per round
- 30 rounds; one local epoch; 512 samples per client
- seeds 7, 13, and 21
- Dirichlet label skew, alpha 0.3
- 12 low-, 6 medium-, and 2 high-resource clients
- LoRA rank 16 on query/value projections
- 320 proxy and 552 disjoint evaluation examples
- nine methods in one runner under shared initialization/data/order policies

Audit status:

- 810 metric rows (9 methods x 30 rounds x 3 seeds)
- 8,100 allocation rows (10 selected clients per metric row)
- all allocation invariants pass
- 12 allocation/scheduling unit tests pass

Key final-round results:

| Method | Accuracy mean +/- SD | Mean time (s) | Mean communication (MB) |
|---|---:|---:|---:|
| Fed-HeLLo | 0.8979 +/- 0.0058 | 190.4 | 834.1 |
| FedRA Random Partial | 0.8973 +/- 0.0121 | 185.1 | 834.1 |
| FIM-HLA | 0.8961 +/- 0.0118 | 192.5 | 834.1 |
| RGD-HLA | 0.8943 +/- 0.0154 | 189.0 | 834.1 |
| PCLA-v1 | 0.8895 +/- 0.0126 | 188.3 | 878.7 |
| FedLoRA | 0.8859 +/- 0.0189 | 313.7 | 1353.5 |
| Resource-only | 0.8653 +/- 0.0420 | 179.8 | 834.1 |
| Importance-resource | 0.7621 +/- 0.2211 | 193.5 | 920.2 |
| LEGEND-like | 0.7512 +/- 0.2126 | 228.7 | 1100.2 |

Interpretation:

- PCLA-v1 is more efficient than full FedLoRA and slightly improves its final
  accuracy, but it does not outperform Fed-HeLLo or random partial allocation.
- Fed-HeLLo and random partial allocation differ by only 0.0006 in final mean;
  this three-seed run does not establish statistical superiority.
- PCLA-v1's requested four clusters collapse to two used IDs and mostly separate
  low-resource from medium/high-resource clients. The current clustering rule
  does not demonstrate data-driven personalization.
- Importance-resource and LEGEND-like each suffer a near-random seed, producing
  very large variance. Their mean final accuracy must not be reported without
  the error bars and per-seed context.
- The reported time is a heterogeneous compute/communication simulation based
  on measured local operations and configured device factors. It is not a
  physical multi-device wall-clock benchmark.

## Other Result Directories

| Directory | Role | Status |
|---|---|---|
| `formal_comparison_sst2_bert_base_v1` | Earlier 5-client, 12-round comparison | Complete; secondary evidence |
| `controlled_sst2_bert_base_v1` | Audited Fed-HeLLo mechanism run | Complete; mechanism check |
| `controlled_sst2_bert_base` | Earlier controlled output | Complete but superseded by `_v1` |
| `controlled_sst2` | BERT-mini target directory | Empty; checkpoint incompatible with installed safety policy |

Completed run-level `run_config.json` files retain their historical paths. They
are immutable provenance records from before the workspace reorganization. The
current configs use the self-contained `data/` and `results/` paths.

## Next Algorithm Direction

Implement PCLA-v2 as a new method and protocol, not an in-place rewrite of v1:

1. Cluster using layer-wise marginal utility or update features; do not mix raw
   resource tier directly into the clustering feature.
2. Put compute and communication into explicit budget constraints.
3. Jointly allocate active layers and rank.
4. Define rank-aware aggregation, ideally with a global shared adapter plus
   cluster residual adapters.
5. Add EMA/periodic importance updates and allocation-stability regularization.
6. Run `K={1,2,3,4}` and Dirichlet-alpha ablations with the same random baseline.
7. Report final accuracy, late-round mean, time-to-target, communication, and
   Pareto plots.

