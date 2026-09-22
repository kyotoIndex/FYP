# Independent discovery probes

These probes are independent of the PCLA aggregation causal study. The supplied
`discovery_math.py` contains numerical operations only; it is not a completed
Qwen training or evaluation backend.

## Track A — fixed-size functional adapter compression

Question: after strong parameter and activation-aware compression at the same
final rank, does a meaningful, recoverable domain-level task loss remain?

Design:

- four natural review domains: books, dvd, electronics, kitchen;
- exploration seeds 17 and 29;
- exact weighted effective-matrix target from four rank-16 adapters;
- parameter compression and pooled activation-weighted compression at ranks
  4, 8 and 16;
- rank 8 is primary;
- a bounded group-excess reference estimates remaining same-rank recoverability.

Use calibration activations only to choose compression. Evaluate per-domain
holdout NLL/accuracy, macro mean, worst-domain change, payload bytes and extra
calibration/statistics bytes.

Stop when ordinary SVD has no practical task loss, activation weighting repairs
the important loss, or the group reference improves matrix error without task
quality.

## Track B — cross-precision adapter transfer

Question: after quantization-aware initialization and correct effective product
increment transfer, does a stable cross-precision loss remain?

Candidate numerical bases:

- BF16;
- NF4;
- INT8;
- fixed-quantized-base one-step LoftQ-NF4, only if the installed interface and
  local assets support it.

For each source/destination pair compare:

1. raw source final adapter;
2. destination initialization plus source final effective matrix minus source
   initialization effective matrix;
3. rank-16 parameter-compressed version of 2;
4. rank-16 activation-weighted version of 2.

The task increment is `B_final A_final - B_initial A_initial`, including each
adapter's alpha/r scaling. It is not `(B_final-B_initial)(A_final-A_initial)`.

Stop when ordinary transfer is unaffected, existing initialization repairs the
main loss, or rank-matched activation compression repairs the remaining loss.

## Required backend checks before A100 science jobs

- exact local model/tokenizer file hashes and resolved upstream revision where
  available;
- fixed source IDs and disjoint train/calibration/selection/holdout roles;
- label scoring interface verified, including multi-token labels if applicable;
- original PEFT versus factor-replacement forward parity;
- cloned, non-aliased initial adapters;
- frozen base parameters and FP32 adapter arithmetic;
- BF16/NF4/INT8/LoftQ compatibility and peak memory on the hardware used;
- no holdout access during compatibility preflight.

A100 availability does not authorize running an incomplete backend. A failed
probe does not block unrelated, valid aggregation jobs.
