# v2 → v3 correction record

## Scientific corrections

### 1. Main scale control

v2 scaled the fixed-denominator LoRA parameter increment to the contributor
increment's A/B parameter-vector norm. That control is coordinate-dependent:
reciprocal rescaling of LoRA factors can alter the parameter norm while leaving
the represented function unchanged.

v3 therefore defines the primary control using the effective matrix increment:

- B round: `ΔW = scaling · ΔB · A_global`;
- A round: `ΔW = scaling · B_global · ΔA`.

The global scalar applied to the fixed increment is

`gain_function = ||ΔW_C||_F / ||ΔW_F||_F`, summed over all LoRA modules.

The parameter-norm control is retained only as an auxiliary fixed-state readout.
The classifier is identical under F, C and both scale controls.

### 2. Numerical environment

v2 enabled deterministic algorithms and disabled TF32 inside the wrapper. That
would change the numerical protocol while reusing old F runs. v3 does not alter
these flags. It records them and requires exact full-run parity before old F
controls are admitted to analysis.

### 3. Meaning of the aggregation intervention

Contributor normalization in this batch is a mechanism transplanted into the
repaired active-but-frozen protocol. It is not a reconstruction of the historical
component-v1 experiment and cannot identify the cause of the old +8.03 pp result.

### 4. Staged evidence

The full 52-job matrix is not opened at initialization.

- Phase 1: two exact F parity runs + seed7 anchors 5/15.
- Phase 2: remaining fixed-state diagnostics.
- Phase 3: seeds7/13 trajectory pilot.
- Phase 4: seeds21/42/87 only after a recorded scientific decision.

The gates do not require PCLA to win. They test implementation validity, whether
an aggregation effect exists, and whether it is largely explained by effective
matrix-update magnitude.

## Engineering corrections

- queue uses a frozen site record and hash;
- jobs use attempt directories and explicit claim/lease records;
- recovery requires a recorded external interruption and stopped process;
- blocked jobs are materialized explicitly;
- archives extract through a temporary directory and atomic rename;
- new metrics are recomputed from logits;
- old F accuracy/recall are recomputed from predictions and source labels, while
  old NLL remains the saved GPU loss;
- final checkpoint, trace, selected-client sequence and aggregation-record count
  are audited;
- coverage, contributor label/capacity composition, matrix-update norm ratio,
  cosine and normalized error are included in fixed-state analysis.

## Remaining boundary

The discovery utilities are executable numerical components, not a completed
Qwen training stack. A100 time may be used for compatibility preflight only until
the model-level interface and its tests exist.
