# Aggregation causal study

Protocol: `aggregation-causal-20260923-v3`

## 1. Questions

1. With the same parent state, participants, data order and local updates, how do
   denominator choices change the global update and immediate model behaviour?
2. Is a contributor-normalized effect mostly explained by a larger effective
   LoRA matrix update, or by coordinate-wise coverage reweighting?
3. When the selector is allowed to react to the changed model trajectory, does
   choosing specific rank IDs acquire value relative to its matched random-rank
   control?

The protocol does not require PCLA or contributor normalization to win.

## 2. Aggregation rules

For participant weight `N`, coordinate coverage `N_j` and weighted numerator
`S_j = Σ_k n_k z_kj Δ_kj`:

- **F — fixed denominator:** `d_F,j = S_j / N`.
- **C — contributor denominator:** `d_C,j = S_j / N_j` for covered coordinates;
  zero increment when `N_j=0`.
- **G_W — function-norm matched fixed update:** keep the complete F LoRA
  parameter direction and multiply it by one global scalar so its represented
  LoRA matrix-update norm equals C.
- **G_P — parameter-norm matched fixed update:** the v2 auxiliary control. It is
  recorded only in fixed-state diagnostics, not used for 30-round trajectories.

The classifier uses the same full-participant weighted update in every rule and
is never rescaled.

For an alternating B round:

`||d||_W² = Σ_modules || scaling · dB · A_global ||_F²`.

For an alternating A round:

`||d||_W² = Σ_modules || scaling · B_global · dA ||_F²`.

The batch rejects both-factor rounds because the first-order expression would
omit the `dB·dA` cross term. The current core protocol alternates one factor, so
this is an explicit validity check rather than a limitation hidden in analysis.

Neither F nor C is claimed to debias deterministic, data-dependent support
selection. C changes both coordinate weighting and effective update magnitude.
G_W isolates the latter only up to a single global matrix-update scale.

## 3. Phase 1 — technical parity and seed-7 smoke

Jobs:

- full fixed replay: seed7 / original PCLA support;
- full fixed replay: seed7 / original replay-random-rank support;
- fixed-state diagnostic: seed7 / anchor5;
- fixed-state diagnostic: seed7 / anchor15.

The two full replays must reproduce source final parameters, trace, predictions,
selected-client sequence and common metric fields. They also verify that the
live numerical flags match the source execution environment; the wrapper does
not change deterministic or TF32 settings.

The fixed-state jobs use train-derived tuning only. They must demonstrate:

- exact reuse of saved parent and support identity;
- F/C/G_W/G_P classifier increments are identical;
- G_W matches the C effective matrix norm within `1e-8` relative error;
- full-support F and C coincide;
- no nonfinite update, metric or coverage value.

This is a technical gate, not a method-quality gate.

## 4. Phase 2 — complete fixed-state intervention

Seeds: 7, 13, 21, 42, 87.
Anchors: saved PCLA rounds 5 and 15.
For each anchor, inspect its next two rounds, covering B and A updates.

Support treatments:

1. exact original PCLA masks;
2. exact original replay-random-rank masks;
3. all 192 components as a higher-budget full-support audit.

Each selected client's local update is computed once per support treatment and
reused under F, C, G_W and G_P. The full-support audit needs only F because all
rules coincide.

The second inspected round always starts from the first round's PCLA-support + F
state. Treatments never create separate parents inside this diagnostic.

Record for every support/round:

- logits, labels, accuracy, float64 NLL, recalls and balanced accuracy;
- every client increment and mask;
- per-position weighted coverage;
- per-position contributor label composition and capacity-tier composition;
- F/C parameter and effective-matrix norms;
- C versus F and G_W versus C cosine, norm ratio and normalized Frobenius error;
- sparse update versus full-support update cosine, norm ratio and normalized
  Frobenius error;
- client calls, optimizer steps and timing.

Size: ten jobs, 600 client calls, 9,600 optimizer steps.

### Phase-2 review

Phase 3 is not opened automatically. `gate.py phase2` writes a review report.
Codex returns it for scientific approval. The report precomputes, but does not
turn into a universal theorem, these project-level diagnostics:

- how often C differs from F beyond numerical tolerance;
- whether G_W explains at least 75% of the C−F immediate-quality movement;
- whether C−G_W is below 0.5 pp in accuracy and small in NLL;
- coverage and contributor-composition heterogeneity;
- consistency across seeds, anchors and A/B rounds.

A near-zero intervention stops the long trajectory matrix. A nonzero effect may
proceed even if it is harmful; the purpose is causal discrimination.

## 5. Phase 3 — two-seed trajectory pilot

Seeds: 7 and 13.
Horizon: 30 rounds.
Primary endpoint: each seed's mean benchmark-dev accuracy over rounds 26–30.

Fixed support sources:

- original PCLA trace;
- original replay-random-rank trace;
- original random-component trace.

New rules per source:

- C;
- G_W.

This produces 12 fixed-support trajectories.

Adaptive trajectories:

- online PCLA + C for each seed;
- replay-random-rank + C using only its new same-seed donor.

This adds four trajectories. Total Phase 3: 16.

Original F cells are reused only after Phase-1 parity. Fixed-support timing is
not compared with online PCLA timing as an aggregation speedup because online
scoring is absent from the replay.

### Phase-3 review

`gate.py phase3` produces all predeclared contrasts for seeds7/13. Expansion is
recommended only when the observed aggregation effect is not negligible, is
not fully explained by G_W, and changes a scientifically meaningful quality or
stability outcome. Two seeds are still development evidence.

## 6. Phase 4 — remaining seeds

Only after an explicit signed gate record, run seeds21/42/87:

- 18 fixed-support C/G_W trajectories;
- three online PCLA+C trajectories;
- three matched online replay+C trajectories.

The full conceptual table then contains the original 15 F cells and 30 new
fixed-support cells, plus ten adaptive C trajectories.

## 7. Analysis

Independent unit: seed. Rounds, clients, positions and host processes are not
extra independent replications.

Report all five seeds and the predeclared groups:

- 7/13: used in earlier LR development;
- 21/42/87: not used for that LR ranking.

Contrasts:

- C−F, G_W−F and C−G_W inside each support source;
- PCLA-support minus random-support for F/C/G_W;
- PCLA-support minus replay-support for F/C/G_W;
- online PCLA+C minus its own online replay+C;
- selection-by-aggregation interactions.

Retain final round, full 30-round curves, tail range, NLL, both recalls,
balanced accuracy and strict one-class predictions. Do not select a best round.

Small-sample t intervals are descriptive. Crossing zero is not equivalence.

## 8. Interpretation boundaries

- This batch studies the repaired active-but-frozen protocol. It cannot assign
  the historical component-v1 +8.03 pp to aggregation.
- If C and G_W move similarly, global effective update magnitude is a leading
  explanation, not a uniquely proven mediator.
- If C differs from G_W, coordinate coverage and contributor composition remain
  plausible mechanisms; do not automatically claim coverage bias is the root.
- If every selector improves but PCLA still fails its matched rank control,
  retain the aggregation finding without reviving rank-ID superiority.
- If online and fixed-support conclusions differ, selection feedback is part of
  the mechanism and must be reported explicitly.

No LR, horizon, seed, task, model or metric may be changed after results arrive.
