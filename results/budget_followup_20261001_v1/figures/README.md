# Figures from public original-value exports

These figures use matplotlib and NumPy only. Each figure is available as a vector PDF with embedded TrueType fonts and a 300-DPI PNG for browsing. Scientific method names replace internal recipe and slot identifiers. The script reads the accepted exports and regenerates the figures.

## Rebuild

From any working directory:

```sh
python3 /path/to/checkout/results/budget_followup_20261001_v1/figures/gen_fig_budget_followup.py
```

The script resolves the data directory relative to itself, checks sample counts and complete seed–round coverage, computes statistics directly from original-value exports, and writes all six images plus `figure_numeric_audit.json`. Requirements: Python 3, NumPy, matplotlib. The audit includes source-file SHA-256 hashes and key numeric checks. The same checkout and inputs reproduce the numerical figures; binary PDF metadata can differ across executions.

## Figure 1: fixed communication cap

`fig_1_fixed_budget.{pdf,png}` compares returned-model accuracy and necessary execution time for the 16 budget-selection methods and nine server-feedback methods.

Inputs:

- `../data/1_budget_selection/validation/v_endpoints.csv`: `b8_accuracy`, grouped by slot; 12 seeds per method.
- `../data/1_budget_selection/costs/cost_endpoints.csv`: original `required_seconds`; one separate exclusive RTX 4090D run, seed 401, for each of the 15 non-replay methods.
- `../data/1_budget_selection/costs/quality_cost_join.csv`: only the slot-to-cost-job association is used; plotted quality and time are recomputed from the original endpoint files.
- `../data/2_server_feedback/validation/endpoints.csv`: `raw_B8_accuracy`; 12 seeds per method.
- `../data/2_server_feedback/validation/required_costs.csv`: `required_prefix_seconds`, restricted to the prespecified timing seeds 601, 607, 613; three exclusive RTX 4090D runs per method.

Accuracy is a percentage; its error bars show the mean ± 95% Student t interval across seeds (n = 12, df = 11). Time is converted from seconds to minutes. Study 2 time error bars use n = 3, df = 2. Study 1 single-seed time has no uncertainty bar. The quota-matched random replay has no separate cost measurement and is explicitly labeled “Not measured”; its time field remains missing.

The communication cap is 2,382,653,750 bytes. The returned model is the last complete global model within that cap, with a maximum of 200 rounds; a method need not consume the full cap. Study 1 quality combines RTX 4090D and A100 quality runs. Study 1 timing is measured separately on exclusive RTX 4090D, seed 401, and is not paired to the quality seeds. Cross-study quality levels and timing estimates are descriptive comparisons of separate protocols, not paired differences.

Necessary execution time includes the computation and protocol handling required to obtain the returned model. Research-only evaluation, parameter moving averages, and saving are excluded, as defined in `../METHODS.md`.

## Figure 2: server-feedback trajectories

`fig_2_server_feedback_curves.{pdf,png}` uses `../data/2_server_feedback/validation/curves.csv.gz`, fields `raw_accuracy` and `raw_NLL`, for rounds 1–100. All nine methods appear in the top row; the lower row expands rounds 60–100 for Full rank8, Server head fitting, Scalar controller (4 levels), and Two-block controller (16 pairs). Both accuracy and NLL are shown.

There are exactly 12 distinct seeds per method at every plotted round: 9 × 12 × 100 = 10,800 records. Lines show seed means; shading is a pointwise 95% t interval of the seed mean, not seed SD, an interval on individual predictions, or a simultaneous confidence band. Accuracy is percent. NLL is the mean negative log-likelihood in nats per example. These are raw models; EMA trajectories are not used. Late panels use explicitly restricted axes to show late behavior.

## Figure 3: controller switch-off

`fig_3_controller_switch_off.{pdf,png}` uses `../data/3_controller_switch_off/paired_values.jsonl`. The two parent controllers are analyzed separately: 12 matched seeds each, 24 continuation branches total. Differences are “switch off − keep controller” within seed, using the accepted original controller trajectory as the control. Positive accuracy differences indicate higher accuracy after switch-off; positive NLL differences indicate higher loss.

**Schema limitation:** the public paired export contains only checkpoints 61, 65, 70, 80, not all 20 individual rounds. Only those checkpoint values are plotted; connecting segments guide the eye and do not assert measured intermediate values. Thin lines show all seeds, and blue lines/bands show the paired mean/95% t interval. Seeds 613 and 647 are highlighted because they have the two largest negative accuracy differences at round 80 in both controller groups. The right panels show every round-80 paired accuracy value and the paired mean with its t95 interval.

Both controllers improve accuracy for 7/12 seeds at round 80 while having negative mean differences. Scalar: −1.635742 pp, t95 [−4.834045, +1.562560]. Two-block: −1.977539 pp, t95 [−5.460103, +1.505024]. Round-80 mean NLL differences are +0.085667 and +0.107900 nats per example, respectively. These are descriptive paired results; the tested switch and horizon define their interpretation.

## Statistical definition

For values x across n independent seeds, sample SD uses denominator n − 1, and t95 = mean(x) ± t(0.975, n − 1) × SD(x) / sqrt(n). Paired intervals apply this formula to within-seed differences. The script uses t critical values 2.200985160082949 (df 11) and 4.302652729696142 (df 2). Rounds, clients, and grid candidates are not treated as independent samples.
