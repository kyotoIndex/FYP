# PCLA executed-source reproduction map

This is the path-redacted publication copy of the source actually used for PCLA V4.1/S1 and V5a–f. Its repository base is `kyotoIndex/pcla-experiments` `origin/main` at `e53078b638ee30a633bad98195c1e2d4476523f5`. `PUBLIC_SOURCE_MANIFEST.json` records each original execution SHA256 and this copy's SHA256; `baseline/` and `diffs/` provide old files and unified diffs. Redaction only changes machine-local paths, and the third-party upstream files described below are omitted. No training was rerun for this package.

V4.1 means the aggregation and LoRA mechanism batch after its documented technical repair. S1 is its separate IID central and FFA supplement. V5a/b/c are parallel branches from S1; V5d adds two matched rank replay controls; V5e records additional metrics by deterministic replay; V5f evaluates saved states without a training queue.

| Formal batch | Executed source and config | Frozen queue | Audit and launch entry |
|---|---|---|---|
| V4.1 aggregation and LoRA mechanisms, 81 jobs | `snapshots/v4_1/`: `run_aggregation.py`, `run_lora_controls.py`, `JOB_PLAN.json` | `queues/v4_1/FROZEN.json` | `audit_run.py`, `analyze.py`, `operations/v4_1/` |
| S1 IID central and FFA, 13 jobs | `snapshots/s1/`: `run_lora_controls.py`, `JOB_PLAN.json` | `queues/s1/FROZEN.json` | `audit_run.py`, `operations/s1/` |
| V5a central/SVD/long-run first batch, 73 jobs | `snapshots/v5a/`: `run_lora_controls.py`, `JOB_PLAN.json` | `queues/v5a/FROZEN.json` | `audit_run.py`, `operations/v5/launch_workers_v5a.py` |
| V5b AG_NEWS/L2/V4, 54 jobs | `snapshots/v5b/`: `run_runtime.py`, `JOB_PLAN.json` | `queues/v5b/FROZEN.json` | `audit_run.py`, `operations/v5/launch_workers_v5b.py` |
| V5c personalization/FedEx-LoRA reference, 70 jobs | `snapshots/v5c/`: `run_personal.py`, `fedex_ref.py`, `JOB_PLAN.json` | `queues/v5c/FROZEN.json` | `audit_run.py`, `operations/v5/launch_workers_v5c.py` |
| V5d matched central rank replay, 5 jobs | `snapshots/v5d_central/`: `run_lora_controls.py`, `JOB_PLAN.json` | `queues/v5d_central/FROZEN.json` | `audit_run.py`, `operations/v5/launch_workers_v5d.py` |
| V5d matched SVD rank replay, 5 jobs | `snapshots/v5d_svd/`: `run_runtime.py`, `JOB_PLAN.json` | `queues/v5d_svd/FROZEN.json` | `audit_run.py`, `operations/v5/launch_workers_v5d.py` |
| V5e instrumented L2/V4 replay, 12 jobs | `snapshots/v5e/`: `run_runtime.py`, `v5e_metrics.py`, `JOB_PLAN.json` | `queues/v5e/FROZEN.json` | `audit_run.py`, `operations/v5/analyze_v5e.py` |
| V5f saved-state logits readout, 25 evaluations | `operations/v5f/evaluate_l2_logits.py` | no new queue | `operations/v5f/summarize_l2_logits.py` |

Each snapshot includes `SOURCE_LOCK.json` and the pinned implementation/configuration under `reference_snapshot/`. All eight source locks match 11 files from the base commit. The eight frozen plans bind 129 source file hashes. `RUN_ARCHIVE_HASHES.json` gives receipt-recorded SHA256 values for 313 completed archives; the large archives themselves are not included or independently rehashed here. `local_analysis/` holds the local result-audit scripts.

`diffs/` compares each published file to the base commit or `/dev/null` for new files. Frozen plans are generated execution records and have no repository predecessor. Their `FROZEN_SHA256.json` records the original executed plan's hash; the redacted plan's hash is in `PUBLIC_SOURCE_MANIFEST.json`. `lineage_diffs/` shows source changes along V4.1→S1, S1→V5a/b/c, V5a→V5d central, and V5b→V5d SVD/V5e; the V5c patch omits the external upstream source referenced below. Run `python3 verify_public.py` to verify file hashes, diffs, source locks, queue bindings and archive-index shape; use `SOURCE_SHA256SUMS.txt` for the entire export.

Machine-local paths were replaced by `$FYP_ROOT`, `$LOCAL_HOME`, `$HPC_HOME`, and `<hpc_account>`. The manifest maps each redacted copy to the actual execution SHA256. FedEx-LoRA upstream files are omitted because the collected upstream subtree had no `LICENSE` file; obtain [RaghavSinghal10/fedex-lora](https://github.com/RaghavSinghal10/fedex-lora) at `2fa2e4a243f93c829e2b601e459e45aa748e4149`. `external_sources` in the manifest lists the exact four upstream files bound by the V5c queue and their executed SHA256. The project's own `fedex_ref.py` adapter and its unified diff are included.
