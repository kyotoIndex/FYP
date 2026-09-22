# Resource plan: 12×RTX 4090 + 2×A100

This file is a scheduling plan, not a live resource assertion. Refresh Slurm,
UUID, GPU processes, CPU quota, RAM, node-local scratch and shared-filesystem
state immediately before launch.

## Principles

- Quality comparisons stay on one GPU family: RTX 4090.
- Each training job uses one independent process and one GPU.
- Same-host jobs share CPU, PCIe, RAM and I/O; record host blocks.
- A100 work is isolated from PCLA aggregation quality and timing comparisons.
- Readiness, not occupancy, determines whether a GPU receives work.
- Raw output goes to node-local scratch, then one verified archive per job.

## Phase allocation

| Stage | 4090 use | A100 use |
|---|---:|---:|
| Static tests and queue freeze | 0 | 0 |
| Phase 1 | 4 | 0–2 discovery compatibility preflights |
| Phase 2 | up to 8 | discovery model-interface/parity only |
| Phase 3 | up to 12 | bounded preflight/source banks only after backend passes |
| Phase 4 | 12 | independent discovery jobs only if separately authorized |

Unused GPUs at a scientific gate are intentionally idle. Do not add protocol
cells to create visual utilization.

## CPU and storage

Default per 4090 worker: four physical CPU cores, site permitting. Do not enable
two training processes per GPU without a separate throughput experiment and a
new timing identity.

The queue root stores small job records and final archives. Each worker writes
checkpoints, logits and local updates to its node-local scratch. Recovery from a
node-local checkpoint is supported only on the same host while the scratch tree
still exists. Otherwise preserve the interrupted attempt and restart the same
protocol from initialization.

## Timing

Record:

- GPU UUID and family;
- host and Slurm allocation/step;
- CPU affinity and requested cores;
- concurrent jobs on the same host;
- process wall time, client training, evaluation and aggregation instrumentation;
- cold-start/import time when included;
- every execution attempt.

Fixed-support replay omits online sensitivity scoring, so its wall time is not a
method-speed comparison against online PCLA.
