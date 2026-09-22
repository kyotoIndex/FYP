# Codex start here

Use branch:

```text
handoff/pcla-aggregation-20260923-v3
```

Base project commit under study:

```text
e53078b638ee30a633bad98195c1e2d4476523f5
```

## Task

Execute the corrected aggregation-ablation handoff without changing `main`, old
results, the paper, Issue #1 or PR #2.

Read, in order:

1. `REVIEW_CHANGES_V2_TO_V3.md`
2. `SCIENTIFIC_PROTOCOL.md`
3. `CODEX_NEXT.md`
4. `RESOURCE_PLAN.md`
5. `DISCOVERY_PROTOCOL.md`

Then resolve a private `site.json` from `site.example.json` and run the fresh
CPU/synthetic tests. Do not quote any test count from chat history; only the new
`cpu_tests.json` is execution evidence.

## Hard rules

- Do not start all 52 jobs.
- Phase 1 has four jobs only.
- Phase 2 opens only after exact parity and seed-7 smoke.
- Phase 3 and Phase 4 require explicit reviewed gate records.
- Aggregation quality runs use RTX 4090 only.
- The A100s may run bounded discovery compatibility checks only after the model
  backend exists and passes its parity tests.
- Any process failure creates a queue abort. Do not silently retry.
- A low accuracy or an unfavorable PCLA result is not an execution failure.
- Raw files stay on node-local scratch; only verified archives go to shared
  storage.
- Stop after the approved batch and return `REPORT_TO_PRO.md`.

## Status

The branch contains source and documentation only. It has not been executed by
the conversation runtime. Syntax, numerical, BERT/CUDA and scientific results
must be generated in the actual environment.
