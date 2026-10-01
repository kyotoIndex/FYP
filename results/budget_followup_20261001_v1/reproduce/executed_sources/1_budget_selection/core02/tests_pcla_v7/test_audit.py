"""Independent fabricated evidence; no training loop is used by these tests."""
import json
import math
from pathlib import Path

import torch

from scripts.audit_pcla_v7_run import audit, canonical_hash, sha


def put(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def evidence(root: Path):
    source, run, data = root / "source", root / "run", root / "data"
    source.mkdir(); run.mkdir(); data.mkdir()
    (source / "x.py").write_text("version = 1\n")
    put(source / "SOURCE_MANIFEST.json", {"files": {"x.py": sha(source / "x.py")}})
    train = data / "SST-2" / "train.tsv"
    train.parent.mkdir()
    train.write_text("sentence\tlabel\n")
    ids = [f"train:{i}" for i in range(4096)]
    labels = [i % 2 for i in range(4096)]
    roles = {"protocol": "pcla-global-budget-20260928-v7",
             "source_sha256": sha(train), "roles": {"recipe_dev": ids},
             "row_labels": dict(zip(ids, labels))}
    roles["split_sha256"] = canonical_hash(roles)
    role_file = data / "train_roles.json"
    put(role_file, roles)
    cache = data / "train_tokens.npz"
    cache.write_bytes(b"dummy-token-cache")
    put(data / "train_tokens.json", {"cache_sha256": sha(cache),
        "identity": {"split_sha256": roles["split_sha256"]}})
    b8 = {"total_bytes": 2000, "definition": "test"}
    b8["wire_manifest_sha256"] = canonical_hash(b8)
    b8_file = data / "b8_wire.json"
    put(b8_file, b8)
    job = {"job_id": "D-test", "protocol": roles["protocol"],
           "eval_role": "recipe_dev", "method": "full_both", "rank": 4, "seed": 401}
    config = {"job": job, "runtime": {"byte_cap": 2000},
              "manifest": str(role_file), "b8_wire": str(b8_file),
              "train_file": str(train), "token_cache": str(cache)}
    config["expanded_sha256"] = canonical_hash(config)
    config_file = root / "config.json"
    put(config_file, config)
    provenance = {"config_sha256": sha(config_file),
                  "source_manifest_sha256": sha(source / "SOURCE_MANIFEST.json"),
                  "manifest_sha256": sha(role_file), "split_sha256": roles["split_sha256"],
                  "source_sha256": sha(train), "token_cache_sha256": sha(cache),
                  "b8_wire_manifest_sha256": sha(b8_file)}
    put(run / "run_identity.json", {"job": job, "provenance": provenance})
    put(run / "execution_attempts.json", [{"identity": "test-identity"}])
    put(run / "evaluation_labels.json", labels)
    logits = [[0., 1.]] * 4096
    accuracy = .5
    loss = .5 * (math.log1p(math.exp(1)) + math.log1p(math.exp(-1)))
    readout = {"accuracy": accuracy, "loss": loss, "samples": 4096}
    put(run / "evaluation_round_000.json", readout)
    for r in range(101):
        put(run / f"logits_round_{r:03d}.json", logits)
    modules = [f"bert.encoder.layer.{layer}.attention.self.{target}"
               for layer in range(12) for target in ("query", "value")]
    adapter = {f"{name}.lora_{factor}": torch.zeros(4, 1)
               for name in modules for factor in ("a", "b")}
    events, rows = [], []
    for r in range(1, 101):
        for cid in range(10):
            for direction in ("down", "up"):
                events.append({"event_id": f"r{r}:c{cid}:{direction}", "round": r,
                               "direction": "download" if direction == "down" else "upload",
                               "bucket": "deployed_prefix", "bytes": 1})
        support = {name: {"a": list(range(4)), "b": list(range(4))} for name in modules}
        trace = {str(cid): {"assigned_quota": 96, "selected_positions": 192,
                            "active_factor": "both", "modules": support} for cid in range(10)}
        rows.append({"round": r, "selected_clients": list(range(10)),
                     "deployed_bytes": 20 * r, "diagnostic_after_budget_bytes": 0,
                     "evaluation": readout, "support_trace": trace,
                     "client_training": [{"client": cid, "optimizer_steps": 16,
                                          "optimizer_tensor_parameters": 192 * 768 + 1538}
                                         for cid in range(10)]})
    ck = {"format": "pcla-v7-runtime-1", "round": 100, "identity": "test-identity",
          "metrics": rows, "adapter": adapter, "budget_adapter": adapter,
          "half_adapter": adapter, "budget_stop_round": 100,
          "deployed_bytes": 2000, "diagnostic_bytes": 0,
          "cache": {"events": events}}
    torch.save(ck, run / "latest.pt")
    put(run / "metrics.json", rows)
    put(run / "status.json", {"state": "completed", "completed_round": 100,
         "budget_stop_round": 100, "deployed_bytes": 2000,
         "diagnostic_after_budget_bytes": 0})
    torch.save({"identity": "test-identity", "round": 100, "adapter": adapter,
                "deployed_bytes": 2000, "budget_cap_bytes": 2000}, run / "budget_return.pt")
    torch.save({"identity": "test-identity", "round": 50, "adapter": adapter,
                "deployed_bytes": 1000, "budget_cap_bytes": 1000,
                "readout": readout}, run / "half_budget_return.pt")
    return config_file, run, source


def test_complete_evidence_passes_then_corruptions_fail(tmp_path):
    config, run, source = evidence(tmp_path)
    result = audit(config, run, source)
    assert result["ok"], result["errors"]
    assert result["checks"]["budget_stop_round"] == 100
    assert result["checks"]["predicted_class_counts_round_100"] == [0, 4096]
    status = json.loads((run / "status.json").read_text())
    put(run / "status.json", {**status, "state": "paused"})
    assert not audit(config, run, source)["ok"]
    put(run / "status.json", status)
    logits = json.loads((run / "logits_round_100.json").read_text())
    logits[0] = [10., 0.]
    put(run / "logits_round_100.json", logits)
    assert any("round 100: accuracy mismatch" in x for x in audit(config, run, source)["errors"])
