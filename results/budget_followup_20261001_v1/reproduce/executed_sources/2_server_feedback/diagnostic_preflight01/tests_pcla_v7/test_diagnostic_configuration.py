import csv
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from pcla_research_v7.diagnostic_configuration import (
    compile_diagnostics, participants, sha, verify_anchor,
)
from scripts.build_pcla_v7_diagnostic_configs import stage

PROJECT = Path(__file__).resolve().parents[3]
RUN = PROJECT / "runs/pcla-global-budget-20260928-v7"
CSV = PROJECT / "handoffs/PCLA_D0_to_V_Received_20260928/attachment/PCLA_D0_to_V_Decision_20260928/jobs/cohort_diagnostics_144.csv"
MANIFEST = RUN / "data_manifest/train_roles.json"
HARDWARE = RUN / "checks/v_execution_intake_20260928/seed_hardware_map.json"
GATE = RUN / "checks/v_execution_intake_20260928/validation02_release_gate.json"
EXECUTION = Path("$HPC_HOME/pcla_global_budget_20260928_v7")


class DiagnosticConfigurationTests(unittest.TestCase):
    def test_actual_144_stage_is_blocked_and_parent_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "stage"
            receipt = stage(CSV, RUN, EXECUTION, MANIFEST, HARDWARE, GATE, out,
                            anchor_root=Path(tmp) / "empty_anchors")
            self.assertEqual(receipt["lines"], {"cohort": 96, "direction": 48})
            self.assertEqual(receipt["anchor_states"], {"pending_parent_anchor": 144})
            self.assertEqual(len(list((out / "diagnostic_configs").glob("*.json"))), 144)
            a = json.loads((out / "diagnostic_configs/Cohort_s101_full_alt_r10_a.json").read_text())
            b = json.loads((out / "diagnostic_configs/Cohort_s101_full_r16_r10_b.json").read_text())
            d = json.loads((out / "diagnostic_configs/Direction_s281_r60_b.json").read_text())
            self.assertEqual(a["parent"]["job_id"], "V_S10_D072_s101")
            self.assertEqual(b["parent"]["job_id"], "V_S09_D069_s101")
            self.assertEqual(d["parent"]["job_id"], "V_S10_D072_s281")
            self.assertEqual(a["participants"], [47, 13, 26, 23, 25, 39, 37, 35, 27, 4])
            self.assertEqual(a["training"]["prefix_steps"], [1, 4, 16])
            self.assertEqual(a["training"]["learning_rate"], .001)
            self.assertEqual(d["training"]["learning_rate"], .0003)
            self.assertEqual(d["hardware_family"], "A100")
            self.assertIsNone(d["visits"])
            self.assertEqual(a["dependencies"]["diagnostic_runner"], "not_implemented")
            with self.assertRaisesRegex(ValueError, "already exists"):
                stage(CSV, RUN, EXECUTION, MANIFEST, HARDWARE, GATE, out,
                      anchor_root=Path(tmp) / "empty_anchors")

    def test_rejects_changed_csv_recipe_or_parent_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with CSV.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            changed = base / "changed.csv"
            rows[0]["learning_rate"] = "0.003"
            with changed.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "CSV contract"):
                compile_diagnostics(changed, RUN, EXECUTION, MANIFEST, HARDWARE, GATE,
                                    anchor_root=base / "empty_anchors")
            gate = json.loads(GATE.read_text())
            gate["config_sha256_by_job"]["V_S10_D072_s101"] = "0" * 64
            bad_gate = base / "gate.json"
            bad_gate.write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError, "parent config binding"):
                compile_diagnostics(CSV, RUN, EXECUTION, MANIFEST, HARDWARE, bad_gate,
                                    anchor_root=base / "empty_anchors")

    def test_anchor_missing_is_pending_and_valid_anchor_uses_its_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "V_S10_D072_s101"
            out.mkdir()
            anchor = out / "anchor_round_010.pt"
            config = root / "parent.json"
            config.write_text(json.dumps({"job": {"seed": 101}}))
            cfg_sha = sha(config)
            pending = verify_anchor(anchor, seed=101, anchor=10, client_ids=list(range(50)),
                                    parent_config=config, parent_config_sha256=cfg_sha,
                                    source_manifest_sha256="a" * 64)
            self.assertEqual(pending["state"], "pending_parent_anchor")
            anchor.write_bytes(b"atomic checkpoint fixture")
            (out / "dispatch_identity.json").write_text(json.dumps({
                "config_sha256": cfg_sha, "source_manifest_sha256": "a" * 64}))
            (out / "execution_attempts.json").write_text(json.dumps([{"identity": "frozen-parent"}]))
            rows = [{"round": r, "selected_clients": participants(101, r, list(range(50)))}
                    for r in range(1, 11)]
            (out / "metrics.json").write_text(json.dumps(rows))
            class Tensor:
                pass
            payload = {"format": "pcla-v7-runtime-1", "identity": "frozen-parent",
                       "round": 10, "adapter": {"x": 1}, "cache": {},
                       "torch_rng": Tensor(), "python_rng": (1,), "cuda_rng": [Tensor()],
                       "metrics": rows}
            fake_torch = types.SimpleNamespace(Tensor=Tensor, load=lambda *a, **k: payload)
            with patch.dict(sys.modules, {"torch": fake_torch}):
                result = verify_anchor(anchor, seed=101, anchor=10, client_ids=list(range(50)),
                    parent_config=config, parent_config_sha256=cfg_sha,
                    source_manifest_sha256="a" * 64)
                self.assertEqual(result["state"], "anchor_verified")
                selected = participants(101, 11, list(range(50)))
                self.assertEqual(result["visits_from_anchor_metrics"],
                                 {str(cid): sum(cid in row["selected_clients"] for row in rows)
                                  for cid in selected})
                payload["metrics"][0]["selected_clients"] = []
                with self.assertRaisesRegex(ValueError, "participation prefix"):
                    verify_anchor(anchor, seed=101, anchor=10, client_ids=list(range(50)),
                        parent_config=config, parent_config_sha256=cfg_sha,
                        source_manifest_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
