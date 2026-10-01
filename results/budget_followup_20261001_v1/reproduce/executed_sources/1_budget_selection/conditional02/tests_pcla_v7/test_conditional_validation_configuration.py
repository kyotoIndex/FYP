"""CPU-only checks for the blocked, separate 36-job V extension compiler."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from pcla_research_v7.conditional_validation_configuration import (
    BRANCHES, CONDITIONAL_SLOTS, build_conditional_plan,
    verify_cost_resolution,
)
from pcla_research_v7.validation_configuration import V_SEEDS


PROJECT = Path(__file__).resolve().parents[3]
WORKTREE = Path(__file__).resolve().parents[1]
ATTACHMENT = PROJECT / ("handoffs/PCLA_D0_to_V_Received_20260928/attachment/"
                        "PCLA_D0_to_V_Decision_20260928")


def _pro_resolver():
    spec = importlib.util.spec_from_file_location("pro_resolve_v_fixture",
                                                  ATTACHMENT / "tools/resolve_v.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipts(expected, winner):
    # Synthetic accepted times exercise each branch, not a scientific winner.
    return {rid: {"schema": "pcla-v7-required-time-v1", "recipe_id": rid,
                  "seed": 401, "gpu_model": "NVIDIA GeForce RTX 4090 D",
                  "parity_passed": True, "interval_coverage_passed": True,
                  "exclusive_gpu": True, "original_quality_retained": True,
                  "required_seconds": float(100 if rid == winner else 101),
                  "interval_log_sha256": "a" * 64,
                  "timing_implementation_sha256": "b" * 64,
                  **{key: expected[rid][key] for key in
                     ("budget_stop_round", "deployed_bytes",
                      "source_manifest_sha256", "config_sha256")}}
            for rid in BRANCHES}


class ConditionalValidationConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads((ATTACHMENT / "configs/expected_cost_identities.json").read_text())
        cls.resolve = _pro_resolver().resolve

    def build(self, winner, mutate=None):
        receipts = _receipts(self.expected, winner)
        resolved = type(self).resolve(receipts)
        jobs = resolved.pop("jobs")
        if mutate:
            mutate(resolved, jobs, receipts)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "V_SELECTION.json").write_text(json.dumps(resolved))
            (root / "V_JOBS.jsonl").write_text("".join(json.dumps(job) + "\n" for job in jobs))
            cost = root / "accepted_cost.json"
            cost.write_text(json.dumps(receipts))
            return build_conditional_plan(
                handoff=PROJECT / "handoffs/PCLA_Longterm_Research_20260928_v7",
                run_root=PROJECT / "runs/pcla-global-budget-20260928-v7",
                source_train=WORKTREE / "data/SST-2/train.tsv",
                attachment=ATTACHMENT, resolution_dir=root, cost_receipts_path=cost)

    def test_both_pro_resolved_branches_have_36_unique_blocked_drafts(self):
        for winner in BRANCHES:
            with self.subTest(winner=winner):
                configs, receipt = self.build(winner)
                self.assertEqual(len(configs), 36)
                self.assertEqual(len({c["job"]["job_id"] for c in configs}), 36)
                self.assertEqual(receipt["resolution"]["R_local_recipe_id"], winner)
                self.assertEqual(receipt["resolution"]["R_cyclic_recipe_id"], BRANCHES[winner])
                self.assertEqual(receipt["conditional_slots"], CONDITIONAL_SLOTS)
                self.assertEqual({c["job"]["seed"] for c in configs}, set(V_SEEDS))
                self.assertEqual({c["conditional_validation"]["slot"] for c in configs},
                                 set(CONDITIONAL_SLOTS))
                for config in configs:
                    job, extension = config["job"], config["conditional_validation"]
                    slot, seed = extension["slot"], job["seed"]
                    self.assertEqual(job["recipe_id"], winner if slot in {3, 16}
                                     else BRANCHES[winner])
                    self.assertEqual(job["hardware_block"], f"V_s{seed}")
                    self.assertEqual(job["status"], "draft_blocked_dependencies")
                    self.assertEqual(extension["release_state"], "blocked_not_enqueued")
                    self.assertEqual(config["runtime"]["byte_cap"], 2382653750)
                    self.assertEqual(config["recipe"]["lr"], job["lr"])
                    self.assertTrue(config["manifest"].endswith("train_roles.json"))
                    if slot == 16:
                        self.assertEqual(extension["execution_policy"], "quota_replay_v1")
                        self.assertEqual(job["depends_on"], f"V_S03_{winner}_s{seed}")
                        self.assertEqual(extension["donor"]["seed"], seed)
                        self.assertEqual(extension["donor"]["recipe_id"], winner)
                        self.assertIn("support_trace_sha256",
                                      extension["donor"]["required_artifacts"])
                    else:
                        self.assertIsNone(extension["donor"])
                        self.assertEqual(job["depends_on"], "")
                self.assertIn("fixed queue requires exactly 156",
                              receipt["integration_gaps"][0])

    def test_receipt_and_pro_resolution_mismatch_rejected(self):
        pending = json.loads((ATTACHMENT /
            "jobs/resolved_without_cost/V_SELECTION.json").read_text())
        with self.assertRaises(ValueError):
            verify_cost_resolution(pending, {}, self.expected)
        for mutation in (
            lambda selection, jobs, receipts: receipts["D011"].update(
                source_manifest_sha256="0" * 64),
            lambda selection, jobs, receipts: receipts["D035"].update(
                interval_coverage_passed=False),
            lambda selection, jobs, receipts: selection["slots"]["R_local"].update(
                recipe_id="D035"),
            lambda selection, jobs, receipts: selection.update(aliases={"fake": "duplicate"}),
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    self.build("D011", mutation)

    def test_wrong_conditional_plan_seed_hardware_donor_or_count_rejected(self):
        mutations = (
            lambda s, jobs, r: jobs.__setitem__(2, {**jobs[2], "seed": 401}),
            lambda s, jobs, r: jobs.__setitem__(2, {**jobs[2], "hardware_family": "A100-PCIE-40GB"}),
            lambda s, jobs, r: next(job for job in jobs if job["slot"] == "R_replay")
                ["depends_on"].__setitem__(0, "V_R_local_s109"),
            lambda s, jobs, r: jobs.pop(),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    self.build("D035", mutation)


if __name__ == "__main__":
    unittest.main()
