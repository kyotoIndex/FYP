from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import torch

from common import (
    atomic_json, bind_project, extract_verified, handoff_hashes, sha256_file,
)
from aggregation import accumulator_type
from analyze import classification
from dispatch import (
    claim_ready, create_abort, initialize, load_queue, make_jobs,
    open_phase_pending, recover, refresh_blocked,
)
from fixed_state import coverage_report

CORE = RUNTIME = REPO = None


class AggregationTests(unittest.TestCase):
    def accumulator(self, base, total=2):
        cls = accumulator_type(CORE.DeltaAccumulator, torch, {"x": 1.0})
        return cls(base, total, "fixed")

    def base(self):
        return {
            "x.lora_a": torch.tensor([[1.0, 2.0], [0.5, -1.0]],
                                     dtype=torch.float64),
            "x.lora_b": torch.zeros(2, 2, dtype=torch.float64),
            "classifier.bias": torch.zeros(1, dtype=torch.float64),
        }

    def add_b(self, accumulator, delta, mask, head, weight=1):
        local = {name: value.clone() for name, value in accumulator.base.items()}
        local["x.lora_b"] += torch.tensor(delta, dtype=torch.float64)
        local["classifier.bias"] += torch.tensor([head], dtype=torch.float64)
        accumulator.add(local, {
            "x.lora_b": torch.tensor(mask, dtype=torch.bool),
            "classifier.bias": torch.ones(1, dtype=torch.bool),
        }, weight)

    def test_missing_client_weight_is_rejected(self):
        acc = self.accumulator(self.base(), total=3)
        self.add_b(acc, [[2, 2], [0, 0]], [[1, 1], [0, 0]], 1, weight=1)
        self.add_b(acc, [[0, 2], [0, 0]], [[0, 1], [0, 0]], 3, weight=1)
        with self.assertRaises(ValueError):
            acc.finish()

    def test_denominators_function_and_parameter_controls(self):
        acc = self.accumulator(self.base(), total=2)
        self.add_b(acc, [[2, 2], [0, 0]], [[1, 1], [0, 0]], 1, weight=1)
        self.add_b(acc, [[0, 2], [0, 0]], [[0, 1], [0, 0]], 3, weight=1)
        fixed = acc.result_for("fixed")
        contributors = acc.result_for("contributors")
        function = acc.result_for("function_norm_matched")
        parameter = acc.result_for("parameter_norm_matched")
        self.assertTrue(torch.equal(fixed["classifier.bias"],
                                    contributors["classifier.bias"]))
        self.assertTrue(torch.equal(fixed["classifier.bias"],
                                    function["classifier.bias"]))
        readout = acc.readout()
        self.assertLessEqual(readout["function_match_relative_error"], 1e-12)
        self.assertLessEqual(readout["parameter_match_relative_error"], 1e-12)
        self.assertFalse(torch.equal(fixed["x.lora_b"],
                                     contributors["x.lora_b"]))
        self.assertFalse(torch.equal(parameter["x.lora_b"],
                                     function["x.lora_b"]))

    def test_function_gain_is_rankwise_gauge_invariant(self):
        cls = accumulator_type(CORE.DeltaAccumulator, torch, {"x": 1.0})
        base = self.base()
        acc = cls(base, 2, "fixed")
        delta1 = torch.tensor([[2.0, 2.0], [0.0, 0.0]])
        delta2 = torch.tensor([[0.0, 2.0], [0.0, 0.0]])
        self.add_b(acc, delta1.tolist(), [[1, 1], [0, 0]], 1)
        self.add_b(acc, delta2.tolist(), [[0, 1], [0, 0]], 1)
        first = acc.readout()

        column_scale = torch.tensor([7.0, 0.5], dtype=torch.float64)
        transformed = {name: value.clone() for name, value in base.items()}
        transformed["x.lora_a"] /= column_scale[:, None]
        transformed["x.lora_b"] *= column_scale[None, :]
        other = cls(transformed, 2, "fixed")
        self.add_b(other, (delta1 * column_scale[None, :]).tolist(),
                   [[1, 1], [0, 0]], 1)
        self.add_b(other, (delta2 * column_scale[None, :]).tolist(),
                   [[0, 1], [0, 0]], 1)
        second = other.readout()
        self.assertAlmostEqual(first["function_gain"], second["function_gain"],
                               places=12)
        self.assertNotAlmostEqual(first["parameter_gain"],
                                  second["parameter_gain"], places=5)

    def test_full_coverage_rules_agree(self):
        acc = self.accumulator(self.base(), total=2)
        self.add_b(acc, [[1, 2], [3, 4]], [[1, 1], [1, 1]], 1)
        self.add_b(acc, [[5, 6], [7, 8]], [[1, 1], [1, 1]], 1)
        reference = acc.result_for("fixed")
        for rule in ("contributors", "function_norm_matched",
                     "parameter_norm_matched"):
            for name in reference:
                torch.testing.assert_close(reference[name],
                                           acc.result_for(rule)[name])

    def test_both_factor_coverage_is_rejected(self):
        cls = accumulator_type(CORE.DeltaAccumulator, torch, {"x": 1.0})
        base = self.base()
        acc = cls(base, 1, "fixed")
        local = {name: value.clone() for name, value in base.items()}
        local["x.lora_a"] += 1
        local["x.lora_b"] += 1
        local["classifier.bias"] += 1
        acc.add(local, {
            "x.lora_a": torch.ones_like(local["x.lora_a"], dtype=torch.bool),
            "x.lora_b": torch.ones_like(local["x.lora_b"], dtype=torch.bool),
            "classifier.bias": torch.ones(1, dtype=torch.bool),
        }, 1)
        with self.assertRaises(ValueError):
            acc.readout()


class UtilityTests(unittest.TestCase):
    def test_coverage_composition(self):
        clients = [
            {"sample_weight": 10, "positive_samples": 8, "capacity": 6,
             "support": [[True, False], [False, False]]},
            {"sample_weight": 20, "positive_samples": 4, "capacity": 12,
             "support": [[False, True], [False, False]]},
        ]
        rows = coverage_report(torch, clients, 30, (2, 2))
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(rows[0]["coverage_fraction"], 1 / 3)
        self.assertAlmostEqual(rows[0]["contributor_positive_ratio"], 0.8)
        self.assertAlmostEqual(rows[1]["contributor_positive_ratio"], 0.2)
        self.assertGreater(rows[0]["capacity_sample_total_variation"], 0)

    def test_classification_recomputation_and_nonfinite_rejection(self):
        result = classification([[2.0, 0.0], [0.0, 2.0]], [0, 1])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["confusion"], [[1, 0], [0, 1]])
        self.assertFalse(result["strict_one_class"])
        with self.assertRaises(ValueError):
            classification([[float("nan"), 0.0]], [0])

    def test_unsafe_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../escape.txt", "x")
                zf.writestr("ARCHIVE_MANIFEST.json", json.dumps({
                    "../escape.txt": {"bytes": 1,
                                      "sha256": "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"}
                }))
            with self.assertRaises(ValueError):
                extract_verified(archive, sha256_file(archive), root / "out")

    def test_job_plan_phase_counts(self):
        site = {"full_run_timeout_seconds": 1800,
                "diagnostic_timeout_seconds": 900}
        jobs = make_jobs(site)
        self.assertEqual({phase: sum(x["phase"] == phase for x in jobs)
                          for phase in (1, 2, 3, 4)},
                         {1: 4, 2: 8, 3: 16, 4: 24})

    def _dummy_queue(self, root):
        repo = root / "repo"; repo.mkdir()
        source = root / "source"; source.mkdir()
        python = root / "python"; python.write_text("x")
        site = {
            "repo": str(repo), "source_root": str(source),
            "queue_root": str(root / "queue"),
            "scratch_root": str(root / "scratch"),
            "python_pcla": str(python), "require_slurm": False,
            "environment": {}, "full_run_timeout_seconds": 1800,
            "diagnostic_timeout_seconds": 900,
        }
        site_path = root / "site.json"; atomic_json(site_path, site)
        report = root / "tests.json"
        atomic_json(report, {"passed": True,
                             "handoff_sources": handoff_hashes()})
        initialize(site_path, report)
        return site_path, load_queue(root / "queue")

    def test_frozen_site_abort_blocking_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site_path, loaded = self._dummy_queue(root)
            queue, plan, frozen, paths = loaded
            original_hash = sha256_file(queue / "frozen_site.json")
            changed = json.loads(site_path.read_text())
            changed["repo"] = str(root / "changed")
            atomic_json(site_path, changed)
            self.assertEqual(original_hash,
                             sha256_file(queue / "frozen_site.json"))

            failed = paths["failed"] / "parity_pcla_7.json"
            atomic_json(failed, {"state": "failed", "job_id": "parity_pcla_7"})
            blocked = refresh_blocked(plan, paths)
            self.assertIn("trace_pcla_contributors_7", blocked)
            create_abort(queue, {"job_id": "parity_pcla_7"})
            job, claim = claim_ready(queue, plan, frozen, paths, "GPU-test")
            self.assertIsNone(job)
            self.assertIsNone(claim)

            recover(queue, "parity_pcla_7", "external_interruption",
                    True, False)
            self.assertFalse((queue / "ABORT.json").exists())
            self.assertFalse((paths["failed"] / "parity_pcla_7.json").exists())
            self.assertFalse((paths["blocked"] /
                              "trace_pcla_contributors_7.json").exists())
            self.assertIn("parity_pcla_7",
                          open_phase_pending(queue, plan, paths))

    def test_four_round_exact_support_replay(self):
        from pcla_repair.testing import fixture, tiny_config
        from pcla_runner import execute
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, layout, clients, capacities, evaluation = fixture()
            cfg = tiny_config()
            RUNTIME.run_experiment(
                model, layout, clients, capacities, evaluation,
                cfg, 7, "pcla", root / "original", {"synthetic": True}
            )
            source = RUNTIME.load_checkpoint(root / "original/checkpoint.pt")
            model, layout, clients, capacities, evaluation = fixture()
            execute(
                torch, CORE, RUNTIME, model, layout, clients, capacities,
                evaluation, cfg, 7, root / "replay", {"synthetic": True},
                None, trace=source["trace"],
            )
            current = RUNTIME.load_checkpoint(root / "replay/checkpoint.pt")
            for name, value in source["adapter"].items():
                self.assertTrue(torch.equal(value, current["adapter"][name]), name)
            for key, value in source["trace"].items():
                self.assertTrue(torch.equal(value, current["trace"][key]), key)


def main():
    global CORE, RUNTIME, REPO
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    REPO = Path(args.repo).resolve()
    _, CORE, RUNTIME, _ = bind_project(REPO)
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(AggregationTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(UtilityTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    atomic_json(args.out, {
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "handoff_sources": handoff_hashes(),
        "torch": str(torch.__version__),
        "scope": "CPU numerical, queue and synthetic replay; not BERT quality",
    })
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
