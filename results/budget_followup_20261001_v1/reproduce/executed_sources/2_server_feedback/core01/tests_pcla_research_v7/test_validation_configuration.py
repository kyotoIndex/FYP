"""Contract checks for the draft-only V matrix and D0 separation."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from pcla_research_v7.configuration import compile_first_batch
from pcla_research_v7.data import digest
from pcla_research_v7.validation_configuration import (
    DEPENDENT_SLOTS, FIXED_SLOTS, build_plan, compile_validation_job,
    load_attachment_plan, load_inputs, sha256, validate_matrix,
)

REPO = Path(__file__).resolve().parents[1]
PCLA = REPO.parents[1]
HANDOFF = PCLA / "handoffs/PCLA_Longterm_Research_20260928_v7"
RUN = PCLA / "runs/pcla-global-budget-20260928-v7"
TRAIN = REPO / "data/SST-2/train.tsv"
ATTACHMENT = PCLA / "handoffs/PCLA_D0_to_V_Received_20260928/attachment/PCLA_D0_to_V_Decision_20260928"


class ValidationConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expanded, cls.receipt = build_plan(HANDOFF, RUN, TRAIN)

    def test_fixed_matrix_and_unresolved_slots(self):
        expanded, receipt = self.expanded, self.receipt
        jobs = [x["job"] for x in expanded]
        self.assertEqual(len(expanded), 156)
        self.assertEqual(len({x["job_id"] for x in jobs}), 156)
        self.assertEqual(Counter(int(x["job_id"].split("_")[1][1:]) for x in jobs),
                         Counter({s: 12 for s in FIXED_SLOTS}))
        self.assertEqual(Counter(x["seed"] for x in jobs),
                         Counter({s: 13 for s in receipt["seeds"]}))
        self.assertEqual({x["phase"] for x in jobs}, {"V"})
        self.assertEqual({x["status"] for x in jobs}, {"draft_not_dispatched"})
        self.assertTrue(all("/V/" in x["output_dir"] and "/D0/" not in x["output_dir"] for x in jobs))
        self.assertEqual(set(receipt["dependent_slots"]), {"3", "4", "16"})
        self.assertEqual(receipt["dependent_slots"]["4"]["choices_by_slot_03"],
                         {"D011": "D015", "D035": "D042"})
        self.assertIsNone(DEPENDENT_SLOTS[16]["recipe"])

    def test_native_and_core_expansion_matches_frozen_recipe(self):
        expanded, receipt = self.expanded, self.receipt
        by_slot = {int(x["job"]["job_id"].split("_")[1][1:]): x
                   for x in expanded if x["job"]["seed"] == 101}
        self.assertTrue(all(by_slot[s]["recipe"]["required_adapter"] == "core"
                            for s in FIXED_SLOTS if s not in {13, 14, 15}))
        self.assertEqual([by_slot[s]["recipe"]["required_adapter"] for s in (13, 14, 15)],
                         ["fedsb", "fedsvd", "fedplora"])
        self.assertEqual(by_slot[13]["recipe"]["rank"], 32)
        self.assertIs(by_slot[13]["recipe"]["uses_public_training"], True)
        self.assertEqual(by_slot[14]["recipe"]["optimizer"], "sgd")
        self.assertEqual(by_slot[14]["recipe"]["adapter_clip_norm"], 2.0)
        self.assertEqual(by_slot[14]["recipe"]["factor_schedule"], "B_only")
        self.assertEqual(by_slot[14]["recipe"]["head_lr"], .001)
        self.assertEqual(by_slot[14]["runtime"]["weight_decay"], 0.0)
        self.assertEqual(by_slot[15]["recipe"]["native_ranks"], [2, 3, 4])
        self.assertEqual(by_slot[15]["recipe"]["factor_schedule"], "both")
        self.assertEqual(by_slot[13]["runtime"]["weight_decay"], .01)
        self.assertEqual(by_slot[15]["runtime"]["weight_decay"], .01)
        self.assertEqual({x["runtime"]["byte_cap"] for x in expanded}, {2382653750})
        identity = receipt["source_identity"]["D0_reference_paths"]
        self.assertTrue(all(x["manifest"] == identity["manifest"] and
                            x["train_file"] == identity["train_file"] and
                            x["token_cache"] == identity["token_cache"] and
                            x["model"] == identity["model"] for x in expanded))
        self.assertIn("/sources/native04/", receipt["D0_audited_execution"]["D107"]["executed_source_manifest"])
        self.assertIn("/configs/native03/", receipt["D0_audited_execution"]["D107"]["remote_config"])
        self.assertIn("/sources/native03/", receipt["D0_audited_execution"]["D105"]["executed_source_manifest"])
        self.assertEqual(receipt["D107_native04_fedsvd_sha256"],
                         hashlib.sha256((REPO / "pcla_research_v7/native_fedsvd.py").read_bytes()).hexdigest())

    def test_reject_invalid_seed_duplicate_and_unresolved(self):
        inputs = load_inputs(HANDOFF, RUN, TRAIN)
        recipes, seeds = inputs["recipes"], inputs["seeds"]
        for seed in (401, 9001, 999, True, 101.0):
            with self.assertRaisesRegex(ValueError, "seed outside"):
                compile_validation_job(1, seed, recipes, seeds)
        with self.assertRaisesRegex(ValueError, "seed outside"):
            compile_validation_job(1, 401, recipes, [401])
        for slot in (3, 4, 16):
            with self.assertRaisesRegex(ValueError, "unresolved"):
                compile_validation_job(slot, 101, recipes, seeds)
        with self.assertRaisesRegex(ValueError, "altered"):
            compile_validation_job(1, 101, recipes, seeds, slot_map={1: "D001"})
        jobs = [dict(job=x["job"]) for x in self.expanded]
        jobs[1] = copy.deepcopy(jobs[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_matrix(jobs, seeds)
        jobs = [dict(job=copy.deepcopy(x["job"])) for x in self.expanded]
        jobs[0]["job"]["recipe_id"] = "D001"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            validate_matrix(jobs, seeds)

    def test_missing_validation_partition_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data_manifest"
            data.mkdir()
            manifest = json.loads((RUN / "data_manifest/train_roles.json").read_text())
            manifest["partitions"].pop("101")
            manifest["split_sha256"] = digest({k: v for k, v in manifest.items() if k != "split_sha256"})
            (data / "train_roles.json").write_text(json.dumps(manifest))
            (data / "b8_wire.json").write_bytes((RUN / "data_manifest/b8_wire.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "validation seed partition absent"):
                load_inputs(HANDOFF, root, TRAIN)

    def test_v_data_identity_and_d0_compiler_unchanged(self):
        receipt = self.receipt
        manifest = json.loads((RUN / "data_manifest/train_roles.json").read_text())
        self.assertEqual(receipt["source_identity"]["split_sha256"], manifest["split_sha256"])
        self.assertTrue(all(str(seed) in manifest["partitions"] for seed in receipt["seeds"]))
        recipe_path = HANDOFF / "configs/recipes.json"
        jobs_path = HANDOFF / "jobs/first_batch.jsonl"
        before = (hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
                  hashlib.sha256(jobs_path.read_bytes()).hexdigest())
        d0 = compile_first_batch(recipe_path, jobs_path)
        after = (hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
                 hashlib.sha256(jobs_path.read_bytes()).hexdigest())
        self.assertEqual(len(d0), 351)
        self.assertTrue(all(x["job"]["phase"] == "D0" for x in d0))
        self.assertEqual(before, after)

    def test_attachment_156_field_parity_and_local_binding(self):
        expanded, receipt = build_plan(HANDOFF, RUN, TRAIN, ATTACHMENT)
        plan = [json.loads(line) for line in
                (ATTACHMENT / "jobs/resolved_without_cost/V_JOBS.jsonl").read_text().splitlines()]
        by_plan_id = {row["job_id"]: row for row in plan}
        self.assertEqual(len(by_plan_id), 156)
        self.assertEqual(len(receipt["attachment_plan_job_id_mapping"]), 156)
        for config in expanded:
            job, ref = config["job"], config["validation_plan"]
            self.assertEqual(ref, {k: v for k, v in by_plan_id[ref["job_id"]].items()
                                   if k not in {"recipe", "common"}})
            self.assertEqual(job["priority"], ref["priority"])
            self.assertEqual(job["requires"].split(";"), ref["requires"])
            self.assertEqual(job["hardware_block"], ref["hardware_block"])
            self.assertEqual(job["method"], ref["execution_policy"])
            self.assertEqual({k: v for k, v in config["recipe"].items()
                              if k != "recipe_sha256"}, by_plan_id[ref["job_id"]]["recipe"])
            self.assertEqual(config["common"]["budget"]["byte_cap"], 2382653750)
            self.assertEqual(config["common"]["data_roles"]["manifest_sha256"],
                             sha256(RUN / "data_manifest/train_roles.json"))
            self.assertEqual(job["status"], "draft_not_dispatched")
        self.assertEqual({x["validation_plan"]["hardware_family"] for x in expanded},
                         {"RTX4090D", "A100-PCIE-40GB"})

    def test_attachment_tamper_and_duplicate_rejected(self):
        inputs = load_inputs(HANDOFF, RUN, TRAIN)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("configs/common_resolved.json", "configs/recipe_catalogue.json",
                         "jobs/resolved_without_cost/V_JOBS.jsonl"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ATTACHMENT / name, target)
            args = (root, inputs["common"], inputs["recipes"], inputs["seeds"],
                    sha256(RUN / "data_manifest/train_roles.json"), inputs["b8"]["total_bytes"])
            load_attachment_plan(*args)
            path = root / "configs/common_resolved.json"
            original = path.read_text()
            value = json.loads(original)
            value["budget"]["byte_cap"] -= 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "attachment common"):
                load_attachment_plan(*args)
            path.write_text(original)
            jobs = root / "jobs/resolved_without_cost/V_JOBS.jsonl"
            lines = jobs.read_text().splitlines()
            lines[1] = lines[0]
            jobs.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate attachment"):
                load_attachment_plan(*args)


if __name__ == "__main__":
    unittest.main()
