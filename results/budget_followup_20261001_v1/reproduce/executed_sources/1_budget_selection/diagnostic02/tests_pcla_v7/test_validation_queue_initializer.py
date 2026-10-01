import json
from pathlib import Path
import tempfile
import unittest

from scripts.initialize_pcla_v7_validation_queue import (
    FIXED_SLOTS,
    SEEDS,
    initialize,
)


class ValidationQueueInitializerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root = base / "pcla-global-budget-20260928-v7"
        self.root.mkdir()
        self.manifest = base / "fixed_jobs.jsonl"
        self.unresolved = base / "unresolved_slots.json"
        self.pool_map = base / "pool_map.json"
        self.jobs = []
        for slot in FIXED_SLOTS:
            for seed in SEEDS:
                jid = f"V_S{slot:02d}_D{slot:03d}_s{seed}"
                self.jobs.append({
                    "job_id": jid, "recipe_id": f"D{slot:03d}",
                    "phase": "V", "protocol": "pcla-global-budget-20260928-v7",
                    "budget_id": "B8", "status": "draft_not_dispatched",
                    "hardware_block": f"V_s{seed}", "seed": seed,
                    "depends_on": "",
                    "output_dir": f"runs/{self.root.name}/V/{jid}",
                })
        self.write_inputs()

    def write_inputs(self):
        self.manifest.write_text("".join(json.dumps(job, sort_keys=True) + "\n"
                                         for job in self.jobs))
        self.unresolved.write_text(json.dumps({
            "3": {"state": "unresolved_cost_tie", "choices": ["D011", "D035"]},
            "4": {"state": "depends_on_slot_03",
                  "choices_by_slot_03": {"D011": "D015", "D035": "D042"}},
            "16": {"state": "depends_on_same_seed_slot_03_donor", "recipe": None},
        }))
        self.pool_map.write_text(json.dumps({f"V_s{seed}": "4090D" for seed in SEEDS}))

    def init(self, **kwargs):
        return initialize(self.root, kwargs.get("manifest", self.manifest),
                          self.unresolved, self.pool_map,
                          queue=kwargs.get("queue"))

    def test_156_fixed_jobs_are_blocked_and_three_dependencies_are_not_jobs(self):
        identity = self.init()
        queue = self.root / "validation_queue"
        files = list((queue / "jobs").glob("*.json"))
        self.assertEqual(len(files), 156)
        self.assertEqual(identity["count"], 156)
        self.assertEqual(set(identity["unresolved_slots"]), {"3", "4", "16"})
        self.assertEqual(identity["pool_map"], {f"V_s{seed}": "4090D" for seed in SEEDS})
        self.assertTrue(all(json.loads(path.read_text())["state"] == "blocked" for path in files))
        self.assertTrue(all(json.loads(path.read_text())["attempts"] == [] for path in files))
        self.assertFalse((self.root / "queue").exists())
        self.assertFalse((self.root / "V").exists())

    def test_repeat_is_byte_identical_and_changed_identity_is_rejected(self):
        self.init()
        queue = self.root / "validation_queue"
        identity = (queue / "matrix_identity.json").read_bytes()
        job_file = queue / "jobs" / f"V_S01_D001_s{SEEDS[0]}.json"
        job = job_file.read_bytes()
        self.init()
        self.assertEqual((queue / "matrix_identity.json").read_bytes(), identity)
        self.assertEqual(job_file.read_bytes(), job)

        pools = json.loads(self.pool_map.read_text())
        pools[f"V_s{SEEDS[0]}"] = "A100"
        self.pool_map.write_text(json.dumps(pools))
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.init()
        self.assertEqual(job_file.read_bytes(), job)

    def test_physical_hpc_root_does_not_change_logical_output_identity(self):
        self.root = self.root.parent / "pcla_global_budget_20260928_v7"
        self.root.mkdir()
        self.assertEqual(self.init()["count"], 156)

    def test_rejects_d0_queue_manifest_and_output_paths(self):
        with self.assertRaisesRegex(ValueError, "independent"):
            self.init(queue=self.root / "queue")
        d0 = self.root / "D0"
        d0.mkdir()
        d0_manifest = d0 / "fixed_jobs.jsonl"
        d0_manifest.write_bytes(self.manifest.read_bytes())
        with self.assertRaisesRegex(ValueError, "D0 path"):
            self.init(manifest=d0_manifest)
        self.jobs[0]["output_dir"] = f"runs/{self.root.name}/D0/{self.jobs[0]['job_id']}"
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "not isolated"):
            self.init()
        self.assertFalse((self.root / "validation_queue").exists())

    def test_rejects_duplicate_id_missing_seed_and_d0_matrix(self):
        self.jobs[1] = dict(self.jobs[0])
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.init()
        self.jobs[1] = dict(self.jobs[1], job_id="V_S01_D001_s401", seed=401,
                            hardware_block="V_s401",
                            output_dir=f"runs/{self.root.name}/V/V_S01_D001_s401")
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "unexpected V seed"):
            self.init()
        self.jobs = [{"job_id": f"D{i:03d}_s401", "phase": "D0"} for i in range(351)]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "156 fixed V jobs"):
            self.init()

    def test_requires_complete_explicit_hardware_family_assignment(self):
        pools = json.loads(self.pool_map.read_text())
        del pools[f"V_s{SEEDS[-1]}"]
        self.pool_map.write_text(json.dumps(pools))
        with self.assertRaisesRegex(ValueError, "all 12 V seed blocks"):
            self.init()
        self.assertFalse((self.root / "validation_queue").exists())

    def test_existing_d0_identity_or_foreign_jobs_are_not_overwritten(self):
        d0_queue = self.root / "queue"
        d0_queue.mkdir()
        (d0_queue / "matrix_identity.json").write_text(json.dumps({"count": 351}))
        original_d0 = (d0_queue / "matrix_identity.json").read_bytes()
        queue = self.root / "validation_queue"
        queue.mkdir()
        (queue / "matrix_identity.json").write_text(json.dumps({"count": 351}))
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.init()
        self.assertEqual(json.loads((queue / "matrix_identity.json").read_text()), {"count": 351})
        self.assertFalse((queue / "queue.lock").exists())
        self.assertEqual((d0_queue / "matrix_identity.json").read_bytes(), original_d0)

    def test_partial_foreign_queue_is_rejected_without_overwrite(self):
        queue = self.root / "validation_queue"
        jobs = queue / "jobs"
        jobs.mkdir(parents=True)
        foreign = jobs / "D001_s401.json"
        foreign.write_text('{"state":"completed"}')
        with self.assertRaisesRegex(ValueError, "partial or foreign queue"):
            self.init()
        self.assertEqual(foreign.read_text(), '{"state":"completed"}')
        self.assertFalse((queue / "matrix_identity.json").exists())


if __name__ == "__main__":
    unittest.main()
