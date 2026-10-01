import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from scripts import dispatch_pcla_v7_validation as dispatch
from scripts.initialize_pcla_v7_validation_queue import FIXED_SLOTS, SEEDS, initialize


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def claim_in_process(args):
    queue, family, worker = args
    result = dispatch.claim(Path(queue), family, worker)
    return result[1]["job"]["job_id"] if result else None


class ValidationDispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.root = base / "pcla_global_budget_20260928_v7"
        self.root.mkdir()
        self.queue = self.root / "validation_queue"
        self.cfg_dir = base / "configs"
        self.cfg_dir.mkdir()
        self.manifest = base / "fixed_jobs.jsonl"
        self.unresolved = base / "unresolved.json"
        self.pool_map = base / "pools.json"
        jobs = []
        for slot in FIXED_SLOTS:
            recipe = "D072" if slot == 10 else "D069" if slot == 9 else f"D{slot:03d}"
            for seed in SEEDS:
                jid = f"V_S{slot:02d}_{recipe}_s{seed}"
                jobs.append({"job_id": jid, "recipe_id": recipe, "phase": "V",
                    "protocol": "pcla-global-budget-20260928-v7", "budget_id": "B8",
                    "status": "draft_not_dispatched", "hardware_block": f"V_s{seed}",
                    "seed": seed, "depends_on": "", "priority": 30,
                    "output_dir": f"runs/pcla-global-budget-20260928-v7/V/{jid}"})
        self.jobs = {j["job_id"]: j for j in jobs}
        self.manifest.write_text("".join(json.dumps(j, sort_keys=True) + "\n" for j in jobs))
        put(self.unresolved, {"3": {"state": "unresolved_cost_tie", "choices": ["D011", "D035"]},
            "4": {"state": "depends_on_slot_03", "choices_by_slot_03": {"D011": "D015", "D035": "D042"}},
            "16": {"state": "depends_on_same_seed_slot_03_donor", "recipe": None}})
        put(self.pool_map, {f"V_s{seed}": "A100" if seed == 281 else "4090D" for seed in SEEDS})
        initialize(self.root, self.manifest, self.unresolved, self.pool_map)
        self.source = base / "source"
        put(self.source / "SOURCE_MANIFEST.json", {"files": {}})
        self.source_hash = dispatch.sha(self.source / "SOURCE_MANIFEST.json")
        self.technical = base / "technical.json"
        self.receipt = base / "release.json"
        self.ids = [f"V_S{slot:02d}_{'D072' if slot == 10 else 'D069' if slot == 9 else f'D{slot:03d}'}_s101"
                    for slot in (1, 9, 10)]
        self.config_hashes = {}
        for jid in self.ids:
            path = self.cfg_dir / f"{jid}.json"
            put(path, {"job": self.jobs[jid], "recipe": {"recipe_id": self.jobs[jid]["recipe_id"]}})
            self.config_hashes[jid] = dispatch.sha(path)
        self.write_receipts()

    def write_receipts(self, *, review="approved", config="passed", source="passed", tech="passed"):
        put(self.technical, {"schema": "pcla-v7-technical-gate-1", "phase": "V",
            "protocol": "pcla-global-budget-20260928-v7", "status": tech,
            "source_manifest_sha256": self.source_hash,
            "config_sha256_by_job": self.config_hashes})
        put(self.receipt, {"schema": "pcla-v7-release-gates-1", "phase": "V",
            "protocol": "pcla-global-budget-20260928-v7", "review_gate": review,
            "configuration_gate": config, "source_gate": source,
            "source_manifest_sha256": self.source_hash,
            "technical_receipt_sha256": dispatch.sha(self.technical),
            "config_sha256_by_job": self.config_hashes})

    def release(self, ids=None, *, family="4090D", mode="exclusive", receipt_hash=None, tech_hash=None):
        return dispatch.release(self.queue, ids or self.ids, self.cfg_dir, self.source,
            self.receipt, receipt_hash or dispatch.sha(self.receipt), self.technical,
            tech_hash or dispatch.sha(self.technical), family, mode)

    def test_156_initialized_and_parent_priority(self):
        self.assertEqual(dispatch.status(self.queue)["counts"], {"blocked": 156})
        self.release()
        self.assertEqual(dispatch.status(self.queue)["counts"], {"blocked": 153, "pending": 3})
        first = dispatch.claim(self.queue, "4090D", "worker-1")
        second = dispatch.claim(self.queue, "4090D", "worker-2")
        self.assertEqual(first[1]["job"]["recipe_id"], "D072")
        self.assertEqual(second[1]["job"]["recipe_id"], "D069")
        self.assertEqual(dispatch.finish(self.queue, first[0], "worker-1", 0), "executed_pending_audit")
        self.assertEqual(dispatch.finish(self.queue, second[0], "worker-2", 7), "failed_needs_review")
        self.assertEqual(dispatch.read(second[0])["attempts"][0]["exit_code"], 7)
        self.assertEqual(len(dispatch.read(second[0])["attempts"]), 1)

    def test_release_rejects_hash_gates_phase_output_and_family(self):
        jid = self.ids[0]
        with self.assertRaisesRegex(ValueError, "release receipt SHA"):
            self.release([jid], receipt_hash="0" * 64)
        with self.assertRaisesRegex(ValueError, "technical receipt SHA"):
            self.release([jid], tech_hash="0" * 64)
        self.write_receipts(tech="pending")
        with self.assertRaisesRegex(ValueError, "technical gate"):
            self.release([jid])
        self.write_receipts(review="pending")
        with self.assertRaisesRegex(ValueError, "release gate"):
            self.release([jid])
        self.write_receipts(config="pending")
        with self.assertRaisesRegex(ValueError, "release gate"):
            self.release([jid])
        self.write_receipts(source="pending")
        with self.assertRaisesRegex(ValueError, "release gate"):
            self.release([jid])
        self.write_receipts()
        with self.assertRaisesRegex(ValueError, "hardware family"):
            self.release([jid], family="A100")
        cfg = self.cfg_dir / f"{jid}.json"
        bad = json.loads(cfg.read_text())
        bad["job"]["phase"] = "D0"
        put(cfg, bad)
        with self.assertRaisesRegex(ValueError, "config identity"):
            self.release([jid])
        bad["job"]["phase"] = "V"
        bad["job"]["output_dir"] = f"runs/pcla-global-budget-20260928-v7/D0/{jid}"
        put(cfg, bad)
        with self.assertRaisesRegex(ValueError, "config identity"):
            self.release([jid])
        put(cfg, {"job": self.jobs[jid], "recipe": {"recipe_id": self.jobs[jid]["recipe_id"]},
                  "changed": True})
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            self.release([jid])
        self.assertEqual(dispatch.read(self.queue / "jobs" / f"{jid}.json")["state"], "blocked")

    def test_atomic_claim_no_duplicate_and_dependency_not_claimed(self):
        self.release()
        dependency = self.queue / "jobs" / f"{self.ids[0]}.json"
        item = dispatch.read(dependency)
        item["job"]["depends_on"] = "V_S03_D011_s101"
        dispatch.write(dependency, item)
        with multiprocessing.get_context("fork").Pool(2) as pool:
            got = pool.map(claim_in_process,
                [(str(self.queue), "4090D", f"worker-{i}") for i in range(2)])
        self.assertEqual(set(got), set(self.ids[1:]))
        self.assertIsNone(dispatch.claim(self.queue, "4090D", "third"))
        self.assertEqual(dispatch.read(dependency)["state"], "pending")

    def test_exclusive_can_claim_without_shared_profile(self):
        with self.assertRaisesRegex(ValueError, "shared mode requires"):
            self.release([self.ids[0]], mode="shared")
        self.release([self.ids[0]])
        self.assertIsNone(dispatch.claim(self.queue, "4090D", "shared", "shared"))
        item = dispatch.claim(self.queue, "4090D", "exclusive")
        self.assertEqual(item[1]["execution_mode"], "exclusive")


if __name__ == "__main__":
    unittest.main()
