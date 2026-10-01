import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import launch_pcla_v7_validation_workers as launcher


class FakeProbe:
    def __init__(self):
        self.sessions_value = []
        self.steps_value = {}
        self.busy = False

    def sessions(self):
        return self.sessions_value[:]

    def allocation(self, jid, spec):
        return {"job_id": jid, "node": "node" + jid, "owner": "owner", "job_state": "RUNNING",
                "hardware_family": spec["hardware_family"], "gpu_type": spec["gpu_type"],
                "gpu_count": spec["count"], "gpu_indices": list(range(spec["count"])),
                "scontrol": "verified"}

    def steps(self, jid):
        return self.steps_value.get(jid, [])

    def cards(self, node, gpu_type, allocated_indices, fallback_allocation):
        if self.busy:
            raise RuntimeError("GPU occupancy")
        return {"node": node, "gpu_type": gpu_type,
                "cards": [{"index": i, "uuid": f"GPU-{node}-{i}"} for i in allocated_indices],
                "compute_apps": [], "raw_gpu": "", "raw_compute_apps": ""}


class ValidationLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root = base / "pcla_global_budget_20260928_v7"
        self.queue = self.root / "validation_queue"
        self.queue.mkdir(parents=True)
        (self.queue / "matrix_identity.json").write_text(json.dumps({
            "schema": "pcla-v7-validation-queue-1", "protocol": "pcla-global-budget-20260928-v7",
            "phase": "V", "count": 156, "manifest_sha256": "a" * 64}))
        self.dispatcher = base / "dispatch_pcla_v7_validation.py"
        self.dispatcher.write_text("# test\n")
        self.python = Path(sys.executable).resolve()
        self.map = base / "allocation_map.json"
        self.specs = {jid: {"hardware_family": family, "gpu_type": gpu, "count": count}
                      for jid, (family, gpu, count) in launcher.EXPECTED.items()}
        self.write_map()
        self.probe = FakeProbe()

    def write_map(self):
        self.map.write_text(json.dumps({"schema": "pcla-v7-existing-allocations-1",
                                        "allocations": self.specs}))

    def plan(self, execute=False, run=None):
        return launcher.launch(self.queue, self.dispatcher, self.python, self.map,
                               "test1", execute=execute, probe=self.probe,
                               run=run or launcher.command)

    def test_dry_run_maps_four_allocations_to_twelve_exclusive_workers(self):
        plan = self.plan()
        self.assertEqual(len(plan["workers"]), 12)
        self.assertEqual({j: sum(w["allocation"] == j for w in plan["workers"])
                          for j in self.specs}, {j: spec["count"] for j, spec in self.specs.items()})
        self.assertEqual(set(w["mode"] for w in plan["workers"]), {"exclusive"})
        self.assertTrue(all("--exact" in w["argv"] and "-n1" in w["argv"] and "-c4" in w["argv"]
                            and f"--gres=gpu:{w['gpu_type']}:1" in w["argv"]
                            and "--overlap" not in w["argv"] for w in plan["workers"]))
        self.assertFalse((self.root / "v_workers").exists())

    def test_rejects_unknown_allocation_steps_busy_gpu_and_worker(self):
        self.specs["173698"]["count"] = 3
        self.write_map()
        with self.assertRaisesRegex(ValueError, "fixed four allocations"):
            self.plan()
        self.specs["173698"]["count"] = 4
        self.write_map()
        self.probe.steps_value["173698"] = [{"step_id": "173698.1"}]
        with self.assertRaisesRegex(RuntimeError, "existing Slurm step"):
            self.plan()
        self.probe.steps_value.clear()
        self.probe.busy = True
        with self.assertRaisesRegex(RuntimeError, "occupancy"):
            self.plan()
        self.probe.busy = False
        self.probe.sessions_value = ["pclav7_v_previous_173697_0"]
        with self.assertRaisesRegex(RuntimeError, "existing V tmux"):
            self.plan()

    def test_live_probe_rejects_wrong_owner_and_unknown_step(self):
        spec = self.specs["173697"]
        def run(argv, **kwargs):
            if argv[:3] == ["scontrol", "show", "job"]:
                raw = "JobId=173697 JobState=RUNNING UserId=other(1) NodeList=node TRES=cpu=16,gres/gpu=4 TresPerNode=gres:gpu:4090:4"
            else:
                raw = "173697.unknown\n"
            return subprocess.CompletedProcess(argv, 0, raw, "")
        live = launcher.LiveProbe(run)
        with self.assertRaisesRegex(RuntimeError, "live allocation"):
            live.allocation("173697", spec)
        with self.assertRaisesRegex(RuntimeError, "unknown Slurm step"):
            live.steps("173697")

    def test_live_per_card_compute_process_blocks_launch(self):
        def run(argv, **kwargs):
            raw = ("0, GPU-a, NVIDIA RTX 4090 D, 12 MiB, 0 %\n"
                   if "--query-gpu=index,uuid,name,memory.used,utilization.gpu" in argv else
                   "GPU-a, 321, python, 100 MiB\n")
            return subprocess.CompletedProcess(argv, 0, raw, "")
        with self.assertRaisesRegex(RuntimeError, "live compute or occupancy"):
            launcher.LiveProbe(run).cards("node173697", "4090", [0], "173697")

    def test_slurm_gres_index_range_is_exact(self):
        self.assertEqual(launcher.gres_indices("gpu:4090:4(IDX:0-1,3,5)", "4090", 4),
                         [0, 1, 3, 5])
        with self.assertRaisesRegex(RuntimeError, "count differs"):
            launcher.gres_indices("gpu:4090:4(IDX:0-1,3)", "4090", 4)

    def test_node_ssh_denied_uses_exact_read_only_step(self):
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            if argv[0] == "ssh":
                raise RuntimeError("SSH authentication denied")
            response = {"job": "173698", "node": "node173698", "step_gpus": "2,3",
                "gpu": ("0, GPU-a, NVIDIA RTX 4090 D, 0 MiB, 0 %\n"
                        "1, GPU-b, NVIDIA RTX 4090 D, 0 MiB, 0 %\n"), "apps": ""}
            return subprocess.CompletedProcess(argv, 0, json.dumps(response), "")
        evidence = launcher.LiveProbe(run).cards("node173698", "4090", [2, 3], "173698")
        self.assertEqual(evidence["evidence_source"], "exact_srun_probe")
        self.assertEqual(evidence["slurm_step_gpus"], [2, 3])
        self.assertIn("--exact", calls[-1])
        self.assertNotIn("--overlap", calls[-1])

    def test_execute_records_intent_and_rejects_repeat(self):
        calls = []
        def fake_run(argv, **kwargs):
            calls.append(argv)
            self.probe.sessions_value.append(argv[4])
            return subprocess.CompletedProcess(argv, 0, "", "")
        result = self.plan(execute=True, run=fake_run)
        self.assertEqual(result["state"], "tmux_started_steps_need_identity_check")
        self.assertEqual(len(result["launched"]), 12)
        self.assertEqual(len(calls), 12)
        self.assertTrue(all(call[:3] == ["tmux", "new-session", "-d"] for call in calls))
        receipt = self.root / "v_workers" / "launch_test1.json"
        self.assertEqual(json.loads(receipt.read_text())["state"], result["state"])
        self.probe.sessions_value = []
        with self.assertRaisesRegex(RuntimeError, "already recorded"):
            self.plan(execute=True, run=fake_run)
        self.assertEqual(len(calls), 12)

    def test_worker_entry_requires_matching_slurm_step(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "wrong", "SLURM_STEP_ID": "1",
                                  "CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_NODELIST": "node"}):
            with self.assertRaisesRegex(RuntimeError, "specified Slurm GPU step"):
                launcher.worker_entry("173697", self.root / "id.json", self.dispatcher,
                                      self.queue, "4090D", self.python)


if __name__ == "__main__":
    unittest.main()
