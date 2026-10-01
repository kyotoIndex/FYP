"""The pair controller may launch only after one evidenced GPU step drains."""
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/continue_pcla_v7_drained_pairs.py"
SPEC = importlib.util.spec_from_file_location("pair_continuation", SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def fixture(tmp_path):
    root = tmp_path / "stage"
    dispatcher = root / "repo/scripts/dispatch_pcla_v7.py"
    dispatcher.parent.mkdir(parents=True)
    dispatcher.write_text("frozen queue dispatcher\n")
    python = "/frozen/venv/bin/python"
    entries = {}
    for allocation, steps in mod.OLD.items():
        for step in steps:
            old_id = f"{allocation}.{step}"
            entries[old_id] = {"session": f"pclav7_pairrefresh_{allocation}_{step}",
                               "argv": ["srun", f"--jobid={allocation}", "--nodes=1", "--ntasks=1",
                                        "--gpus-per-task=1", "--cpus-per-task=8", "--exact", "--cpu-bind=cores",
                                        python, str(dispatcher), "--queue", str(root / "queue"),
                                        "pair", "--pool", "4090D_shared", "--python", python]}
    receipt = root / "workers/pair_refresh_core02.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"replacements": entries}))
    return root, receipt, root / "workers/pair_continuation_v1.json", entries


class FakeProbe:
    user = "researcher"
    host = "manager"

    def __init__(self, entries):
        self.entries = entries
        self.mapping = {old_id: f"{old_id.split('.')[0]}.{238 + i}"
                        for i, old_id in enumerate(entries)}
        self.pid = {old_id: 1000 + i for i, old_id in enumerate(entries)}
        self.active_old = set(entries)
        self.live_steps = set(self.mapping.values())
        self.running_steps = set()
        self.shared_pending = ["D010_s419"]
        self.exclusive_pending = ["D010_s401"]
        self.owner = "researcher"
        self.allocation_state = "RUNNING"
        self.sessions = set()
        self.launches = []
        self.fail_launch = False

    def allocation(self, allocation):
        return {"state": self.allocation_state, "owner": self.owner,
                "gpus": mod.GPUS[allocation], "node": "gpu"}

    def steps(self, allocation):
        return [{"step_id": step_id, "srun_host": self.host, "srun_pid": self.pid[old_id],
                 "state": "RUNNING", "tres": "gres/gpu=1"}
                for old_id, step_id in self.mapping.items()
                if old_id.startswith(allocation + ".") and step_id in self.live_steps]

    def session(self, name, expected_argv):
        old_id = next(key for key, value in self.entries.items() if value["session"] == name)
        if old_id not in self.active_old:
            return None
        assert expected_argv == self.entries[old_id]["argv"]
        return {"pane_pid": self.pid[old_id] - 100, "srun_pid": self.pid[old_id]}

    def queue(self):
        return {"shared_pending": self.shared_pending[:],
                "exclusive_pending": self.exclusive_pending[:],
                "running_steps": sorted(self.running_steps), "uncertain_running": []}

    def session_exists(self, name):
        return name in self.sessions

    def launch(self, session, argv, log):
        self.launches.append((session, argv, log))
        if self.fail_launch:
            raise RuntimeError("injected tmux failure")
        self.sessions.add(session)


def one(root, receipt, state, probe, execute=False):
    return mod.check_once(root, receipt, state, probe, execute=execute)


def drain(probe, old_id):
    probe.active_old.remove(old_id)
    probe.live_steps.remove(probe.mapping[old_id])


def test_live_pair_and_shared_pending_never_launch(tmp_path):
    root, receipt, state, entries = fixture(tmp_path)
    probe = FakeProbe(entries)
    old_id = next(iter(entries))
    result = one(root, receipt, state, probe, execute=True)
    assert result["reasons"][old_id] == "original_pair_still_alive"
    assert not probe.launches
    assert mod.read(state)["entries"][old_id]["step_id"] == probe.mapping[old_id]
    probe.active_old.remove(old_id)
    result = one(root, receipt, state, probe, execute=True)
    assert result["reasons"][old_id] == "original_step_still_alive"
    probe.live_steps.remove(probe.mapping[old_id])
    result = one(root, receipt, state, probe, execute=True)
    assert result["reasons"][old_id] == "shared_pending"
    assert not probe.launches


def test_running_allocation_and_pool_gates(tmp_path):
    root, receipt, state, entries = fixture(tmp_path)
    probe = FakeProbe(entries)
    old_id = next(iter(entries))
    one(root, receipt, state, probe)
    drain(probe, old_id)
    probe.shared_pending = []
    probe.running_steps.add(probe.mapping[old_id])
    assert one(root, receipt, state, probe, True)["reasons"][old_id] == "queue_running_or_uncertain"
    probe.running_steps.clear()
    probe.owner = "someone-else"
    assert one(root, receipt, state, probe, True)["reasons"][old_id] == "allocation_not_valid"
    probe.owner = probe.user
    probe.allocation_state = "COMPLETED"
    assert one(root, receipt, state, probe, True)["reasons"][old_id] == "allocation_not_valid"
    probe.allocation_state = "RUNNING"
    probe.exclusive_pending = []
    assert one(root, receipt, state, probe, True)["reasons"][old_id] == "exclusive_pool_empty"
    assert not probe.launches


def test_one_drained_pair_starts_one_exclusive_and_is_idempotent(tmp_path):
    root, receipt, state, entries = fixture(tmp_path)
    probe = FakeProbe(entries)
    old_id = next(iter(entries))
    one(root, receipt, state, probe)  # dry-run records exact live step/PID map
    drain(probe, old_id)
    probe.shared_pending = []
    result = one(root, receipt, state, probe, execute=True)
    assert result["reasons"][old_id] == "tmux_started_step_unverified"
    assert len(probe.launches) == 1
    session, argv, log = probe.launches[0]
    assert "--cpus-per-task=4" in argv and "--gpus-per-task=1" in argv
    assert argv[-5:] == ["4090D", "--python", "/frozen/venv/bin/python", "--mode", "exclusive"]
    assert "A100" not in argv and "4090D_single" not in argv
    assert session.endswith(old_id.replace(".", "_"))
    assert log.parent == root / "workers"
    assert one(root, receipt, state, probe, execute=True)["reasons"][old_id] == "already_launched"
    assert len(probe.launches) == 1
    history = mod.read(state)["history"]
    assert [x["event"] for x in history if x["old_step"] == old_id] == [
        "mapped_live_step", "launch_intent", "tmux_started"]


def test_crash_after_intent_fails_closed_on_restart(tmp_path):
    root, receipt, state, entries = fixture(tmp_path)
    probe = FakeProbe(entries)
    old_id = next(iter(entries))
    one(root, receipt, state, probe)
    drain(probe, old_id)
    probe.shared_pending = []
    probe.fail_launch = True
    with pytest.raises(RuntimeError, match="injected"):
        one(root, receipt, state, probe, execute=True)
    assert mod.read(state)["entries"][old_id]["phase"] == "launching"
    with pytest.raises(RuntimeError, match="manual inspection"):
        one(root, receipt, state, probe, execute=True)
    assert len(probe.launches) == 1


def test_unmapped_disappeared_pair_is_ambiguous(tmp_path):
    root, receipt, state, entries = fixture(tmp_path)
    probe = FakeProbe(entries)
    old_id = next(iter(entries))
    drain(probe, old_id)
    with pytest.raises(RuntimeError, match="before step mapping"):
        one(root, receipt, state, probe)
    assert mod.read(state)["entries"][old_id]["phase"] == "ambiguous"
    with pytest.raises(RuntimeError, match="manual inspection"):
        one(root, receipt, state, probe, execute=True)
