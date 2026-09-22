from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from common import atomic_json, read_json, require, sha256_file


def load(queue):
    queue = Path(queue).resolve()
    plan = read_json(queue / "plan.json")
    control = read_json(queue / "control.json")
    return queue, plan, control


def completed(queue, job_id):
    path = queue / "done" / f"{job_id}.json"
    require(path.exists(), f"required job is not complete: {job_id}")
    return read_json(path)


def phase1(queue, output):
    queue, plan, control = load(queue)
    required = ("parity_pcla_7", "parity_replay_random_rank_7",
                "diagnose_7_5", "diagnose_7_15")
    records = {job: completed(queue, job) for job in required}
    from common import extract_verified
    temp = Path(output).with_suffix(".materialized")
    require(not temp.exists(), "phase1 temporary output already exists")
    temp.mkdir(parents=True)
    try:
        checks = {}
        for job, record in records.items():
            extract_verified(record["archive"]["path"],
                             record["archive"]["sha256"], temp / job)
        for job in required[:2]:
            parity = read_json(temp / job / "result" / "source_parity.json")
            checks[job] = parity
            require(all(parity.values()), f"parity failed: {job}")
        for job in required[2:]:
            result = read_json(temp / job / "result" / "result.json")
            require(result["state"] == "completed" and
                    result["client_calls"] == 60 and
                    result["optimizer_steps"] == 960,
                    f"fixed-state smoke failed: {job}")
            summary = read_json(temp / job / "result" / "summary.json")
            for row in summary:
                diagnostics = row.get("aggregation", {})
                if row["support"] != "full":
                    require(diagnostics["function_match_relative_error"] <= 1e-8,
                            "function-norm matching failed")
            checks[job] = {"rows": len(summary), "state": "passed"}
        control["open_phase"] = max(int(control["open_phase"]), 2)
        control["history"].append({"phase": 2, "action": "opened_after_phase1",
                                   "utc": time.time()})
        atomic_json(queue / "control.json", control)
        report = {"state": "passed", "opened_phase": 2, "checks": checks}
        atomic_json(output, report)
        print(json.dumps(report, indent=2))
    finally:
        import shutil
        shutil.rmtree(temp, ignore_errors=True)


def phase_review(queue, phase, analysis, output):
    queue, plan, control = load(queue)
    analysis = Path(analysis)
    require(analysis.is_dir(), "analysis directory is missing")
    expected = [x["id"] for x in plan["jobs"] if x["phase"] <= phase]
    missing = [job for job in expected
               if not (queue / "done" / f"{job}.json").exists()]
    require(not missing, f"phase {phase} has incomplete jobs: {missing}")
    summary_file = (analysis / "fixed_state_gate_summary.json"
                    if phase == 2 else analysis / "trajectory_gate_summary.json")
    require(summary_file.is_file(), "required gate summary is missing")
    summary = read_json(summary_file)
    report = {
        "state": "review_required",
        "phase_completed": phase,
        "current_open_phase": control["open_phase"],
        "analysis_sha256": sha256_file(summary_file),
        "predeclared_summary": summary,
        "automatic_open": False,
    }
    atomic_json(output, report)
    print(json.dumps(report, indent=2))


def approve(queue, open_phase, review, decision, reason):
    queue, plan, control = load(queue)
    require(open_phase in (3, 4), "only phases 3 and 4 need approval")
    require(decision in ("continue", "stop"), "unknown decision")
    review = Path(review)
    require(review.is_file(), "review record is missing")
    expected_completed = open_phase - 1
    record = read_json(review)
    require(record.get("phase_completed") == expected_completed,
            "review does not correspond to the preceding phase")
    approval = {
        "open_phase": open_phase,
        "decision": decision,
        "reason": reason,
        "review_path": str(review.resolve()),
        "review_sha256": sha256_file(review),
        "utc": time.time(),
    }
    atomic_json(queue / "gates" / f"phase_{open_phase}_decision.json", approval)
    if decision == "continue":
        require(int(control["open_phase"]) == expected_completed,
                "queue phase changed since review")
        control["open_phase"] = open_phase
        control["history"].append({"phase": open_phase,
                                   "action": "opened_after_scientific_review",
                                   "reason": reason, "utc": time.time()})
        atomic_json(queue / "control.json", control)
    print(json.dumps(approval, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("phase1")
    cmd.add_argument("--queue", required=True)
    cmd.add_argument("--out", required=True)
    for name, phase in (("phase2", 2), ("phase3", 3)):
        cmd = sub.add_parser(name)
        cmd.add_argument("--queue", required=True)
        cmd.add_argument("--analysis", required=True)
        cmd.add_argument("--out", required=True)
        cmd.set_defaults(phase=phase)
    cmd = sub.add_parser("approve")
    cmd.add_argument("--queue", required=True)
    cmd.add_argument("--open-phase", type=int, required=True)
    cmd.add_argument("--review", required=True)
    cmd.add_argument("--decision", required=True)
    cmd.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.command == "phase1":
        phase1(args.queue, args.out)
    elif args.command in ("phase2", "phase3"):
        phase_review(args.queue, args.phase, args.analysis, args.out)
    else:
        approve(args.queue, args.open_phase, args.review,
                args.decision, args.reason)


if __name__ == "__main__":
    main()
