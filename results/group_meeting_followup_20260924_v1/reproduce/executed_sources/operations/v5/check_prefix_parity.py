"""Verify that the first 30 rounds of V5 L1/L3 reproduce earlier trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile


SEEDS = (7, 13, 21, 42, 87)


def run(v5: Path, core: Path, v41: Path) -> dict:
    reports = []
    for family, job_prefix in (("L1", "L1_fed_alt_r100"),
                               ("L3", "L3_fed_both_mean_r100")):
        for seed in SEEDS:
            archive = v5 / "queue_v5a" / "archives" / f"{job_prefix}_s{seed}_attempt1.zip"
            old_metrics = (json.loads((core / f"seed_{seed}" / "full_alt" / "metrics.json").read_text())
                           if family == "L1" else None)
            fields = ("accuracy", "predictions", "clients") + (("adapter_parent",) if family == "L3" else ())
            equal = {field: 0 for field in fields}
            first_mismatch = {}
            with zipfile.ZipFile(archive) as z:
                for rd in range(1, 31):
                    new = json.loads(z.read(f"run/rounds/{rd:03d}/record.json"))
                    if family == "L1":
                        metric = old_metrics[rd - 1]
                        prediction = json.loads((core / f"seed_{seed}" / "full_alt" /
                                                 f"predictions_round_{rd:03d}.json").read_text())
                        matches = {
                            "accuracy": new["evaluation"]["accuracy"] == metric["accuracy"],
                            "predictions": new["evaluation"]["predictions"] == prediction,
                            "clients": new["selected_clients"] == metric["selected_clients"],
                        }
                    else:
                        ref = json.loads((v41 / f"fed_both_mean_s{seed}" / "run" / "rounds" /
                                          f"{rd:03d}" / "record.json").read_text())
                        matches = {
                            "accuracy": new["evaluation"]["accuracy"] == ref["evaluation"]["accuracy"],
                            "predictions": new["evaluation"]["predictions"] == ref["evaluation"]["predictions"],
                            "clients": new["selected_clients"] == ref["selected_clients"],
                            "adapter_parent": new["base_before_sha256"] == ref["base_before_sha256"],
                        }
                    for field, ok in matches.items():
                        if ok:
                            equal[field] += 1
                        else:
                            first_mismatch.setdefault(field, rd)
            reports.append({"family": family, "seed": seed, "rounds": 30,
                            "equal_rounds": equal, "first_mismatch": first_mismatch})
    return {"passed": all(not x["first_mismatch"] for x in reports), "checks": reports}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v5-root", type=Path, required=True)
    p.add_argument("--core-root", type=Path, required=True)
    p.add_argument("--v41-materialized", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = run(a.v5_root, a.core_root, a.v41_materialized)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "pairs": len(result["checks"])}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
