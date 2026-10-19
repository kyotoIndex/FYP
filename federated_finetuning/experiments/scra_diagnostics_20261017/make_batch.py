"""Generate the 2026-10-18 validation batch (README, "批量验证"): plans_batch/NN_<name>.json.

    python make_batch.py
Each plan is plan.json with the changes below; run roots are runs/scra_batch_20261018/<name>.
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = "runs/scra_batch_20261018"

AMP = {"aggregation": "hajek", "weak_amp_alpha": "auto"}
L2 = {"aggregation": "owner"}
VREF = {"aggregation": "hajek", "weak_train_full": True}
# order = priority: if the array is cut short, the main comparison finishes first
RUNS = [
    ("amp_s2040", 2040, AMP, {}),
    ("amp_s2041", 2041, AMP, {}),
    ("amp_s2042", 2042, AMP, {}),
    ("L2_s2041", 2041, L2, {}),
    ("L2_s2042", 2042, L2, {}),
    ("vref_s2041", 2041, VREF, {}),
    ("vref_s2042", 2042, VREF, {}),
    ("amp_owner_s2040", 2040, {**AMP, "aggregation": "owner"}, {}),
    ("amp_prefix_s2040", 2040, {**AMP, "allocation": "prefix"}, {}),
    ("amp_a2_s2040", 2040, {**AMP, "weak_amp_alpha": 2.0}, {}),
    ("amp_a8_s2040", 2040, {**AMP, "weak_amp_alpha": 8.0}, {}),
    ("amp_r1_s2040", 2040, AMP, {"weak_rank": 1}),
    ("L2_r1_s2040", 2040, L2, {"weak_rank": 1}),
    ("amp_het_s2040", 2040, AMP, {"weak_rank_choices": [1, 2, 4]}),
    ("vref_het_s2040", 2040, VREF, {"weak_rank_choices": [1, 2, 4]}),
    ("L2_het_s2040", 2040, L2, {"weak_rank_choices": [1, 2, 4]}),
    ("amp_sf20_s2040", 2040, {**AMP, "strong_fraction": 0.2}, {}),
    ("vref_sf20_s2040", 2040, {**VREF, "strong_fraction": 0.2}, {}),
    ("L2_sf20_s2040", 2040, {**L2, "strong_fraction": 0.2}, {}),
]


def main() -> None:
    base = json.loads((HERE / "plan.json").read_text())
    out = HERE / "plans_batch"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    for i, (name, seed, d0, lora) in enumerate(RUNS):
        plan = copy.deepcopy(base)
        plan["id"] = f"scra_batch_20261018_{name}"
        plan["run_root"] = f"{ROOT}/{name}"
        plan["d0"].update({"seed": seed, "shadow_strong_replay": False, "save_stats": False, **d0})
        plan["lora"].update(lora)
        (out / f"{i:02d}_{name}.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"wrote {len(RUNS)} plans to {out}")


if __name__ == "__main__":
    main()
