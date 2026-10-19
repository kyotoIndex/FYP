"""Generate batch 2 (positioning controls, roadmap_2026-10-18 stage A): plans_batch2/NN_<name>.json.

    python make_batch2.py
Each plan is plan.json with the changes below; run roots are runs/scra_batch2_20261018/<name>.
The reference runs (amp, vref, L2, r8 for seeds 2040-2042) come from batch 1.
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = "runs/scra_batch2_20261018"
SEEDS = [2040, 2041, 2042]

METHODS = [
    # ---- mechanism controls: each pair differs in one factor (FIX_BATCH2_CONTROLS_2026-10-18 §1-§2)
    # A1a: FSLoRA forward value, amp's gradient path, balanced + Hajek -> amp vs this isolates sketching the state
    ("fslora_st", {"aggregation": "hajek", "weak_forward": "fslora_st"}),
    # A1b: FSLoRA forward with its true gradient -> fslora_st vs this isolates gradient scale / Adam eps
    ("fslora_hajek", {"aggregation": "hajek", "weak_forward": "fslora"}),
    # A1c: same forward and allocation, zero-filled average -> fslora_hajek vs this isolates aggregation
    ("fslora_zerofill", {"aggregation": "zerofill", "weak_forward": "fslora"}),
    # A2: first-order twin of amplification: weak LoRA lr x R/r, server / R/r
    ("lrcomp", {"aggregation": "hajek", "weak_lr_compensation": "auto"}),
    # ---- external positioning (several factors differ; not mechanism evidence)
    # FSLoRA-style under our unified AdamW protocol: independent random selection, zero-filled average
    ("fslora_paper", {"aggregation": "zerofill", "allocation": "random", "weak_forward": "fslora"}),
    # Fed-PLoRA style: independent random selection, owner averaging, no amplification
    ("fedplora", {"aggregation": "owner", "allocation": "random"}),
    # FLASC-style: full local training, global top r/R entries uploaded by weak clients, zero-filled average
    ("flasc", {"aggregation": "zerofill", "weak_train_full": True, "weak_upload": "topk"}),
]


def runs():
    return [(f"{m}_s{s}", s, d0) for m, d0 in METHODS for s in SEEDS]


def main() -> None:
    base = json.loads((HERE / "plan.json").read_text())
    out = HERE / "plans_batch2"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    for i, (name, seed, d0) in enumerate(runs()):
        plan = copy.deepcopy(base)
        plan["id"] = f"scra_batch2_20261018_{name}"
        plan["run_root"] = f"{ROOT}/{name}"
        plan["d0"].update({"seed": seed, "shadow_strong_replay": False, "save_stats": False, **d0})
        (out / f"{i:02d}_{name}.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"wrote {len(runs())} plans to {out}")


if __name__ == "__main__":
    main()
