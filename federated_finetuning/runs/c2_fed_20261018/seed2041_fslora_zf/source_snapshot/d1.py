"""D1: data difference vs training-mode effect on fixed global states (design §5.6, §7.1).

For each snapshot (from D0) and each selected client, with one fixed batch schedule:
  full       : train all components (20 steps)                                -> delta_full
  restricted : 28 runs; run i trains, in adapter j, the pair perm_j[i] of the  -> E_S[delta_k(S) | k in S]
               C(R,r) pairs (Latin design: every adapter sees every pair once,
               pairs differ across adapters within a run, as in real training)
Groups: strong (all strong clients), core weak (non-strong domains), demoted weak (strong-domain weak).
Saves group means of both modes and per-client scalars to <run_root>/d1/round_XXXX.pt.
"""
from __future__ import annotations

import itertools
from pathlib import Path
import random

import torch

from .core import adapter_names, atomic_save, local_train, to_blocks
from .data import ClientStream, Dataset, derived_seed, override_roles
from .model import TaskModel


def select_clients(data: Dataset, strong: list[int], per_core_domain: int, demoted_per_domain: int):
    strong_domains = sorted({data.clients[c]["domain"] for c in strong})
    by_domain: dict[str, list[int]] = {}
    for c in sorted(data.clients):
        if c in strong:
            continue
        by_domain.setdefault(data.clients[c]["domain"], []).append(c)
    core = [c for d in sorted(by_domain) if d not in strong_domains for c in by_domain[d][:per_core_domain]]
    demoted = [c for d in strong_domains for c in by_domain.get(d, [])[:demoted_per_domain]]
    return {"strong": list(strong), "core": core, "demoted": demoted}


def run_d1(plan: dict, snapshot_round: int, output: Path, device: torch.device) -> Path:
    cfg, d1 = plan["train"], plan["d1"]
    out_path = output / f"round_{snapshot_round:04d}.pt"
    if out_path.exists():
        print(f"{out_path} exists; skipping", flush=True)
        return out_path
    torch.set_num_threads(plan.get("cpu_threads", 4))
    data = Dataset(Path(plan["cache_directory"]), Path(plan["manifest_path"]))
    strong = override_roles(data.clients, plan["d0"]["strong_fraction"])
    R, r = plan["lora"]["full_rank"], plan["lora"]["weak_rank"]
    model = TaskModel.build(plan["model"], plan["num_labels"], R, plan["lora"]["scale"], plan["d0"]["seed"]).to(device)
    names = adapter_names(model)
    snap = Path(plan["run_root"]) / "d0" / "snapshots" / f"round_{snapshot_round:04d}.pt"
    state = torch.load(snap, map_location="cpu", weights_only=True)
    g_blocks = to_blocks(state, names)
    J, _, D = g_blocks.shape
    pairs = list(itertools.combinations(range(R), r))
    perms = []
    for j in range(J):
        order = list(range(len(pairs)))
        random.Random(derived_seed(d1["seed"], "latin", snapshot_round, j)).shuffle(order)
        perms.append(order)
    groups = select_clients(data, strong, d1["core_per_domain"], d1["demoted_per_domain"])
    per_client = {}
    sums = {g: {"full": torch.zeros(J, R, D), "restr": torch.zeros(J, R, D), "n": 0} for g in groups}
    for group, members in groups.items():
        for c in members:
            def stream():
                return ClientStream(data.clients[c]["train_indices"], d1["seed"] + snapshot_round, c, 0)
            full_state, _ = local_train(model, state, data, stream(), None, cfg, device)
            full = to_blocks(full_state, names) - g_blocks
            acc = torch.zeros(J, R, D); cnt = torch.zeros(J, R)
            for i in range(len(pairs)):
                sets = [list(pairs[perms[j][i]]) for j in range(J)]
                restr_state, _ = local_train(model, state, data, stream(), sets, cfg, device)
                restr = to_blocks(restr_state, names) - g_blocks
                mask = torch.zeros(J, R)
                for j, chosen in enumerate(sets):
                    mask[j, chosen] = 1.0
                acc += restr * mask[..., None]; cnt += mask
            restr_mean = acc / cnt.clamp_min(1)[..., None]
            m = restr_mean - full
            per_client[c] = {
                "group": group,
                "cos_restr_full": torch.nn.functional.cosine_similarity(restr_mean, full, dim=-1),
                "norm_ratio": restr_mean.norm(dim=-1) / full.norm(dim=-1).clamp_min(1e-12),
                "m_rel": m.norm(dim=-1) / full.norm(dim=-1).clamp_min(1e-12),
                "full_sq": full.pow(2).sum(-1),
            }
            sums[group]["full"] += full; sums[group]["restr"] += restr_mean; sums[group]["n"] += 1
            print(f"[D1 r{snapshot_round}] client {c} ({group}) done: median |m|/|full| = "
                  f"{per_client[c]['m_rel'].median():.3f}", flush=True)
    means = {g: {"full": v["full"] / max(v["n"], 1), "restr": v["restr"] / max(v["n"], 1), "n": v["n"]}
             for g, v in sums.items()}
    population = {"strong": len(strong), "weak": len(data.clients) - len(strong),
                  "core_total": sum(1 for c in data.clients if c not in strong and
                                    data.clients[c]["domain"] not in {data.clients[s]["domain"] for s in strong})}
    atomic_save(out_path, {"snapshot_round": snapshot_round, "groups": groups, "means": means,
                           "per_client": per_client, "population": population, "adapter_names": names,
                           "R": R, "r": r})
    return out_path
