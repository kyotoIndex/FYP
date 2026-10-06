"""Immutable donor-count replay supports for the two resolved R branches.

This module never loads donor weights, chooses a representation, changes
coordinates, or trains a model. A runtime must call ``replay_support`` after
its own D/T coordinate work and before its real wire preview.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import re
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from pcla_repair.core import keyed_seed
from .data import PROTOCOL
from .selectors import module_order
from .validation_configuration import V_SEEDS


SCHEMA = "pcla-v7-quota-replay-donor-book-1"
RECIPE_REPRESENTATION = {"D011": "D", "D035": "T"}
IDENTITY_KEYS = {"donor_config_file_sha256", "source_manifest_sha256",
                 "data_manifest_sha256", "initial_state_sha256",
                 "raw_metrics_file_sha256"}
RECIPE_FIELDS = ("recipe_id", "method", "representation", "period", "rank", "lr",
                 "min_rounds", "max_rounds", "budget_id", "eval_role")


class DonorSuffixRequired(ValueError):
    """The replay needs an unrecorded donor round; no quota was substituted."""

    def __init__(self, requested_round: int, available_through: int):
        self.requested_round = requested_round
        self.available_through = available_through
        super().__init__(f"donor suffix required from round {available_through + 1} "
                         f"for replay round {requested_round}")


@dataclass(frozen=True)
class DonorQuotaBook:
    schema: str
    donor_job_id: str
    seed: int
    recipe: dict[str, Any]
    identities: dict[str, str]
    modules: tuple[str, ...]
    rank: int
    rounds: tuple[dict[str, Any], ...]
    donor_support_trace_sha256: str
    sha256: str


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _job_recipe(job: Mapping[str, Any], slot: int) -> tuple[int, dict[str, Any]]:
    if not isinstance(job, Mapping):
        raise ValueError("replay job must be an explicit V job")
    rid, seed = job.get("recipe_id"), job.get("seed")
    if (type(seed) is not int or seed not in V_SEEDS or rid not in RECIPE_REPRESENTATION or
            job.get("job_id") != f"V_S{slot:02d}_{rid}_s{seed}" or
            job.get("protocol") != PROTOCOL or job.get("phase") != "V" or
            job.get("method") != "local" or
            job.get("representation") != RECIPE_REPRESENTATION[rid] or
            job.get("period") != 5 or job.get("rank") != 16 or
            job.get("eval_role") != "recipe_dev" or job.get("budget_id") != "B8" or
            job.get("min_rounds") != 100 or job.get("max_rounds") != 200 or
            not isinstance(job.get("lr"), (int, float)) or job["lr"] <= 0):
        raise ValueError("replay donor/job is not the resolved R-local V recipe")
    return seed, {field: job[field] for field in RECIPE_FIELDS}


def _modules(names: Sequence[str]) -> tuple[str, ...]:
    items = tuple(module_order(names))
    if (len(items) != 24 or len(set(items)) != 24 or
            any(not isinstance(name, str) or not name or
                not name.endswith((".query", ".value")) for name in items) or
            sum(name.endswith(".query") for name in items) != 12 or
            sum(name.endswith(".value") for name in items) != 12 or
            {name[:-5] + "value" for name in items if name.endswith(".query")} !=
            {name for name in items if name.endswith(".value")}):
        raise ValueError("donor must name all 24 canonical Q/V projections")
    return items


def _rank_list(value: Any, rank: int, context: str) -> list[int]:
    if (not isinstance(value, list) or
            any(type(item) is not int or item < 0 or item >= rank for item in value) or
            value != sorted(set(value))):
        raise ValueError(f"invalid donor rank IDs: {context}")
    return value


def _record(book: DonorQuotaBook) -> dict[str, Any]:
    return {"schema": book.schema, "donor_job_id": book.donor_job_id,
            "seed": book.seed, "recipe": book.recipe,
            "identities": book.identities, "modules": book.modules,
            "rank": book.rank, "rounds": book.rounds,
            "donor_support_trace_sha256": book.donor_support_trace_sha256}


def freeze_donor_trace(*, donor_job: Mapping[str, Any],
                       rows: Sequence[Mapping[str, Any]], module_names: Sequence[str],
                       capacities: Mapping[int, int], identities: Mapping[str, str]) -> DonorQuotaBook:
    """Verify committed support rows and seal only their per-module counts.

    ``raw_metrics_file_sha256`` identifies the original complete metrics file.
    Rows must be its committed round records, contiguous from r1. Donor IDs are
    validated and hashed, but only counts enter the replay plan.
    """
    seed, recipe = _job_recipe(donor_job, 3)
    names = _modules(module_names)
    if (set(identities) != IDENTITY_KEYS or
            any(not _valid_sha(identities[key]) for key in IDENTITY_KEYS) or
            set(capacities) != set(range(50)) or
            any(type(capacities[cid]) is not int or capacities[cid] not in {96, 128, 192}
                for cid in range(50)) or
            not rows or len(rows) > 200):
        raise ValueError("donor file/data/initialization identity or quota table missing")
    normalized, source_trace = [], []
    for expected_round, row in enumerate(rows, start=1):
        if type(row.get("round")) is not int or row["round"] != expected_round:
            raise ValueError(f"donor round sequence missing or duplicated at r{expected_round}")
        cohort = row.get("selected_clients")
        expected_cohort = random.Random(keyed_seed(seed, "participants", expected_round)).sample(
            list(range(50)), 10)
        if cohort != expected_cohort:
            raise ValueError(f"donor participant identity differs at r{expected_round}")
        trace = row.get("support_trace")
        if not isinstance(trace, Mapping) or set(trace) != {str(cid) for cid in cohort}:
            raise ValueError(f"donor support trace missing clients at r{expected_round}")
        factor = "b" if expected_round % 2 else "a"
        quota_by_client = {}
        for cid in cohort:
            entry = trace[str(cid)]
            if (not isinstance(entry, Mapping) or entry.get("active_factor") != factor or
                    entry.get("assigned_quota") != capacities[cid] or
                    type(entry.get("selected_positions")) is not int or
                    set(entry.get("modules", {})) != set(names)):
                raise ValueError(f"donor support identity/quota differs at r{expected_round}/c{cid}")
            counts, total = {}, 0
            for name in names:
                pair = entry["modules"][name]
                if not isinstance(pair, Mapping) or set(pair) != {"a", "b"}:
                    raise ValueError(f"donor module/factor omitted: r{expected_round}/c{cid}/{name}")
                active = _rank_list(pair[factor], 16, f"r{expected_round}/c{cid}/{name}/{factor}")
                inactive = _rank_list(pair["a" if factor == "b" else "b"], 16,
                                      f"r{expected_round}/c{cid}/{name}/inactive")
                if inactive:
                    raise ValueError("R-local donor unexpectedly updates inactive factor")
                counts[name] = len(active)
                total += len(active)
            if total != capacities[cid] or total != entry["selected_positions"]:
                raise ValueError(f"donor total does not spend fixed quota: r{expected_round}/c{cid}")
            quota_by_client[str(cid)] = counts
        normalized.append({"round": expected_round, "selected_clients": cohort,
                           "factor": factor, "counts": quota_by_client})
        source_trace.append({"round": expected_round, "selected_clients": cohort,
                             "support_trace": trace})
    record = {"schema": SCHEMA, "donor_job_id": donor_job["job_id"], "seed": seed,
              "recipe": recipe, "identities": dict(identities), "modules": names,
              "rank": 16, "rounds": tuple(normalized),
              "donor_support_trace_sha256": _hash(source_trace)}
    return DonorQuotaBook(**record, sha256=_hash(record))


def rank_ids(*, seed: int, round_id: int, client: int, module: str,
             factor: str, count: int, rank: int) -> list[int]:
    """Exact Pro ``quota_replay_reference.rank_ids`` canonical draw."""
    if any(type(value) is not int for value in (seed, round_id, client, count, rank)):
        raise ValueError("integer identity and quota required")
    if round_id < 1 or client < 0 or not module or factor not in ("a", "b"):
        raise ValueError("invalid replay identity")
    if rank <= 0 or not 0 <= count <= rank:
        raise ValueError("quota outside rank capacity")
    identity = [PROTOCOL, "quota_replay_v1", seed, round_id, client, module, factor]
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))
    return sorted(rng.sample(range(rank), count))


def replay_support(*, book: DonorQuotaBook, replay_job: Mapping[str, Any],
                   round_id: int, cohort: Sequence[int], module_ranks: Mapping[str, int],
                   expected_identities: Mapping[str, str]) -> dict[int, dict[str, dict[str, Tensor]]]:
    """Return masks from sealed counts, rejecting wrong seed or absent suffix."""
    if (not isinstance(book, DonorQuotaBook) or book.schema != SCHEMA or
            book.sha256 != _hash(_record(book)) or
            not _valid_sha(book.donor_support_trace_sha256)):
        raise ValueError("donor quota book hash/schema differs")
    seed, recipe = _job_recipe(replay_job, 16)
    if (seed != book.seed or recipe != book.recipe or
            book.donor_job_id != f"V_S03_{recipe['recipe_id']}_s{seed}" or
            set(expected_identities) != IDENTITY_KEYS or
            dict(expected_identities) != book.identities or
            set(module_ranks) != set(book.modules) or
            any(type(module_ranks[name]) is not int or module_ranks[name] != book.rank
                for name in book.modules) or
            type(round_id) is not int or round_id < 1 or round_id > 200):
        raise ValueError("replay seed/recipe/initialization/data/module identity differs")
    if round_id > len(book.rounds):
        raise DonorSuffixRequired(round_id, len(book.rounds))
    row = book.rounds[round_id - 1]
    if (row["round"] != round_id or list(cohort) != row["selected_clients"] or
            row["factor"] != ("b" if round_id % 2 else "a") or
            set(row["counts"]) != {str(cid) for cid in cohort}):
        raise ValueError("replay round/cohort/factor differs from donor trace")
    factor = row["factor"]
    supports = {}
    for cid in cohort:
        counts = row["counts"][str(cid)]
        if set(counts) != set(book.modules):
            raise ValueError("donor per-module counts incomplete")
        support, total = {}, 0
        for name in book.modules:
            count = counts[name]
            ids = rank_ids(seed=seed, round_id=round_id, client=cid, module=name,
                           factor=factor, count=count, rank=book.rank)
            active = torch.zeros(book.rank, dtype=torch.bool)
            active[ids] = True
            support[name] = {"a": active if factor == "a" else torch.zeros_like(active),
                             "b": active if factor == "b" else torch.zeros_like(active)}
            total += len(ids)
        if total not in {96, 128, 192}:
            raise ValueError("replay support did not preserve donor client quota")
        supports[cid] = support
    return supports
