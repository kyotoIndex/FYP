"""Fixed-parent cohort bank execution; support construction is an input.

This module does not select masks or read diagnostic outcomes to choose JQ.
Callers must supply four already constructed masks per participant and a
pretraining JQ lock made from the same masks.  The executor deliberately uses
the v7 local_train/aggregate contracts, with injectable functions for CPU tests.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence


SUPPORTS = ("local", "cyclic", "JL_h32", "public512_linear")
JQ_CHOICES = SUPPORTS[:3]
PREFIXES = (1, 4, 16)
FORBIDDEN_ACCESS_TOKENS = ("bank", "outcome", "benefit", "readout", "selection1024", "16_step")


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _sha_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _seed(*parts: object) -> int:
    """Same canonical encoding as pcla_repair.core.keyed_seed, without torch."""
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def _same(a: Any, b: Any) -> bool:
    """Compare snapshots with tensor/array values without ambiguous truth values."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if hasattr(a, "equal") and callable(a.equal):
        return bool(a.equal(b))
    if hasattr(a, "shape") and hasattr(b, "shape"):
        import numpy as np
        return bool(np.array_equal(a, b))
    return bool(a == b)


def support_digest(support: Mapping[str, Mapping[str, Any]], *, factor: str) -> str:
    """Hash exact boolean masks (including their projection names and both factors)."""
    if factor not in ("a", "b") or not support:
        raise ValueError("invalid factor or empty support")
    canonical = {}
    active = 0
    for name in sorted(support):
        if not isinstance(name, str) or not name or set(support[name]) != {"a", "b"}:
            raise ValueError("support must include both masks for every projection")
        canonical[name] = {}
        lengths = set()
        for side in ("a", "b"):
            value = support[name][side]
            if hasattr(value, "detach"):
                value = value.detach().cpu()
                if str(value.dtype) != "torch.bool" or value.ndim != 1:
                    raise ValueError("support tensor must be a boolean rank vector")
                value = value.tolist()
            if not isinstance(value, (list, tuple)) or not value or any(type(x) is not bool for x in value):
                raise ValueError("support mask must be a nonempty boolean rank vector")
            lengths.add(len(value))
            canonical[name][side] = list(value)
            if side == factor:
                active += sum(value)
            elif any(value):
                raise ValueError("inactive factor has selected positions")
        if len(lengths) != 1:
            raise ValueError("factor mask ranks differ")
    if active == 0:
        raise ValueError("support has no active positions")
    return _sha_json(canonical)


@dataclass(frozen=True)
class JQLock:
    object_id: str
    h32_sha256: str
    locked_at_utc: str
    information_access: tuple[str, ...]
    choice_by_client: Mapping[int, str]
    support_sha256_by_client: Mapping[int, str]
    sha256: str


def make_jq_lock(*, object_id: str, participants: Sequence[int],
                 choice_by_client: Mapping[int, str],
                 support_sha256_by_client: Mapping[int, str],
                 h32_sha256: str, locked_at_utc: str,
                 information_access: Sequence[str]) -> JQLock:
    """Seal an externally chosen xi=1 JQ combination before bank training."""
    clients = tuple(participants)
    if (not object_id or len(clients) != 10 or len(set(clients)) != 10
            or set(choice_by_client) != set(clients)
            or set(support_sha256_by_client) != set(clients)):
        raise ValueError("JQ lock requires the fixed ten participants")
    try:
        timestamp = datetime.fromisoformat(locked_at_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("JQ lock needs a UTC timestamp") from exc
    if (any(choice not in JQ_CHOICES for choice in choice_by_client.values())
            or not _is_sha256(h32_sha256) or timestamp.tzinfo != timezone.utc
            or not information_access or not any("h32" in item.lower() for item in information_access)
            or any(token in item.lower() for item in information_access
                   for token in FORBIDDEN_ACCESS_TOKENS)):
        raise ValueError("JQ lock lacks valid pretraining h32 provenance")
    if any(not _is_sha256(digest) for digest in support_sha256_by_client.values()):
        raise ValueError("JQ support digest missing")
    record = {"schema": "pcla-v7-jq-xi1-lock-1", "object_id": object_id,
              "h32_sha256": h32_sha256, "locked_at_utc": locked_at_utc,
              "information_access": list(information_access),
              "choice_by_client": {str(cid): choice_by_client[cid] for cid in clients},
              "support_sha256_by_client": {str(cid): support_sha256_by_client[cid] for cid in clients}}
    return JQLock(object_id, h32_sha256, locked_at_utc, tuple(information_access),
                  dict(choice_by_client), dict(support_sha256_by_client), _sha_json(record))


def collect_cohort_bank(*, object_id: str, anchor_sha256: str,
                        anchor_identity: str, factor: str, participants: Sequence[int],
                        visits: Mapping[int, int], anchor_metrics: Sequence[Mapping[str, Any]],
                        parent_state: Mapping[str, Any],
                        datasets: Mapping[int, Any],
                        supports: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]],
                        jq_lock: JQLock, model: Any,
                        load_parent: Callable[[Any, Mapping[str, Any]], None],
                        local_train: Callable[..., Mapping[str, Any]],
                        aggregate: Callable[[Mapping[str, Any], list[Mapping[str, Any]], list[int]], Mapping[str, Any]],
                        capture_rng: Callable[[], Any], restore_rng: Callable[[Any], None],
                        cache: Any = None) -> dict[str, Any]:
    """Train four masks per client from one parent, then aggregate each prefix.

    The caller supplies anchor+1 participants and the verified checkpoint's
    metrics prefix; visits are checked against that history.  No support
    generation, h32 calculation, readout or queue
    work happens here.  On success or error the model and global RNG are
    restored; input cache and parent are never passed to training for mutation.
    """
    clients = tuple(participants)
    if (not object_id or not _is_sha256(anchor_sha256) or not anchor_identity
            or factor not in ("a", "b") or len(clients) != 10
            or len(set(clients)) != 10 or set(visits) != set(clients)
            or set(datasets) != set(clients) or set(supports) != set(clients)):
        raise ValueError("bank requires exactly ten fixed clients and their inputs")
    if any(type(visits[cid]) is not int or visits[cid] < 0 for cid in clients):
        raise ValueError("visits must be actual nonnegative counts")
    if len(anchor_metrics) not in (10, 60):
        raise ValueError("bank requires an anchor-10 or anchor-60 metrics prefix")
    actual_visits = {cid: 0 for cid in clients}
    for round_id, row in enumerate(anchor_metrics, 1):
        selected = row.get("selected_clients")
        if (row.get("round") != round_id or not isinstance(selected, list)
                or len(selected) != 10 or len(set(selected)) != 10):
            raise ValueError("anchor metrics participation history is invalid")
        for cid in selected:
            if cid in actual_visits:
                actual_visits[cid] += 1
    if dict(visits) != actual_visits:
        raise ValueError("visits differ from actual anchor participation history")
    if any(len(datasets[cid]) != 256 for cid in clients):
        raise ValueError("each client must provide its fixed 256 training rows")
    keys = set(parent_state)
    projection_names = {key.rsplit(".", 1)[0] for key in keys if key.endswith(".lora_a")}
    if (not {"classifier.weight", "classifier.bias"} <= keys
            or not any(key.endswith(".lora_a") for key in keys)
            or not any(key.endswith(".lora_b") for key in keys)):
        raise ValueError("parent must include adapter factors and classification head")
    if not isinstance(jq_lock, JQLock) or jq_lock.object_id != object_id:
        raise ValueError("a pretraining JQ lock for this object is required")
    digests: dict[int, dict[str, str]] = {}
    for cid in clients:
        if set(supports[cid]) != set(SUPPORTS):
            raise ValueError("all four bank supports must be preconstructed")
        digests[cid] = {name: support_digest(supports[cid][name], factor=factor)
                        for name in SUPPORTS}
        if any(set(supports[cid][name]) != projection_names for name in SUPPORTS):
            raise ValueError("bank support projections differ from the parent")
        choice = jq_lock.choice_by_client.get(cid)
        if (choice not in JQ_CHOICES
                or jq_lock.support_sha256_by_client.get(cid) != digests[cid][choice]):
            raise ValueError("JQ lock mask differs from the bank candidate")
    resealed = make_jq_lock(object_id=object_id, participants=clients,
                            choice_by_client=jq_lock.choice_by_client,
                            support_sha256_by_client=jq_lock.support_sha256_by_client,
                            h32_sha256=jq_lock.h32_sha256,
                            locked_at_utc=jq_lock.locked_at_utc,
                            information_access=jq_lock.information_access)
    if resealed.sha256 != jq_lock.sha256:
        raise ValueError("JQ lock contents were modified after sealing")

    parent = copy.deepcopy(parent_state)
    cache_before = copy.deepcopy(cache)
    rng = capture_rng()
    prefixes: dict[str, dict[int, dict[int, Mapping[str, Any]]]] = {name: {} for name in SUPPORTS}
    seeds = {cid: _seed(object_id, cid) for cid in clients}
    try:
        for cid in clients:
            for name in SUPPORTS:
                if support_digest(supports[cid][name], factor=factor) != digests[cid][name]:
                    raise ValueError("bank support changed after JQ was locked")
                load_parent(model, parent)
                result = local_train(model, datasets[cid], supports[cid][name],
                                     seed=seeds[cid], lr=.001, batch_size=16,
                                     epochs=1, betas=(.9, .999), eps=1e-8,
                                     weight_decay=.01, capture_steps=PREFIXES)
                if support_digest(supports[cid][name], factor=factor) != digests[cid][name]:
                    raise ValueError("bank support changed during training")
                states = result.get("prefix_states")
                if result.get("optimizer_steps") != 16 or not isinstance(states, Mapping) or set(states) != set(PREFIXES):
                    raise ValueError("local training did not capture one 16-step trajectory")
                if any(set(states[step]) != keys for step in PREFIXES):
                    raise ValueError("prefix lacks adapter or head parameters")
                prefixes[name][cid] = {step: copy.deepcopy(states[step]) for step in PREFIXES}
        if not _same(cache, cache_before) or not _same(parent_state, parent):
            raise ValueError("bank training mutated parent state or cache")
        aggregates = {}
        for name in (*SUPPORTS, "JQ_xi1"):
            aggregates[name] = {}
            for step in PREFIXES:
                children = [prefixes[jq_lock.choice_by_client[cid] if name == "JQ_xi1" else name][cid][step]
                            for cid in clients]
                aggregates[name][step] = aggregate(parent, children, [1] * 10)
                if set(aggregates[name][step]) != keys:
                    raise ValueError("aggregation omitted adapter or head parameters")
        if not _same(cache, cache_before) or not _same(parent_state, parent):
            raise ValueError("bank aggregation mutated parent state or cache")
        return {"schema": "pcla-v7-cohort-bank-1", "object_id": object_id,
                "anchor_sha256": anchor_sha256, "anchor_identity": anchor_identity,
                "factor": factor, "participants": list(clients), "visits": dict(visits),
                "training_seed_by_client": seeds, "support_sha256": digests,
                "jq_lock_sha256": jq_lock.sha256, "prefix_states": prefixes,
                "aggregate_states": aggregates}
    finally:
        try:
            load_parent(model, parent)
        finally:
            restore_rng(rng)


def collect_with_runtime(**kwargs: Any) -> dict[str, Any]:
    """Production adapter using existing v7 training/aggregation; imports torch lazily."""
    import random
    import numpy as np
    import torch
    from .runtime import aggregate, load_state, local_train

    def capture_rng() -> tuple[Any, Any, Any, Any]:
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        return random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), cuda

    def restore_rng(saved: tuple[Any, Any, Any, Any]) -> None:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
        if saved[3] is not None:
            torch.cuda.set_rng_state_all(saved[3])

    return collect_cohort_bank(**kwargs, load_parent=load_state,
                               local_train=local_train, aggregate=aggregate,
                               capture_rng=capture_rng, restore_rng=restore_rng)
