"""Fixed-parent, read-only measurements of an already trained cohort bank.

The caller supplies the real model loss and fixed-denominator aggregation
functions.  This module never trains a new mask and never uses readout2048 to
select a combination.  The bank is the output of diagnostic_bank.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

from .diagnostic_bank import JQLock, PREFIXES, SUPPORTS


SELECTION_CHOICES = SUPPORTS
MEASURED_CHOICES = (*SUPPORTS, "JQ_xi1", "finite_bank_search")


@dataclass(frozen=True)
class DiagnosticView:
    role: str
    dataset: Any
    source_id: str
    row_ids: tuple[int, ...]


@dataclass(frozen=True)
class SelectionLock:
    object_id: str
    anchor_sha256: str
    bank_factor: str
    prefix: int
    selection_source_id: str
    selection_row_ids: tuple[int, ...]
    selection_row_sha256: str
    choices: tuple[tuple[int, str], ...]
    evaluations: int
    sha256: str


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                         separators=(",", ":")).encode()).hexdigest()


def _validate_view(view: DiagnosticView, role: str, size: int) -> None:
    if (not isinstance(view, DiagnosticView) or view.role != role or
            not view.source_id or len(view.dataset) != size or
            len(view.row_ids) != size or len(set(view.row_ids)) != size or
            any(type(row) is not int or row < 0 for row in view.row_ids)):
        raise ValueError(f"{role} requires {size} unique rows and a source identity")


def _validate_bank(bank: Mapping[str, Any]) -> tuple[int, ...]:
    if bank.get("schema") != "pcla-v7-cohort-bank-1" or bank.get("factor") not in {"a", "b"}:
        raise ValueError("readout requires a completed cohort bank")
    clients = tuple(bank.get("participants", ()))
    if len(clients) != 10 or len(set(clients)) != 10 or any(type(cid) is not int for cid in clients):
        raise ValueError("readout requires the bank's fixed ten-client order")
    if set(bank.get("prefix_states", {})) != set(SUPPORTS) or set(bank.get("aggregate_states", {})) != set((*SUPPORTS, "JQ_xi1")):
        raise ValueError("bank supports or aggregate states are incomplete")
    for support in SUPPORTS:
        if set(bank["prefix_states"][support]) != set(clients):
            raise ValueError("bank client states are incomplete")
        for cid in clients:
            if set(bank["prefix_states"][support][cid]) != set(PREFIXES):
                raise ValueError("bank prefixes are incomplete")
    for support in (*SUPPORTS, "JQ_xi1"):
        if set(bank["aggregate_states"][support]) != set(PREFIXES):
            raise ValueError("bank aggregate prefixes are incomplete")
    return clients


def _children(bank: Mapping[str, Any], clients: tuple[int, ...],
              choices: Mapping[int, str], prefix: int) -> list[Mapping[str, Any]]:
    if prefix not in PREFIXES or set(choices) != set(clients) or any(
            choice not in SUPPORTS for choice in choices.values()):
        raise ValueError("choice must use actual bank candidates for all ten clients")
    return [bank["prefix_states"][choices[cid]][cid][prefix] for cid in clients]


def _aggregate(parent: Mapping[str, Any], children: Sequence[Mapping[str, Any]],
               aggregate: Callable[..., Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(children) != 10 or any(set(child) != set(parent) for child in children):
        raise ValueError("aggregate needs ten complete states")
    result = aggregate(parent, list(children), [1] * 10)
    if set(result) != set(parent):
        raise ValueError("real aggregate omitted parent parameters")
    import torch
    for key, base in parent.items():
        independent = base.double() + sum(
            ((child[key].double() - base.double()) * .1 for child in children),
            torch.zeros_like(base, dtype=torch.float64))
        if not torch.allclose(result[key].double(), independent, atol=1e-7, rtol=1e-6):
            raise ValueError(f"real aggregate differs from fixed ten-client denominator: {key}")
    return result


def _loss(loss: Callable[[Mapping[str, Any], Any], float],
          state: Mapping[str, Any], view: DiagnosticView) -> float:
    result = float(loss(state, view.dataset))
    if not math.isfinite(result):
        raise ValueError("model loss is nonfinite")
    return result


def _same_state(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    import torch
    return set(actual) == set(expected) and all(
        torch.allclose(actual[key], expected[key], atol=1e-7, rtol=1e-6)
        for key in actual)


def _verify_single_weight(parent: Mapping[str, Any], child: Mapping[str, Any],
                          weighted: Mapping[str, Any]) -> None:
    import torch
    if not _same_state(weighted, {key: parent[key] + .1 * (child[key] - parent[key])
                                  for key in parent}):
        raise ValueError("single-client aggregate differs from fixed .1 delta")


def _projection_names(parent: Mapping[str, Any]) -> tuple[str, ...]:
    names = tuple(sorted(key[:-7] for key in parent if key.endswith(".lora_a")))
    if (len(names) != 24 or any(f"{name}.lora_b" not in parent for name in names) or
            not {"classifier.weight", "classifier.bias"} <= set(parent)):
        raise ValueError("parent requires 24 LoRA Q/V modules and the classification head")
    return names


def _effective_delta(parent: Mapping[str, Any], child: Mapping[str, Any],
                     name: str):
    return (child[f"{name}.lora_b"].double() @ child[f"{name}.lora_a"].double() -
            parent[f"{name}.lora_b"].double() @ parent[f"{name}.lora_a"].double())


def cancellation_and_linearity(parent: Mapping[str, Any],
                               children: Sequence[Mapping[str, Any]],
                               aggregated: Mapping[str, Any]) -> dict[str, Any]:
    """Compute 24-module effective-matrix cancellation with the real denominator."""
    import torch
    names = _projection_names(parent)
    if len(children) != 10:
        raise ValueError("ten client deltas are required")
    numerator = 0.0
    denominator = 0.0
    worst_relative_error = 0.0
    for name in names:
        deltas = [_effective_delta(parent, child, name) for child in children]
        combined = sum((.1 * delta for delta in deltas), torch.zeros_like(deltas[0]))
        actual = _effective_delta(parent, aggregated, name)
        error = torch.linalg.vector_norm(actual - combined).item()
        scale = max(torch.linalg.vector_norm(combined).item(), 1e-12)
        worst_relative_error = max(worst_relative_error, error / scale)
        if error > 1e-6 * scale + 1e-7:
            raise ValueError(f"effective aggregate differs from independent deltaM recomputation: {name}")
        numerator += torch.sum(combined.square()).item()
        denominator += .1 * sum(torch.sum(delta.square()).item() for delta in deltas)
    value = None if denominator == 0 else 1.0 - numerator / denominator
    return {"cancellation": value, "cancellation_status":
            "NA_zero_denominator" if value is None else "measured",
            "numerator_effective_norm_sq": numerator,
            "denominator_mean_effective_norm_sq": denominator,
            "max_effective_linearity_relative_error": worst_relative_error}


def finite_bank_selection(*, bank: Mapping[str, Any], parent: Mapping[str, Any],
                          selection: DiagnosticView,
                          loss: Callable[[Mapping[str, Any], Any], float],
                          aggregate: Callable[..., Mapping[str, Any]],
                          prefix: int = 16) -> SelectionLock:
    """Two ordered passes, four actual choices/client, <=80 whole-cohort calls.

    Only selection1024 is passed to ``loss``. Equal-loss ties keep the current
    candidate, and all-JL is the immutable starting combination.
    """
    clients = _validate_bank(bank)
    _validate_view(selection, "selection1024", 1024)
    if prefix not in PREFIXES:
        raise ValueError("selection prefix is not in the trained bank")
    choices = {cid: "JL_h32" for cid in clients}
    cache: dict[tuple[str, ...], float] = {}

    def evaluate() -> float:
        key = tuple(choices[cid] for cid in clients)
        if key not in cache:
            if len(cache) >= 80:
                raise RuntimeError("finite search exceeded 80 whole-cohort evaluations")
            children = _children(bank, clients, choices, prefix)
            state = _aggregate(parent, children, aggregate)
            cache[key] = _loss(loss, state, selection)
        return cache[key]

    best = evaluate()
    for _pass in range(2):
        for cid in clients:
            retained = choices[cid]
            winner = retained
            winner_loss = best
            for candidate in SUPPORTS:
                choices[cid] = candidate
                score = evaluate()
                if score < winner_loss:
                    winner, winner_loss = candidate, score
            choices[cid] = winner
            best = winner_loss
    ordered = tuple((cid, choices[cid]) for cid in clients)
    record = {"object_id": bank["object_id"], "anchor_sha256": bank["anchor_sha256"],
              "bank_factor": bank["factor"], "prefix": prefix,
              "selection_source_id": selection.source_id,
              "selection_row_ids": selection.row_ids,
              "selection_row_sha256": _digest(selection.row_ids),
              "choices": ordered, "evaluations": len(cache)}
    return SelectionLock(**record, sha256=_digest(record))


def _verify_lock(bank: Mapping[str, Any], lock: SelectionLock,
                 clients: tuple[int, ...]) -> dict[int, str]:
    if not isinstance(lock, SelectionLock):
        raise ValueError("readout2048 requires a locked selection1024 search")
    record = {name: getattr(lock, name) for name in (
        "object_id", "anchor_sha256", "bank_factor", "prefix", "selection_source_id",
        "selection_row_ids", "selection_row_sha256", "choices", "evaluations")}
    choices = dict(lock.choices)
    if (lock.sha256 != _digest(record) or lock.object_id != bank["object_id"] or
            lock.anchor_sha256 != bank["anchor_sha256"] or lock.bank_factor != bank["factor"] or
            tuple(cid for cid, _ in lock.choices) != clients or
            not lock.selection_source_id or len(lock.selection_row_ids) != 1024 or
            len(set(lock.selection_row_ids)) != 1024 or
            any(type(row) is not int or row < 0 for row in lock.selection_row_ids) or
            lock.selection_row_sha256 != _digest(lock.selection_row_ids) or
            lock.prefix not in PREFIXES or not 1 <= lock.evaluations <= 80 or
            len(choices) != 10 or any(choice not in SUPPORTS for choice in choices.values())):
        raise ValueError("finite bank selection lock is invalid for this bank")
    return choices


def measure_bank_readouts(*, bank: Mapping[str, Any], parent: Mapping[str, Any],
                          local64: Mapping[int, DiagnosticView],
                          readout: DiagnosticView, selection_lock: SelectionLock,
                          jq_lock: JQLock,
                          loss: Callable[[Mapping[str, Any], Any], float],
                          aggregate: Callable[..., Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate three real-model gains plus read-only head/LoRA projections."""
    clients = _validate_bank(bank)
    search_choices = _verify_lock(bank, selection_lock, clients)
    if (not isinstance(jq_lock, JQLock) or jq_lock.sha256 != bank.get("jq_lock_sha256") or
            jq_lock.object_id != bank["object_id"] or
            set(jq_lock.choice_by_client) != set(clients) or
            set(jq_lock.support_sha256_by_client) != set(clients) or
            any(jq_lock.support_sha256_by_client[cid] !=
                bank["support_sha256"][cid][jq_lock.choice_by_client[cid]] for cid in clients)):
        raise ValueError("JQ pretraining lock differs from bank supports")
    _validate_view(readout, "readout2048", 2048)
    if set(local64) != set(clients):
        raise ValueError("local64 views must cover the fixed ten participants")
    views = [readout]
    for cid in clients:
        _validate_view(local64[cid], "local64", 64)
        views.append(local64[cid])
    seen: set[tuple[str, int]] = {
        (selection_lock.selection_source_id, row) for row in selection_lock.selection_row_ids}
    for view in views:
        identifiers = {(view.source_id, row) for row in view.row_ids}
        if seen & identifiers:
            raise ValueError("diagnostic data roles overlap")
        seen.update(identifiers)
    _projection_names(parent)
    parent_readout = _loss(loss, parent, readout)
    parent_local = {cid: _loss(loss, parent, local64[cid]) for cid in clients}
    values: dict[str, dict[int, dict[str, Any]]] = {}
    for support in MEASURED_CHOICES:
        values[support] = {}
        for prefix in PREFIXES:
            if support == "finite_bank_search" and prefix != selection_lock.prefix:
                continue
            choices = (search_choices if support == "finite_bank_search" else
                       dict(jq_lock.choice_by_client) if support == "JQ_xi1" else
                       {cid: support for cid in clients})
            children = _children(bank, clients, choices, prefix)
            local_benefit = {}
            weighted_benefit = {}
            for cid, child in zip(clients, children):
                local_benefit[cid] = parent_local[cid] - _loss(loss, child, local64[cid])
                weighted_children = [child if other == cid else parent for other in clients]
                weighted = _aggregate(parent, weighted_children, aggregate)
                _verify_single_weight(parent, child, weighted)
                weighted_benefit[cid] = parent_readout - _loss(loss, weighted, readout)
            aggregated = _aggregate(parent, children, aggregate)
            if support in bank["aggregate_states"] and not _same_state(
                    aggregated, bank["aggregate_states"][support][prefix]):
                raise ValueError("real aggregate differs from stored bank aggregate")
            cancellation = cancellation_and_linearity(parent, children, aggregated)
            head_only = {key: aggregated[key] if key.startswith("classifier.") else parent[key]
                         for key in parent}
            lora_only = {key: parent[key] if key.startswith("classifier.") else aggregated[key]
                         for key in parent}
            aggregate_benefit = parent_readout - _loss(loss, aggregated, readout)
            values[support][prefix] = {
                "local64_benefit_by_client": local_benefit,
                "weighted_readout2048_benefit_by_client": weighted_benefit,
                "aggregate_readout2048_benefit": aggregate_benefit,
                "nonadditivity": aggregate_benefit - sum(weighted_benefit.values()),
                "head_only_readout2048_benefit": parent_readout - _loss(loss, head_only, readout),
                "lora_only_readout2048_benefit": parent_readout - _loss(loss, lora_only, readout),
                **cancellation}
    return {"schema": "pcla-v7-cohort-diagnostic-readout-1", "object_id": bank["object_id"],
            "anchor_sha256": bank["anchor_sha256"], "selection_lock_sha256": selection_lock.sha256,
            "local64_source_ids": {cid: local64[cid].source_id for cid in clients},
            "readout2048_source_id": readout.source_id,
            "parent_readout2048_loss": parent_readout, "values": values}
