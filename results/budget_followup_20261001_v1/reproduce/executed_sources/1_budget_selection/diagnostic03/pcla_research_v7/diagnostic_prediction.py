"""Compare h32 and exact same-probe directions on four fixed bank masks.

Prediction consumes no trained update, selection1024 or readout2048. The
returned full-direction lock can later select *existing* bank trajectories for
an independent readout; it never creates a fifth support or trains a model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from pcla_repair.core import tensor_digest
from .coordination import ClientPayload, projection, sketch, virtual_adamw_step
from .diagnostic_bank import SUPPORTS, support_digest
from .diagnostic_readout import DiagnosticView, _aggregate, _validate_view
from .selectors import module_order


FULL_CHOICE_SCHEMA = "pcla-v7-full-gradient-four-bank-choice-1"
INITIAL_CHOICE = "JL_h32"


@dataclass(frozen=True)
class FullChoiceLock:
    schema: str
    object_id: str
    anchor_sha256: str
    factor: str
    prefix: int
    h32_sha256: str
    full_gradient_sha256: str
    choices: tuple[tuple[int, str], ...]
    support_sha256_by_client: tuple[tuple[int, str], ...]
    evaluations: int
    lambda_value: float
    objective_initial: float
    objective_final: float
    sha256: str


def _json_sha(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _gradient_sha(gradients: Mapping[int, Mapping[str, Tensor]]) -> str:
    return tensor_digest({f"{cid}/{name}": value for cid, modules in gradients.items()
                          for name, value in modules.items()})


def _h32_sha(payloads: Sequence[ClientPayload]) -> str:
    tensors = {}
    for item in payloads:
        for field, values in (("gradient", item.gradient_sketches),
                              ("update", item.update_sketches),
                              ("local_score", item.local_scores)):
            tensors.update({f"{item.client_id}/{field}/{name}": value
                            for name, value in values.items()})
    return tensor_digest(tensors)


def _average_rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average = (index + end - 1) / 2 + 1
        for position in order[index:end]:
            ranks[position] = average
        index = end
    return ranks


def spearman_four(a: Sequence[float], b: Sequence[float]) -> tuple[float | None, str]:
    """Average ties; a constant predictor has undefined rank correlation."""
    if len(a) != 4 or len(b) != 4 or any(not math.isfinite(float(x)) for x in (*a, *b)):
        raise ValueError("Spearman requires four finite fixed-candidate scores")
    ra, rb = _average_rank(a), _average_rank(b)
    ma, mb = sum(ra) / 4, sum(rb) / 4
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((x - mb) ** 2 for x in rb)
    if va == 0 or vb == 0:
        return None, "NA_constant"
    value = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(va * vb)
    return value, "measured"


def _active_mask(support: Mapping[str, Mapping[str, Tensor]], factor: str,
                 names: Sequence[str], rank: int) -> dict[str, Tensor]:
    if set(support) != set(names):
        raise ValueError("fixed candidate omits a projection")
    masks = {}
    for name in names:
        pair = support[name]
        if set(pair) != {"a", "b"}:
            raise ValueError("fixed candidate needs both factor masks")
        active, inactive = pair[factor], pair["b" if factor == "a" else "a"]
        if (not isinstance(active, Tensor) or not isinstance(inactive, Tensor) or
                active.dtype != torch.bool or inactive.dtype != torch.bool or
                active.shape != (rank,) or inactive.shape != (rank,) or
                bool(inactive.any())):
            raise ValueError("fixed candidate factor/budget shape differs")
        masks[name] = active.detach().cpu()
    return masks


def _directions(*, clients: tuple[int, ...], names: list[str], factor: str,
                parent: Mapping[str, Tensor],
                full_gradients: Mapping[int, Mapping[str, Tensor]],
                payloads: Sequence[ClientPayload], seed: int, round_id: int,
                learning_rate: float) -> tuple[dict[str, dict[int, dict[str, Tensor]]],
                                               dict[str, dict[str, Tensor]],
                                               dict[str, Tensor]]:
    by_id = {payload.client_id: payload for payload in payloads}
    full_c, full_t, h32_c, h32_t = {}, {}, {}, {}
    grams = {}
    for name in names:
        b, a = parent[name + ".lora_b"], parent[name + ".lora_a"]
        partner = (a if factor == "b" else b).detach().cpu().double()
        grams[name] = partner @ partner.T if factor == "b" else partner.T @ partner
    for cid in clients:
        full_c[cid], full_t[cid], h32_c[cid], h32_t[cid] = {}, {}, {}, {}
        for name in names:
            b, a = parent[name + ".lora_b"], parent[name + ".lora_a"]
            parameter = a if factor == "a" else b
            gradient = full_gradients[cid][name]
            if gradient.shape != parameter.shape or not torch.isfinite(gradient).all():
                raise ValueError(f"full probe gradient differs from parent: {cid}/{name}")
            fresh = virtual_adamw_step(parameter, gradient, learning_rate)
            width = gradient.shape[1 if factor == "a" else 0]
            shared_projection = projection(seed, round_id, name, factor, width, h=32)
            expected_c = sketch(gradient, shared_projection, factor)
            expected_t = sketch(fresh, shared_projection, factor)
            payload = by_id[cid]
            if (not torch.equal(expected_c, payload.gradient_sketches[name]) or
                    not torch.equal(expected_t, payload.update_sketches[name])):
                raise ValueError(f"h32 payload is not the same full-probe gradient/update: {cid}/{name}")
            full_c[cid][name] = (gradient.double().T if factor == "a" else gradient.double())
            full_t[cid][name] = (fresh.double().T if factor == "a" else fresh.double())
            h32_c[cid][name] = payload.gradient_sketches[name].double()
            h32_t[cid][name] = payload.update_sketches[name].double()
    c = {"h32": h32_c, "full": full_c}
    t = {"h32": h32_t, "full": full_t}
    return ({direction: {cid: t[direction][cid] for cid in clients}
             for direction in ("h32", "full")},
            {direction: {name: sum((.1 * c[direction][cid][name] for cid in clients),
                                      torch.zeros_like(c[direction][clients[0]][name]))
                         for name in names} for direction in ("h32", "full")}, grams)


def _terms(choices: Mapping[int, str], *, clients: tuple[int, ...],
           names: list[str], menu: Mapping[int, Mapping[str, Mapping[str, Tensor]]],
           updates: Mapping[int, Mapping[str, Tensor]], cbar: Mapping[str, Tensor],
           grams: Mapping[str, Tensor]) -> tuple[float, float]:
    linear = quadratic = 0.
    for name in names:
        total = sum((.1 * updates[cid][name] * menu[cid][choices[cid]][name].double()[None, :]
                     for cid in clients), torch.zeros_like(cbar[name]))
        linear -= float((cbar[name] * total).sum())
        quadratic += float(torch.einsum("hr,rs,hs->", total, grams[name], total))
    return linear, max(0., quadratic)


def _predict_direction(*, clients: tuple[int, ...], names: list[str],
                       menu: Mapping[int, Mapping[str, Mapping[str, Tensor]]],
                       updates: Mapping[int, Mapping[str, Tensor]],
                       cbar: Mapping[str, Tensor], grams: Mapping[str, Tensor],
                       jq_choices: Mapping[int, str]) -> dict[str, Any]:
    reference = {cid: "cyclic" for cid in clients}
    reference_linear, reference_quadratic = _terms(reference, clients=clients, names=names,
        menu=menu, updates=updates, cbar=cbar, grams=grams)
    numerator = max(0., reference_linear)  # xi=1, identical cyclic reference.
    curvature = (numerator / max(reference_quadratic, np.finfo(np.float64).tiny)
                 if reference_quadratic else 0.)
    if not math.isfinite(curvature):
        raise FloatingPointError("nonfinite fixed-candidate curvature")
    cached: dict[tuple[str, ...], dict[str, float]] = {}
    def objective(choices: Mapping[int, str]) -> dict[str, float]:
        key = tuple(choices[cid] for cid in clients)
        if key not in cached:
            if len(cached) >= 80:
                raise RuntimeError("four-bank coordinate search exceeded 80 evaluations")
            linear, quad = _terms(choices, clients=clients, names=names, menu=menu,
                                  updates=updates, cbar=cbar, grams=grams)
            cached[key] = {"linear": linear, "quadratic": quad,
                           "phi": linear - .5 * curvature * quad}
        return cached[key]
    choices = {cid: INITIAL_CHOICE for cid in clients}
    initial = objective(choices)["phi"]
    current = initial
    for _pass in range(2):
        for cid in clients:
            original, winner, best = choices[cid], choices[cid], current
            for candidate in SUPPORTS:
                if candidate == original:
                    continue
                choices[cid] = candidate
                value = objective(choices)["phi"]
                if value > best + 1e-12:
                    winner, best = candidate, value
            choices[cid] = winner
            current = best
    fixed = {"all_JL_h32": {cid: "JL_h32" for cid in clients},
             "all_cyclic": reference,
             "all_local": {cid: "local" for cid in clients},
             "all_public512_linear": {cid: "public512_linear" for cid in clients},
             "practical_JQ_xi1": jq_choices,
             "four_bank_selected": choices}
    return {"lambda_xi1": curvature, "reference_cyclic_linear": reference_linear,
            "reference_cyclic_quadratic": reference_quadratic,
            "objective_initial": initial, "objective_final": current,
            "choices": dict(choices), "evaluations": len(cached),
            "fixed_combinations": {label: dict(objective(selected))
                                   for label, selected in fixed.items()}}


def compare_fixed_bank_predictions(*, supports: Mapping[int, Mapping[str, Any]],
                                   jq_lock: Any, full_probe_gradients: Mapping[int, Mapping[str, Tensor]],
                                   h32_payloads: Sequence[ClientPayload],
                                   h32_sha256: str, object_id: str, anchor_sha256: str,
                                   parent: Mapping[str, Tensor], factor: str,
                                   participants: Sequence[int], budgets: Mapping[int, int],
                                   seed: int, round_id: int,
                                   learning_rate: float) -> dict[str, Any]:
    """Predict four already fixed masks from one full32 gradient/payload pair."""
    clients = tuple(sorted(participants))
    if (len(clients) != 10 or len(set(clients)) != 10 or factor not in {"a", "b"} or
            set(supports) != set(clients) or set(full_probe_gradients) != set(clients) or
            set(budgets) != set(clients) or any(type(budgets[cid]) is not int or budgets[cid] <= 0
                                            for cid in clients) or
            not math.isfinite(learning_rate) or learning_rate <= 0 or
            len(h32_payloads) != 10 or {p.client_id for p in h32_payloads} != set(clients) or
            {p.parent_version for p in h32_payloads} != {round_id - 1} or
            h32_sha256 != _h32_sha(h32_payloads) or
            jq_lock.object_id != object_id or jq_lock.h32_sha256 != h32_sha256 or
            set(jq_lock.choice_by_client) != set(clients) or
            not object_id or len(anchor_sha256) != 64):
        raise ValueError("fixed bank prediction identity/probe inputs differ")
    names = module_order(name[:-7] for name in parent if name.endswith(".lora_a"))
    if len(names) != 24 or any(name + ".lora_b" not in parent for name in names):
        raise ValueError("prediction needs all 24 parent Q/V modules")
    ranks = {parent[name + ".lora_a"].shape[0] for name in names}
    if len(ranks) != 1:
        raise ValueError("prediction requires common component rank")
    rank = ranks.pop()
    menu = {}
    for cid in clients:
        if set(supports[cid]) != set(SUPPORTS) or set(full_probe_gradients[cid]) != set(names):
            raise ValueError("four bank candidates/full gradients incomplete")
        menu[cid] = {choice: _active_mask(supports[cid][choice], factor, names, rank)
                     for choice in SUPPORTS}
        if any(sum(int(mask.sum()) for mask in menu[cid][choice].values()) != budgets[cid]
               for choice in SUPPORTS):
            raise ValueError("candidate mask differs from fixed client budget")
        selected = jq_lock.choice_by_client[cid]
        if (selected not in SUPPORTS[:3] or
                jq_lock.support_sha256_by_client[cid] != support_digest(
                    supports[cid][selected], factor=factor)):
            raise ValueError("pretraining JQ lock differs from fixed bank masks")
    payload_by_id = {p.client_id: p for p in h32_payloads}
    if any(set(payload_by_id[cid].gradient_sketches) != set(names) or
           set(payload_by_id[cid].update_sketches) != set(names) for cid in clients):
        raise ValueError("incomplete h32 payload")
    updates, cbars, grams = _directions(clients=clients, names=names, factor=factor,
        parent=parent, full_gradients=full_probe_gradients,
        payloads=h32_payloads, seed=seed, round_id=round_id,
        learning_rate=learning_rate)
    jq_choices = {cid: jq_lock.choice_by_client[cid] for cid in clients}
    predictions = {direction: _predict_direction(clients=clients, names=names,
        menu=menu, updates=updates[direction], cbar=cbars[direction],
        grams=grams, jq_choices=jq_choices) for direction in ("h32", "full")}
    per_client = {}
    for cid in clients:
        scores = {}
        for direction in ("h32", "full"):
            scores[direction] = {}
            for choice in SUPPORTS:
                scores[direction][choice] = -sum(float((cbars[direction][name] *
                    updates[direction][cid][name] * menu[cid][choice][name].double()[None, :]).sum())
                    for name in names) * .1
        correlation, status = spearman_four(
            [scores["h32"][choice] for choice in SUPPORTS],
            [scores["full"][choice] for choice in SUPPORTS])
        per_client[cid] = {"one_step_linear_gain": scores, "spearman": correlation,
                           "spearman_status": status}
    chosen = predictions["full"]["choices"]
    lock_record = {"schema": FULL_CHOICE_SCHEMA, "object_id": object_id,
                   "anchor_sha256": anchor_sha256, "factor": factor, "prefix": 16,
                   "h32_sha256": h32_sha256,
                   "full_gradient_sha256": _gradient_sha(full_probe_gradients),
                   "choices": tuple((cid, chosen[cid]) for cid in clients),
                   "support_sha256_by_client": tuple((cid, support_digest(
                       supports[cid][chosen[cid]], factor=factor)) for cid in clients),
                   "evaluations": predictions["full"]["evaluations"],
                   "lambda_value": predictions["full"]["lambda_xi1"],
                   "objective_initial": predictions["full"]["objective_initial"],
                   "objective_final": predictions["full"]["objective_final"]}
    lock = FullChoiceLock(**lock_record, sha256=_json_sha(lock_record))
    valid = [row["spearman"] for row in per_client.values() if row["spearman"] is not None]
    return {"schema": "pcla-v7-h32-vs-full-bank-prediction-1", "object_id": object_id,
            "anchor_sha256": anchor_sha256, "factor": factor,
            "candidate_order": list(SUPPORTS), "head_prediction": "constant_across_supports_omitted",
            "per_client": per_client, "spearman_mean_defined":
                sum(valid) / len(valid) if valid else None,
            "spearman_defined_clients": len(valid),
            "prediction": predictions, "full_choice_lock": lock}


def readout_locked_full_choice(*, lock: FullChoiceLock, bank: Mapping[str, Any],
                               parent: Mapping[str, Tensor], readout: DiagnosticView,
                               loss: Callable[[Mapping[str, Tensor], Any], float],
                               aggregate: Callable[..., Mapping[str, Tensor]]) -> dict[str, Any]:
    """Only after locking, read independent 2048 on actual trained bank choices."""
    if not isinstance(lock, FullChoiceLock):
        raise ValueError("full-gradient bank choice must be locked before readout")
    _validate_view(readout, "readout2048", 2048)
    record = {field: getattr(lock, field) for field in (
        "schema", "object_id", "anchor_sha256", "factor", "prefix", "h32_sha256",
        "full_gradient_sha256", "choices", "support_sha256_by_client",
        "evaluations", "lambda_value", "objective_initial", "objective_final")}
    if (lock.sha256 != _json_sha(record) or lock.schema != FULL_CHOICE_SCHEMA or
            bank.get("schema") != "pcla-v7-cohort-bank-1" or
            bank.get("object_id") != lock.object_id or
            bank.get("anchor_sha256") != lock.anchor_sha256 or
            bank.get("factor") != lock.factor or lock.prefix != 16 or
            not readout.source_id):
        raise ValueError("bank/readout identity differs from full-gradient lock")
    clients = tuple(cid for cid, _ in lock.choices)
    choices = dict(lock.choices)
    if (len(clients) != 10 or clients != tuple(sorted(clients)) or
            set(bank.get("participants", ())) != set(clients) or
            not 1 <= lock.evaluations <= 80 or
            any(choice not in SUPPORTS for choice in choices.values()) or
            dict(lock.support_sha256_by_client) != {cid: bank["support_sha256"][cid][choices[cid]]
                                                    for cid in clients}):
        raise ValueError("locked combination differs from actual bank masks")
    children = [bank["prefix_states"][choices[cid]][cid][16] for cid in clients]
    selected = _aggregate(parent, children, aggregate)
    before, after = float(loss(parent, readout.dataset)), float(loss(selected, readout.dataset))
    if not math.isfinite(before) or not math.isfinite(after):
        raise ValueError("nonfinite independent bank readout")
    return {"schema": "pcla-v7-full-gradient-bank-readout-1", "object_id": lock.object_id,
            "full_choice_lock_sha256": lock.sha256, "readout_source_id": readout.source_id,
            "readout_rows": 2048, "parent_loss": before, "selected_loss": after,
            "actual_bank_benefit": before - after, "choices": dict(lock.choices)}
