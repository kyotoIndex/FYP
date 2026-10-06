"""Build one fixed-parent cohort's masks before any diagnostic bank training.

The practical local/cyclic/JL/JQ path is delegated to the original v7
coordination code. The public512 comparison uses the pre-result frozen rule
public512_mean_eval_h32_JL_v1: only the shared gradient direction changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from pcla_repair.core import lora_modules, tensor_digest
from pcla_repair.runtime import batches, to_device
from .cohort_runtime import _cyclic_offsets, _factors, _full_support, probe_full_gradients, probe_seed
from .coordination import ClientPayload, client_payload, coordinate_masks, projection, sketch
from .diagnostic_bank import JQLock, make_jq_lock, support_digest
from .runtime import attach_slices, load_state, release_slices, state
from .selectors import module_order, topk_mask


@dataclass(frozen=True)
class PublicSupportDecision:
    """The named public-linear rule and its exact masks."""
    rule_id: str
    public_gradient_sha256: str
    masks: Mapping[int, Mapping[str, Tensor]]


PUBLIC_RULE_ID = "public512_mean_eval_h32_JL_v1"


def public512_mean_eval_gradient(model: nn.Module, dataset: Any, factor: str,
                                 *, batch_size: int = 16) -> tuple[dict[str, Tensor], int]:
    """Mean active-factor gradient over all public512 rows in source order.

    This is the 32-batch extension of the real two-batch eval probe: identical
    compact full support, loss.backward, CPU float64 accumulation and factor-B
    transpose. All batches have 16 rows, so their arithmetic mean is the
    512-example mean. No dropout or optimizer update occurs.
    """
    if (factor not in {"a", "b"} or batch_size != 16 or len(dataset) != 512 or
            getattr(dataset, "role", None) != "public_training" or
            getattr(dataset, "actor", None) != "algorithm"):
        raise ValueError("public gradient requires the authorized 512 rows and batch16")
    previous_mode = model.training
    attach_slices(model, _full_support(model, factor), train_head=False)
    device = next(model.parameters()).device
    model.eval()
    gradients = {name: torch.zeros_like(module.lora_a if factor == "a" else module.lora_b,
                                       dtype=torch.float64, device="cpu")
                 for name, module in lora_modules(model).items()}
    rows = batches_seen = 0
    try:
        for batch in batches(dataset, 16, seed=0, shuffle=False):
            model.zero_grad(set_to_none=True)
            output = model(**to_device(batch, device))
            if not torch.isfinite(output.loss):
                raise FloatingPointError("nonfinite public eval probe")
            if len(batch["labels"]) != 16:
                raise ValueError("public gradient batch differs from frozen batch16")
            output.loss.backward()
            rows += 16
            batches_seen += 1
            for name, module in lora_modules(model).items():
                active = getattr(module, "active_" + factor)
                if active.grad is None:
                    raise RuntimeError(f"missing public full gradient: {name}")
                value = active.grad.detach().to(device="cpu", dtype=torch.float64)
                gradients[name] += value if factor == "a" else value.T
    finally:
        model.zero_grad(set_to_none=True)
        release_slices(model)
        model.train(previous_mode)
    if rows != 512 or batches_seen != 32:
        raise ValueError("public gradient did not consume 32 full source-order batches")
    return {name: gradient / batches_seen for name, gradient in gradients.items()}, rows


def public512_mean_eval_h32_JL_v1(*, public_gradients: Mapping[str, Tensor],
                                   payloads: Sequence[ClientPayload],
                                   factors: Mapping[str, tuple[Tensor, Tensor]],
                                   factor: str, budgets: Mapping[int, int],
                                   weights: Mapping[int, float], seed: int,
                                   round_id: int) -> PublicSupportDecision:
    """The JL score with Cbar replaced only by projected public mean C_public."""
    if (factor not in {"a", "b"} or set(public_gradients) != set(factors) or
            len(payloads) != 10 or len({payload.client_id for payload in payloads}) != 10 or
            set(budgets) != set(weights) or set(budgets) != {p.client_id for p in payloads} or
            any(weight != .1 for weight in weights.values())):
        raise ValueError("public linear rule needs the fixed ten-client JL inputs")
    public_sketches = {}
    for name, (b, a) in factors.items():
        gradient = public_gradients[name]
        target = a if factor == "a" else b
        if gradient.shape != target.shape or not torch.isfinite(gradient).all():
            raise ValueError(f"invalid public full gradient: {name}")
        width = gradient.shape[1 if factor == "a" else 0]
        public_sketches[name] = sketch(gradient,
            projection(seed, round_id, name, factor, width, h=32), factor).double()
    masks = {}
    for payload in payloads:
        cid = payload.client_id
        if set(payload.update_sketches) != set(public_sketches):
            raise ValueError("client update sketches differ from public projections")
        scores = {name: -float(weights[cid]) *
                  (public_sketches[name] * payload.update_sketches[name].double()).sum(0)
                  for name in module_order(public_sketches)}
        masks[cid] = topk_mask(scores, budgets[cid])
    return PublicSupportDecision(PUBLIC_RULE_ID, tensor_digest(public_gradients), masks)


def _gradient_map(value: Any, factors: Mapping[str, tuple[Tensor, Tensor]],
                  factor: str, expected_rows: int, label: str) -> dict[str, Tensor]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{label} callback must return (full_gradients, row_count)")
    gradients, count = value
    if count != expected_rows or not isinstance(gradients, Mapping) or set(gradients) != set(factors):
        raise ValueError(f"{label} must use all {expected_rows} rows and all parent projections")
    result = {}
    for name, (b, a) in factors.items():
        source = gradients[name]
        expected = a if factor == "a" else b
        if (not isinstance(source, Tensor) or source.shape != expected.shape or
                not torch.isfinite(source).all()):
            raise ValueError(f"{label} gradient shape/finite check failed: {name}")
        result[name] = source.detach().cpu().double().clone()
    return result


def _active_support(mask: Mapping[str, Tensor], factor: str,
                    ranks: Mapping[str, int], budget: int) -> dict[str, dict[str, Tensor]]:
    if set(mask) != set(ranks):
        raise ValueError("candidate mask projections differ from parent")
    selected = 0
    support = {}
    for name in module_order(mask):
        chosen = mask[name]
        if (not isinstance(chosen, Tensor) or chosen.dtype != torch.bool or
                chosen.ndim != 1 or len(chosen) != ranks[name]):
            raise ValueError(f"invalid boolean support: {name}")
        chosen = chosen.detach().cpu().clone()
        selected += int(chosen.sum())
        support[name] = {"a": chosen if factor == "a" else torch.zeros_like(chosen),
                         "b": chosen if factor == "b" else torch.zeros_like(chosen)}
    if selected != budget:
        raise ValueError(f"candidate uses {selected} positions, expected {budget}")
    return support


def _payload_sha(payloads: Sequence[ClientPayload]) -> str:
    tensors = {}
    for item in payloads:
        for label, values in (("gradient", item.gradient_sketches),
                              ("update", item.update_sketches),
                              ("local_score", item.local_scores)):
            for name, value in values.items():
                tensors[f"{item.client_id}/{label}/{name}"] = value
    return tensor_digest(tensors)


def _save_rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _restore_rng(saved) -> None:
    random.setstate(saved[0])
    np.random.set_state(saved[1])
    torch.set_rng_state(saved[2])
    if saved[3] is not None:
        torch.cuda.set_rng_state_all(saved[3])


def construct_diagnostic_supports(*, model: nn.Module,
                                  parent_state: Mapping[str, Tensor],
                                  object_id: str, anchor_identity: str,
                                  anchor_sha256: str, seed: int, round_id: int,
                                  factor: str, participants: Sequence[int],
                                  clients: Mapping[int, Any],
                                  public: Any, visits: Mapping[int, int],
                                  capacities: Mapping[int, int], locked_at_utc: str,
                                  public_gradient: Callable[..., tuple[Mapping[str, Tensor], int]] | None = public512_mean_eval_gradient,
                                  public_linear_rule: Callable[..., PublicSupportDecision] | None = public512_mean_eval_h32_JL_v1,
                                  client_gradient: Callable[..., tuple[Mapping[str, Tensor], int]] = probe_full_gradients,
                                  learning_rate: float = .001) -> dict[str, Any]:
    """Produce bank-ready supports and a JQ xi=1 lock from one h32 stream.

    The full client_train datasets contain 256 rows. The original keyed v7
    probe picks exactly two 16-row batches from each, preserving both row
    identity and batch order. The client callback is invoked once/client and
    must return those 32 rows' full active-factor gradients. The public
    callback must similarly use all public512 rows in source order. The named
    default linear rule changes only JL's shared direction source to public512.
    Neither callback receives bank outcomes, selection1024 or readout2048.
    """
    client_datasets = clients
    clients = tuple(participants)
    if (not object_id or not anchor_identity or len(anchor_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in anchor_sha256) or
            factor not in {"a", "b"} or type(seed) is not int or seed < 0 or
            type(round_id) is not int or round_id < 2 or
            len(clients) != 10 or len(set(clients)) != 10 or
            any(type(cid) is not int or cid < 0 for cid in clients) or
            not set(clients) <= set(client_datasets) or set(visits) != set(clients) or
            not set(clients) <= set(capacities) or
            any(type(visits[cid]) is not int or visits[cid] < 0 for cid in clients) or
            any(type(capacities[cid]) is not int or capacities[cid] <= 0 for cid in clients)):
        raise ValueError("fixed anchor, ten clients, visits and positive budgets are required")
    if (public_gradient is None or public_linear_rule is None or
            not callable(public_gradient) or not callable(public_linear_rule)):
        raise ValueError("public512 gradient estimator and linear score rule must be frozen")
    if any(len(client_datasets[cid]) != 256 or getattr(client_datasets[cid], "role", None) != "client_train"
           or getattr(client_datasets[cid], "actor", None) != "algorithm" for cid in clients):
        raise ValueError("each client must provide its authorized 256 train rows")
    if (len(public) != 512 or getattr(public, "role", None) != "public_training" or
            getattr(public, "actor", None) != "algorithm"):
        raise ValueError("public direction may read only algorithm-authorized public512")
    if not (0 < learning_rate < float("inf")):
        raise ValueError("invalid common cohort learning rate")
    modules = lora_modules(model)
    if len(modules) != 24:
        raise ValueError("cohort support construction requires all 24 Q/V modules")
    if set(parent_state) != set(state(model)) or any(
            not torch.equal(parent_state[name].cpu(), current) for name, current in state(model).items()):
        raise ValueError("model differs from the frozen parent adapter/head state")
    ranks = {name: module.rank for name, module in modules.items()}
    budgets = {cid: capacities[cid] for cid in clients}
    if any(budgets[cid] > sum(ranks.values()) for cid in clients):
        raise ValueError("client budget exceeds active-factor positions")
    saved_rng = _save_rng()
    was_training = model.training
    requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    initial_sha = tensor_digest(state(model))
    try:
        factors = _factors(model)
        gradients = {}
        payloads = []
        for cid in clients:
            load_state(model, parent_state)
            value = client_gradient(model, client_datasets[cid], factor,
                                    seed=probe_seed(seed, round_id, cid, factor),
                                    batch_size=16, probe_batches=2)
            gradients[cid] = _gradient_map(value, factors, factor, 32, "client32")
            payloads.append(client_payload(cid, round_id - 1, factors,
                            gradients[cid], factor, learning_rate, seed, round_id, h=32))
        weights = {cid: .1 for cid in clients}
        names = module_order(factors)
        if len({rank for rank in ranks.values()}) != 1:
            raise ValueError("v7 cyclic/coordination requires a common rank")
        rank = next(iter(ranks.values()))
        offsets = {cid: _cyclic_offsets(seed, cid, factor, names, rank) for cid in clients}
        practical = coordinate_masks(payloads, weights, budgets, factors, factor,
                    visits, "JQ", xi=1., offsets_by_client=offsets,
                    expected_parent_version=round_id - 1)
        if practical.choices is None:
            raise RuntimeError("JQ did not produce a pretraining choice")
        supports = {}
        names_by_choice = {"local": practical.local_masks,
                           "cyclic": practical.cyclic_masks,
                           "JL_h32": practical.jl_masks}
        for cid in clients:
            supports[cid] = {name: _active_support(mask[cid], factor, ranks, budgets[cid])
                             for name, mask in names_by_choice.items()}
        jq_choices = {cid: "JL_h32" if practical.choices[cid] == "JL"
                      else practical.choices[cid] for cid in clients}
        for cid in clients:
            actual = _active_support(practical.masks[cid], factor, ranks, budgets[cid])
            if support_digest(actual, factor=factor) != support_digest(
                    supports[cid][jq_choices[cid]], factor=factor):
                raise ValueError("JQ practical mask is not the bank candidate mask")
        lock = make_jq_lock(object_id=object_id, participants=clients,
                 choice_by_client=jq_choices,
                 support_sha256_by_client={cid: support_digest(
                     supports[cid][jq_choices[cid]], factor=factor) for cid in clients},
                 h32_sha256=_payload_sha(payloads), locked_at_utc=locked_at_utc,
                 information_access=("parent_state", "client_train_probe32", "local_h32",
                                     "JQ_xi1_coordination"))
        load_state(model, parent_state)
        public_gradients = _gradient_map(public_gradient(model, public, factor, batch_size=16),
                                         factors, factor, 512, "public512")
        public_sha = tensor_digest(public_gradients)
        decision = public_linear_rule(public_gradients=public_gradients, payloads=tuple(payloads),
                        factors=factors, factor=factor, budgets=dict(budgets),
                        weights=weights, seed=seed, round_id=round_id)
        if (not isinstance(decision, PublicSupportDecision) or not decision.rule_id or
                decision.public_gradient_sha256 != public_sha or
                set(decision.masks) != set(clients)):
            raise ValueError("public linear rule lacks frozen identity or valid public gradient binding")
        for cid in clients:
            supports[cid]["public512_linear"] = _active_support(
                decision.masks[cid], factor, ranks, budgets[cid])
        if tensor_digest(state(model)) != initial_sha:
            raise ValueError("support construction mutated frozen parent")
        return {"supports": supports, "jq_lock": lock,
                "full_probe_gradients": gradients, "h32_payloads": tuple(payloads),
                "h32_sha256": lock.h32_sha256,
                "public_full_gradient": public_gradients,
                "public_gradient_sha256": public_sha,
                "public_linear_rule_id": decision.rule_id,
                "public_rows": 512, "client_probe_rows": 32,
                "object_id": object_id, "anchor_identity": anchor_identity,
                "anchor_sha256": anchor_sha256, "factor": factor,
                "participants": list(clients), "visits": dict(visits),
                "budgets": dict(budgets),
                "jq_objective_initial": practical.objective_initial,
                "jq_objective_final": practical.objective_final,
                "jq_lambda": practical.lambda_value}
    finally:
        if any(getattr(module, "_v7_active", False) for module in modules.values()):
            release_slices(model)
        load_state(model, parent_state)
        model.train(was_training)
        for name, parameter in model.named_parameters():
            if name in requires_grad:
                parameter.requires_grad_(requires_grad[name])
        _restore_rng(saved_rng)
