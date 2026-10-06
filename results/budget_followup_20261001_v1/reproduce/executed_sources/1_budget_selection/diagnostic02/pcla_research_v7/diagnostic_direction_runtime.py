"""Real fixed-parent direction diagnostic; the caller owns queue and persistence.

One object trains six coordinate representations under local and cyclic masks
on the same ten clients.  The 1/4/16 states come from each single 16-step
trajectory.  No diagnostic readout selects a mask or a direction.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from pcla_repair.core import keyed_seed, lora_modules, tensor_digest
from .cohort_runtime import probe_full_gradients, probe_seed
from .coordination import (aggregate_gradient_sketches, client_payload,
                           projection, sketch, virtual_adamw_step)
from .coordinates import rotation_from_sketch, rotate_gradient
from .data import PROTOCOL
from .diagnostic_directions import (HEAD_KEYS, NormMatchedReadout, VARIANTS,
                                    build_direction_variants, matched_training_seed,
                                    read_norm_matched)
from .diagnostic_model import ModelLoss
from .diagnostic_readout import DiagnosticView
from .runtime import (aggregate, evaluate, load_state, local_train,
                      select_support, state)


PREFIXES = (1, 4, 16)
METHODS = ("local", "cyclic")


def _rng_state() -> tuple:
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _restore_rng(saved: tuple) -> None:
    random.setstate(saved[0])
    np.random.set_state(saved[1])
    torch.set_rng_state(saved[2])
    if saved[3] is not None:
        torch.cuda.set_rng_state_all(saved[3])


def _clone(values: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in values.items()}


def _same(a: Mapping[str, Tensor], b: Mapping[str, Tensor]) -> bool:
    return set(a) == set(b) and all(torch.equal(a[key].cpu(), b[key].cpu()) for key in a)


def _names(values: Mapping[str, Tensor]) -> tuple[str, ...]:
    names = tuple(sorted(key[:-7] for key in values if key.endswith(".lora_a")))
    if (len(names) != 24 or
            set(values) != {key for name in names for key in (name + ".lora_a", name + ".lora_b")} | set(HEAD_KEYS)):
        raise ValueError("direction object needs exactly 24 LoRA pairs and head")
    return names


def _factors(values: Mapping[str, Tensor]) -> dict[str, tuple[Tensor, Tensor]]:
    return {name: (values[name + ".lora_b"], values[name + ".lora_a"])
            for name in _names(values)}


def _validate(config: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[tuple[int, ...], int]:
    training = config.get("training", {})
    clients = tuple(config.get("participants", ()))
    if (config.get("schema") != "pcla-v7-diagnostic-config-1" or
            config.get("protocol") != PROTOCOL or config.get("line") != "direction" or
            config.get("phase") != "diagnostic" or not isinstance(config.get("job_id"), str) or
            type(config.get("seed")) is not int or config["seed"] < 0 or
            len(clients) != 10 or len(set(clients)) != 10 or
            any(type(cid) is not int or cid < 0 for cid in clients) or
            config.get("factor") not in {"a", "b"} or
            config.get("anchor_round") not in {10, 60} or
            config.get("participant_schedule_round") != config["anchor_round"] + 1 or
            training.get("paired_rng_key_scope") != ["diagnostic_object_id", "client_id"] or
            training.get("prefix_steps") != list(PREFIXES) or
            training.get("local_optimizer_steps") != 16 or
            training.get("local_examples") != 256 or
            training.get("batch_size") != 16 or
            training.get("optimizer") != "adamw" or
            float(training.get("learning_rate", -1)) != .0003 or
            float(training.get("head_learning_rate", -1)) != .0003 or
            tuple(training.get("betas", ())) != (.9, .999) or
            float(training.get("eps", -1)) != 1e-8 or
            float(training.get("weight_decay", -1)) != .01 or
            config.get("direction", {}).get("representations") != [
                "N", "D", "S", "T", "S_nonzero_nullspace_rotation", "S_zero_direction"] or
            config["direction"].get("support_policies") != list(METHODS) or
            config["direction"].get("forced_coordinate_intervention") is not True or
            config["direction"].get("norm_matched_readout") is not True or
            config["direction"].get("selection_evaluations") != 0 or
            set(context.get("clients", {})) != set(clients) or
            set(context.get("capacities", {})) != set(clients) or
            set(context.get("visits", {})) != set(clients) or
            any(len(context["clients"][cid]) != 256 for cid in clients) or
            context.get("anchor", {}).get("state") != "anchor_verified" or
            context["anchor"].get("participants") != list(clients)):
        raise ValueError("direction config/context differs from frozen 48-object contract")
    if config.get("visits") is not None:
        planned = {int(key): int(value) for key, value in config["visits"].items()}
        if planned != dict(context["visits"]):
            raise ValueError("actual visits differ from frozen anchor")
    view = context.get("readout")
    if (not isinstance(view, DiagnosticView) or view.role != "readout2048" or
            len(view.dataset) != 2048 or len(view.row_ids) != 2048 or not view.source_id):
        raise ValueError("direction loss must use the registered independent readout2048")
    if (config.get("calls", {}).get("clients") != 10 or
            config["calls"].get("local_calls") != 120 or
            config["calls"].get("optimizer_steps") != 1920):
        raise ValueError("direction call budget differs")
    _names(context["parent_state"])
    return clients, config["participant_schedule_round"]


def _check_support(support: Mapping[str, Mapping[str, Tensor]],
                   factors: Mapping[str, tuple[Tensor, Tensor]],
                   factor: str, budget: int) -> None:
    if set(support) != set(factors):
        raise ValueError("support omitted a projection")
    selected = 0
    for name, (_, a) in factors.items():
        masks = support[name]
        if set(masks) != {"a", "b"}:
            raise ValueError("support omitted a factor")
        for f in ("a", "b"):
            mask = masks[f]
            if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.shape != (a.shape[0],):
                raise ValueError("support has invalid mask")
            count = int(mask.sum())
            if f == factor:
                selected += count
            elif count:
                raise ValueError("direction trial updated the inactive factor")
    if selected != budget:
        raise ValueError("direction support did not spend actual capacity")


def _check_aggregate(parent: Mapping[str, Tensor], children: list[Mapping[str, Tensor]],
                     result: Mapping[str, Tensor]) -> None:
    if len(children) != 10 or set(result) != set(parent):
        raise ValueError("aggregate needs ten complete states")
    for key, base in parent.items():
        expected = base.double() + sum(
            ((child[key].double() - base.double()) * .1 for child in children),
            torch.zeros_like(base, dtype=torch.float64))
        if not torch.allclose(result[key].double(), expected, atol=1e-7, rtol=1e-6):
            raise ValueError(f"aggregate differs from fixed ten-client denominator: {key}")


def norm_match_distinct_parents(n_parent: Mapping[str, Tensor],
                                n_aggregated: Mapping[str, Tensor],
                                candidate_parent: Mapping[str, Tensor],
                                candidate_aggregated: Mapping[str, Tensor], *,
                                scaling_by_module: Mapping[str, float]) -> NormMatchedReadout:
    """Use N's own parent and the variant's own parent for their effective deltas."""
    names = _names(n_parent)
    if any(_names(values) != names for values in (n_aggregated, candidate_parent,
                                                  candidate_aggregated)) or set(scaling_by_module) != set(names):
        raise ValueError("norm-match states/scales differ")
    reference_sq = candidate_sq = 0.
    candidate_base, candidate_delta = {}, {}
    for name in names:
        scale = float(scaling_by_module[name])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid LoRA scale")
        def effective(values: Mapping[str, Tensor]) -> Tensor:
            return (values[name + ".lora_b"].double() @ values[name + ".lora_a"].double()) * scale
        nd = effective(n_aggregated) - effective(n_parent)
        base = effective(candidate_parent)
        cd = effective(candidate_aggregated) - base
        reference_sq += float(nd.square().sum())
        candidate_sq += float(cd.square().sum())
        candidate_base[name], candidate_delta[name] = base, cd
    head = {key: candidate_aggregated[key].detach().clone() for key in HEAD_KEYS}
    reference_norm, candidate_norm = math.sqrt(reference_sq), math.sqrt(candidate_sq)
    if reference_sq == 0 or candidate_sq == 0:
        return NormMatchedReadout("NA_zero_reference" if reference_sq == 0 else
                                  "NA_zero_candidate", None, reference_norm,
                                  candidate_norm, None, head)
    multiplier = math.sqrt(reference_sq / candidate_sq)
    matched = {name: candidate_base[name] + multiplier * candidate_delta[name]
               for name in names}
    return NormMatchedReadout("matched", multiplier, reference_norm,
                              candidate_norm, matched, head)


def compare_t_full_h32_directions(*, full_gradients: Mapping[int, Mapping[str, Tensor]],
                                   h32_cbar: Mapping[str, Tensor],
                                   d_parent: Mapping[str, Tensor],
                                   t_parent: Mapping[str, Tensor],
                                   t_supports: Mapping[str, Mapping[int, Any]],
                                   clients: tuple[int, ...], factor: str,
                                   seed: int, round_id: int,
                                   lr: float = .0003) -> dict[str, Any]:
    """Read-only exact-gradient T comparator on the six-representation bank.

    For each module, full Cbar is the ten-client average of the *same* D-parent
    probe gradients (G_B or G_A.T). O_full and deployed O_h32 both use the v7
    eigenvector/sign rule on their respective Cbar.T @ Cbar. The scalar
    predictions use the fixed, already trained T local/cyclic masks. Each
    client's full gradient is chain-rule rotated by O_h32; a fresh one-step
    AdamW delta is then computed at the actual T factor. The exact linear
    score is -sum_m,k .1 <Gbar_full_rot, U_k_rot masked>. Its h32 counterpart
    replaces both matrices with the fixed Rademacher projections. These are
    one-step, LoRA-only predictions, never 16-step outcomes or a seventh T run.
    """
    names = _names(d_parent)
    if (factor not in {"a", "b"} or len(clients) != 10 or len(set(clients)) != 10 or
            set(full_gradients) != set(clients) or set(h32_cbar) != set(names) or
            set(t_supports) != set(METHODS) or _names(t_parent) != names):
        raise ValueError("T full/h32 comparison requires the same ten-client D parent")
    full_bar: dict[str, Tensor] = {}
    rotations: dict[str, Any] = {}
    rotated_full_bar: dict[str, Tensor] = {}
    rotated_h32_bar: dict[str, Tensor] = {}
    projections: dict[str, Tensor] = {}
    for name in names:
        b, a = d_parent[name + ".lora_b"], d_parent[name + ".lora_a"]
        target = a if factor == "a" else b
        gradients = []
        for cid in clients:
            if set(full_gradients[cid]) != set(names):
                raise ValueError("T full probe omitted a D-parent module")
            value = full_gradients[cid][name]
            if (not isinstance(value, Tensor) or value.shape != target.shape or
                    not bool(torch.isfinite(value).all())):
                raise ValueError(f"invalid T full probe gradient: {cid}/{name}")
            gradients.append(value.detach().cpu().double())
        full = sum((.1 * (value.T if factor == "a" else value) for value in gradients),
                   torch.zeros((target.shape[1 if factor == "a" else 0], a.shape[0]),
                               dtype=torch.float64))
        hbar = h32_cbar[name].detach().cpu().double()
        if hbar.shape != (32, a.shape[0]) or not bool(torch.isfinite(hbar).all()):
            raise ValueError(f"T h32 aggregate shape/finite mismatch: {name}")
        oh = rotation_from_sketch(hbar)
        of = rotation_from_sketch(full)
        gram_full = full.T @ full
        gram_h32 = hbar.T @ hbar
        difference = torch.linalg.vector_norm(gram_h32 - gram_full).item()
        reference = torch.linalg.vector_norm(gram_full).item()
        full_bar[name] = full
        rotated_full_bar[name] = full @ oh
        rotated_h32_bar[name] = hbar @ oh
        projections[name] = projection(seed, round_id, name, factor, full.shape[0], h=32)
        rotations[name] = {
            "o_full_untrained": of.clone(), "o_h32_deployed": oh.clone(),
            "full_gram": gram_full.clone(), "h32_gram": gram_h32.clone(),
            "gram_frobenius_relative_error": None if reference == 0 else difference / reference,
            "gram_frobenius_absolute_error": difference,
            "column_absolute_cosine_descriptive_only": torch.diag((of.T @ oh).abs()).clone(),
        }
    predictions = {}
    for method in METHODS:
        if set(t_supports[method]) != set(clients):
            raise ValueError("T prediction lacks a fixed bank support")
        per_client = {}
        for cid in clients:
            _check_support(t_supports[method][cid], _factors(t_parent), factor,
                           sum(int(pair[factor].sum()) for pair in t_supports[method][cid].values()))
            exact = projected = 0.
            for name in names:
                rotation = rotations[name]["o_h32_deployed"]
                rotated = rotate_gradient(full_gradients[cid][name], rotation, factor)
                actual_factor = t_parent[name + ".lora_" + factor]
                update = virtual_adamw_step(actual_factor, rotated, lr,
                                             eps=1e-8, weight_decay=.01)
                mask = t_supports[method][cid][name][factor].detach().cpu().bool()
                if factor == "b":
                    masked = update.double() * mask.double()[None, :]
                    full_update = masked
                else:
                    masked = update.double() * mask.double()[:, None]
                    full_update = masked.T
                projected_update = sketch(masked, projections[name], factor).double()
                exact -= .1 * float((rotated_full_bar[name] * full_update).sum())
                projected -= .1 * float((rotated_h32_bar[name] * projected_update).sum())
            per_client[cid] = {"full": exact, "h32": projected}
        predictions[method] = {"by_client": per_client,
                               "full": sum(row["full"] for row in per_client.values()),
                               "h32": sum(row["h32"] for row in per_client.values())}
    return {"schema": "pcla-v7-direction-T-full-h32-comparator-1",
            "status": "diagnostic_only_no_extra_training",
            "full_cbar_by_module": full_bar,
            "full_cbar_sha256": tensor_digest(full_bar),
            "h32_cbar_sha256": tensor_digest(h32_cbar),
            "rotations_and_grams": rotations,
            "prediction_kind": "same_fixed_T_masks_fresh_adamw_one_step_linear_LoRA_only",
            "one_step_linear_gain": predictions}


class TemporaryEffectiveReadout:
    """Evaluate W0+matched effective matrices and the actual head, then restore."""

    def __init__(self, model: nn.Module, view: DiagnosticView, *,
                 evaluate_fn: Callable[..., Mapping[str, float]] = evaluate):
        if view.role != "readout2048" or len(view.dataset) != 2048:
            raise ValueError("only registered readout2048 is allowed")
        self.model, self.view, self.evaluate_fn = model, view, evaluate_fn

    def __call__(self, matrices: Mapping[str, Tensor], head: Mapping[str, Tensor],
                 view: DiagnosticView) -> float:
        if view is not self.view:
            raise PermissionError("unregistered effective readout view")
        modules = lora_modules(self.model)
        if set(matrices) != set(modules) or set(head) != set(HEAD_KEYS):
            raise ValueError("effective readout needs all modules and actual head")
        parameters = dict(self.model.named_parameters())
        before = {name: module.base.weight.detach().clone() for name, module in modules.items()}
        heads = {name: parameters[name].detach().clone() for name in HEAD_KEYS}
        scales = {name: module.scaling for name, module in modules.items()}
        modes = {name: module.training for name, module in self.model.named_modules()}
        rng = _rng_state()
        try:
            with torch.no_grad():
                for name, module in modules.items():
                    effective = matrices[name]
                    if effective.shape != before[name].shape or not bool(torch.isfinite(effective).all()):
                        raise ValueError(f"invalid effective matrix: {name}")
                    module.base.weight.copy_(before[name] + effective.to(
                        device=before[name].device, dtype=before[name].dtype))
                    module.scaling = 0.
                for name in HEAD_KEYS:
                    value = head[name]
                    if value.shape != heads[name].shape or not bool(torch.isfinite(value).all()):
                        raise ValueError(f"invalid head: {name}")
                    parameters[name].copy_(value.to(parameters[name].device,
                                                     parameters[name].dtype))
            result = float(self.evaluate_fn(self.model, view.dataset)["loss"])
            if not math.isfinite(result):
                raise ValueError("nonfinite effective readout loss")
            return result
        finally:
            with torch.no_grad():
                for name, module in modules.items():
                    module.base.weight.copy_(before[name])
                    module.scaling = scales[name]
                for name in HEAD_KEYS:
                    parameters[name].copy_(heads[name])
            for name, module in self.model.named_modules():
                module.training = modes[name]
            _restore_rng(rng)
            if any(not torch.equal(module.base.weight, before[name]) for name, module in modules.items()):
                raise AssertionError("temporary effective readout changed W0")


def collect_direction_diagnostic(
    config: Mapping[str, Any], context: Mapping[str, Any], *,
    probe_fn: Callable = probe_full_gradients,
    payload_fn: Callable = client_payload,
    aggregate_sketches_fn: Callable = aggregate_gradient_sketches,
    select_support_fn: Callable = select_support,
    train_fn: Callable = local_train,
    aggregate_fn: Callable = aggregate,
    load_state_fn: Callable = load_state,
    state_fn: Callable = state,
    original_readout: Callable | None = None,
    effective_readout: Callable | None = None,
    scaling_by_module: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Execute exactly 120 real local calls; return raw bank and both readouts.

    ``context`` is the verified output of diagnostic_model.prepare_context.
    Tests may inject probe/selector/trainer/evaluator callables; production
    defaults use the original v7 implementation. No output file or queue is
    touched here. The caller atomically persists the returned raw tensors.
    """
    clients, round_id = _validate(config, context)
    model = context["model"]
    parent = _clone(context["parent_state"])
    if not _same(state_fn(model), parent):
        raise ValueError("model does not hold the verified parent anchor")
    names = _names(parent)
    if scaling_by_module is None:
        modules = lora_modules(model)
        scaling_by_module = {name: float(module.scaling) for name, module in modules.items()}
    if set(scaling_by_module) != set(names):
        raise ValueError("model LoRA scaling differs from 24 modules")
    view = context["readout"]
    if original_readout is None:
        loss = ModelLoss(model, [view])
        original_readout = lambda values, registered: loss(values, registered.dataset)
    if effective_readout is None:
        effective_readout = TemporaryEffectiveReadout(model, view)
    object_id, factor = config["job_id"], config["factor"]
    weights = {cid: .1 for cid in clients}
    t_probe: dict[str, Any] = {}
    saved_rng = _rng_state()
    try:
        def sketch_for_d_parent(d_parent: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
            load_state_fn(model, d_parent)
            factors = _factors(d_parent)
            payloads = []
            examples = {}
            seeds = {}
            full_gradients = {}
            for cid in clients:
                seed = probe_seed(config["seed"], round_id, cid, factor)
                gradients, count = probe_fn(model, context["clients"][cid], factor,
                                            seed=seed, batch_size=16)
                if count != 32 or set(gradients) != set(names):
                    raise ValueError("T probe needs the same 32 rows and all D projections")
                full_gradients[cid] = _clone(gradients)
                payloads.append(payload_fn(cid, config["anchor_round"], factors,
                    gradients, factor, .0003, config["seed"], round_id,
                    h=32, eps=1e-8, weight_decay=.01))
                examples[cid], seeds[cid] = count, seed
            cbar = aggregate_sketches_fn(payloads, weights,
                                         expected_parent_version=config["anchor_round"])
            if set(cbar) != set(names):
                raise ValueError("T aggregate sketch omitted a projection")
            t_probe.update(examples_by_client=examples, seed_by_client=seeds,
                           full_gradients_by_client=full_gradients,
                           full_gradients_sha256=tensor_digest({f"{cid}/{name}": value
                               for cid, values in full_gradients.items() for name, value in values.items()}),
                           payloads=payloads, cbar_sha256=tensor_digest(cbar),
                           cbar={name: value.detach().cpu().clone() for name, value in cbar.items()})
            return cbar

        variants = build_direction_variants(parent, object_id=object_id,
                                            sketch_for_d_parent=sketch_for_d_parent)
        supports: dict[str, dict[str, dict[int, Any]]] = {}
        prefixes: dict[str, dict[str, dict[int, dict[int, Mapping[str, Tensor]]]]] = {}
        aggregates: dict[str, dict[str, dict[int, Mapping[str, Tensor]]]] = {}
        selection_seeds = {cid: keyed_seed(config["seed"], "selection", cid) for cid in clients}
        training_seeds = {cid: matched_training_seed(object_id, cid) for cid in clients}
        probe_seeds = {cid: {f: probe_seed(config["seed"], round_id, cid, f)
                             for f in ("a", "b")} for cid in clients}
        calls = 0
        for label in VARIANTS:
            variant_parent = variants[label].state
            factors = _factors(variant_parent)
            supports[label], prefixes[label], aggregates[label] = {}, {}, {}
            for method in METHODS:
                supports[label][method], prefixes[label][method], aggregates[label][method] = {}, {}, {}
                for cid in clients:
                    budget = int(context["capacities"][cid])
                    load_state_fn(model, variant_parent)
                    support = select_support_fn(model, context["clients"][cid], method,
                        factor, budget, seed=selection_seeds[cid],
                        visit=int(context["visits"][cid]), batch_size=16,
                        probe_batches=2, probe_seeds=probe_seeds[cid])
                    _check_support(support, factors, factor, budget)
                    supports[label][method][cid] = support
                    load_state_fn(model, variant_parent)
                    result = train_fn(model, context["clients"][cid], support,
                        seed=training_seeds[cid], lr=.0003, batch_size=16,
                        epochs=1, betas=(.9, .999), eps=1e-8,
                        weight_decay=.01, capture_steps=PREFIXES)
                    calls += 1
                    states = result.get("prefix_states")
                    if (result.get("optimizer_steps") != 16 or not isinstance(states, Mapping)
                            or set(states) != set(PREFIXES) or
                            any(set(states[step]) != set(variant_parent) for step in PREFIXES)):
                        raise ValueError("direction trial did not capture one 16-step trajectory")
                    prefixes[label][method][cid] = {step: _clone(states[step]) for step in PREFIXES}
                for step in PREFIXES:
                    children = [prefixes[label][method][cid][step] for cid in clients]
                    aggregate_state = aggregate_fn(variant_parent, children, [1] * 10)
                    _check_aggregate(variant_parent, children, aggregate_state)
                    aggregates[label][method][step] = _clone(aggregate_state)
        if calls != 120:
            raise AssertionError("direction object did not train exactly 120 candidates")

        t_comparator = compare_t_full_h32_directions(
            full_gradients=t_probe["full_gradients_by_client"],
            h32_cbar=t_probe["cbar"], d_parent=variants["D"].state,
            t_parent=variants["T"].state, t_supports=supports["T"],
            clients=clients, factor=factor, seed=config["seed"], round_id=round_id)
        before_loss = float(original_readout(parent, view))
        if not math.isfinite(before_loss):
            raise ValueError("nonfinite direction parent readout")

        readouts = {}
        for label in VARIANTS:
            readouts[label] = {}
            for method in METHODS:
                readouts[label][method] = {}
                for step in PREFIXES:
                    actual = aggregates[label][method][step]
                    original_loss = float(original_readout(actual, view))
                    if not math.isfinite(original_loss):
                        raise ValueError("nonfinite original direction readout")
                    match = norm_match_distinct_parents(
                        variants["N"].state, aggregates["N"][method][step],
                        variants[label].state, actual,
                        scaling_by_module=scaling_by_module)
                    matched_loss = read_norm_matched(
                        match, lambda matrices, head: effective_readout(matrices, head, view))
                    readouts[label][method][step] = {
                        "original_loss": original_loss,
                        "original_benefit": before_loss - original_loss,
                        "normmatched_loss": matched_loss,
                        "normmatched_benefit": None if matched_loss is None else before_loss - matched_loss,
                        "normmatch_status": match.status,
                        "normmatch_multiplier": match.multiplier,
                        "reference_n_effective_norm": match.reference_norm,
                        "candidate_effective_norm": match.candidate_norm,
                    }
        return {"schema": "pcla-v7-direction-diagnostic-bank-1",
                "status": "executed_unreviewed", "object_id": object_id,
                "anchor_sha256": context["anchor"].get("anchor_sha256"),
                "factor": factor, "participants": list(clients),
                "training_seed_by_client": training_seeds,
                "selection_seed_by_client": selection_seeds,
                "probe_seed_by_client": probe_seeds,
                "actual_visits": dict(context["visits"]),
                "capacities": dict(context["capacities"]),
                "t_probe": t_probe,
                "t_full_h32_comparator": t_comparator,
                "parent_readout2048_loss": before_loss,
                "representations": {label: {"active": variants[label].active,
                    "parent_state": _clone(variants[label].state),
                    "parent_state_sha256": tensor_digest(variants[label].state),
                    "numerical_ranks": variants[label].numerical_ranks,
                    "rank_thresholds": variants[label].rank_thresholds,
                    "relative_errors": variants[label].relative_errors,
                    "random_rotations": {name: record.as_dict() for name, record in
                        variants[label].random_rotations.items()}}
                    for label in VARIANTS},
                "supports": supports, "prefix_states": prefixes,
                "aggregate_states": aggregates, "readouts": readouts,
                "local_calls": calls, "optimizer_steps": calls * 16}
    finally:
        try:
            load_state_fn(model, parent)
            if not _same(state_fn(model), parent):
                raise AssertionError("direction diagnostic did not restore parent")
        finally:
            _restore_rng(saved_rng)
