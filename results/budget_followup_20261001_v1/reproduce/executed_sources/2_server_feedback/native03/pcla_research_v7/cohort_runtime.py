"""Whole-cohort v7 R/J bridge for the federated round driver.

``prepare_cohort`` is a pure simulation of communication until its returned
packets are admitted and committed. It performs real eval probes on client
training data and builds the bytes that would be exchanged. The round driver
must account for ``control_bytes`` plus its ordinary model/delta preview,
then call ``committed_control_cache`` only for an admitted completed round.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping

import torch
from torch import Tensor, nn

from pcla_repair.core import keyed_seed, lora_modules, tensor_digest
from pcla_repair.runtime import batches, to_device
from .coordinates import (map_pair, refresh_due, rotate_balanced_t,
                          rotate_gradient, rotate_with_matrix)
from .coordination import (ClientPayload,
                           client_payload, coordinate_masks, coordination_due,
                           projection, sketch, virtual_adamw_step)
from .selectors import (cyclic_mask, energy_scores, equalized_quotas,
                        layer_mask, module_order, normalized_scores,
                        tied_local_mask, topk_mask)
from .transport import TensorSpec, decode_packet, encode_packet, packet_length
from .timing_hooks import timed, timing_context, measured, span, verification


@dataclass(frozen=True)
class ControlPacket:
    event_id: str
    direction: str
    kind: str
    client_id: int
    raw: bytes

    @property
    def nbytes(self) -> int:
        return len(self.raw)


@dataclass(frozen=True)
class CohortPlan:
    parent: dict[str, Tensor]
    supports: dict[int, dict[str, dict[str, Tensor]]]
    control_packets: tuple[ControlPacket, ...]
    cache_after_controls: dict
    round_id: int
    parent_version: int
    phase: str
    probe_examples: dict[int, int]
    representation_error_max: float
    decision: dict

    @property
    def control_bytes(self) -> int:
        return sum(packet.nbytes for packet in self.control_packets)


@timed("state_and_optimizer_setup")
def _state(model: nn.Module) -> dict[str, Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if n.endswith((".lora_a", ".lora_b")) or n.startswith("classifier.")}


@torch.no_grad()
@timed("state_and_optimizer_setup")
def _load_state(model: nn.Module, values: Mapping[str, Tensor]) -> None:
    params = dict(model.named_parameters())
    if set(values) != set(_state(model)):
        raise ValueError("parent adapter/head keys differ")
    for name, value in values.items():
        if value.shape != params[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid parent tensor: {name}")
        params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))


@timed("technical_logits_check")
def _base_digest(model: nn.Module) -> dict[str, str]:
    result = {}
    for name, module in lora_modules(model).items():
        for parameter_name, value in module.base.named_parameters():
            x = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            result[name + "." + parameter_name] = hashlib.sha256(x).hexdigest()
    return result


def _tensor_sha(value: Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@timed("state_and_optimizer_setup")
def _factors(model: nn.Module) -> dict[str, tuple[Tensor, Tensor]]:
    return {name: (module.lora_b.detach().cpu().clone(), module.lora_a.detach().cpu().clone())
            for name, module in lora_modules(model).items()}


@torch.no_grad()
@timed("technical_logits_check")
def fixed_coordinate_verification_dataset(clients: Mapping[int, object]):
    """Choose one stable train-only 32-row check set for every refresh."""
    if not clients:
        raise ValueError("coordinate logit check needs clients")
    dataset = clients[min(clients)]
    if len(dataset) < 32:
        raise ValueError("coordinate logit check needs 32 fixed train rows")
    return dataset


@timed("technical_logits_check")
def _fixed_train_logits(model: nn.Module, dataset) -> Tensor:
    """The same first 32 train rows for every coordinate check in a run."""
    if len(dataset) < 32:
        raise ValueError("coordinate logit check needs 32 fixed train rows")
    previous_mode = model.training
    model.eval()
    try:
        batch = next(iter(batches(dataset, 32, 0, shuffle=False)))
        result = model(**to_device(batch, next(model.parameters()).device)).logits.detach().cpu()
        if result.shape[0] != 32 or not torch.isfinite(result).all():
            raise FloatingPointError("invalid fixed coordinate logits")
        return result
    finally:
        model.train(previous_mode)


@torch.no_grad()
def _apply_map(model: nn.Module, representation: str,
               sketches: Mapping[str, Tensor] | None = None) -> float:
    worst = 0.
    for name, module in lora_modules(model).items():
        if representation == "T":
            if sketches is None or name not in sketches:
                raise ValueError("T needs this cohort's sketch for every module")
            result = rotate_balanced_t(module.lora_b, module.lora_a, sketches[name])
        else:
            result = map_pair(representation, module.lora_b, module.lora_a)
        module.lora_b.copy_(result.b.to(module.lora_b.device))
        module.lora_a.copy_(result.a.to(module.lora_a.device))
        worst = max(worst, result.relative_error)
    return worst


def _full_support(model: nn.Module, factor: str) -> dict[str, dict[str, Tensor]]:
    return {name: {f: torch.full((module.rank,), f == factor, dtype=torch.bool)
                   for f in ("a", "b")} for name, module in lora_modules(model).items()}


@timed("scoring_and_shared_information")
def probe_full_gradients(model: nn.Module, dataset, factor: str, *,
                         seed: int, batch_size: int = 16,
                         probe_batches: int = 2) -> tuple[dict[str, Tensor], int]:
    """True current-parent eval gradients from two keyed client train batches.

    The compact full support keeps input gradients and shares one dropout path;
    model.eval() disables that dropout during the probe. Training uses its own
    seed and model.train(), which is supplied by the round driver.
    """
    if factor not in ("a", "b") or batch_size < 1 or probe_batches != 2:
        raise ValueError("v7 probe requires two batches and one active factor")
    from .runtime import attach_slices, release_slices

    previous_mode = model.training
    support = _full_support(model, factor)
    attach_slices(model, support, train_head=False)
    device = next(model.parameters()).device
    model.eval()
    gradients = {name: torch.zeros_like(module.lora_a if factor == "a" else module.lora_b,
                                       dtype=torch.float64, device="cpu")
                 for name, module in lora_modules(model).items()}
    count, examples = 0, 0
    try:
        for i, batch in enumerate(batches(dataset, batch_size, seed)):
            if i >= probe_batches:
                break
            model.zero_grad(set_to_none=True)
            output = model(**to_device(batch, device))
            if not torch.isfinite(output.loss):
                raise FloatingPointError("nonfinite eval probe")
            output.loss.backward()
            labels = batch["labels"]
            if len(labels) != batch_size:
                raise ValueError("probe batch smaller than protocol batch_size")
            examples += len(labels)
            for name, module in lora_modules(model).items():
                active = getattr(module, "active_" + factor)
                if active.grad is None:
                    raise RuntimeError(f"missing full probe gradient: {name}")
                g = active.grad.detach().to(device="cpu", dtype=torch.float64)
                gradients[name] += g if factor == "a" else g.T
            count += 1
    finally:
        model.zero_grad(set_to_none=True)
        release_slices(model)
        model.train(previous_mode)
    if count != 2:
        raise ValueError("client has fewer than two full probe batches")
    return {name: value / count for name, value in gradients.items()}, examples


@timed("message_processing")
def _packet(kind: str, direction: str, round_id: int, cid: int,
            tensors: Mapping[str, Tensor], metadata: dict,
            *, mask: list[int] | None = None,
            mask_size: int | None = None) -> ControlPacket:
    raw = encode_packet(kind, round_id - 1, cid, tensors, metadata,
                        mask=mask, mask_size=mask_size)
    if len(raw) != packet_length(kind, round_id - 1, cid, tensors, metadata,
                                 mask=mask, mask_size=mask_size):
        raise AssertionError("coordination packet length mismatch")
    header, decoded = decode_packet(raw)
    if header["version"] != round_id - 1 or header["client_id"] != cid or header["kind"] != kind or set(decoded) != set(tensors):
        raise AssertionError("coordination packet round trip mismatch")
    return ControlPacket(f"r{round_id}:c{cid}:{kind}", direction, kind, cid, raw)


@timed("message_processing")
def _download_packets(kind: str, model_state: Mapping[str, Tensor],
                      cohort: list[int], cache: dict, round_id: int,
                      on_packet: Callable[[ControlPacket], None] | None = None) -> list[ControlPacket]:
    clients = cache.get("clients", {})
    digests = {name: _tensor_sha(value) for name, value in model_state.items()}
    packets = []
    for cid in cohort:
        known = clients.get(str(cid), {}).get("digests", {})
        changed = {name: value for name, value in model_state.items() if known.get(name) != digests[name]}
        packet = _packet(kind, "download", round_id, cid, changed,
                         {"task_init": str(cid) not in clients,
                          "parent_version": round_id - 1, "phase": "probe"})
        if on_packet is not None:
            on_packet(packet)
        packets.append(packet)
    return packets


@timed("message_processing")
def _cache_after(cache: dict, parent: Mapping[str, Tensor], cohort: list[int], version: int) -> dict:
    updated = {"clients": dict(cache.get("clients", {})), "events": list(cache.get("events", []))}
    digests = {name: _tensor_sha(value) for name, value in parent.items()}
    for cid in cohort:
        updated["clients"][str(cid)] = {"version": version, "digests": dict(digests)}
    return updated


def _support(mask: Mapping[str, Tensor], factor: str) -> dict[str, dict[str, Tensor]]:
    return {name: {f: value.clone() if f == factor else torch.zeros_like(value)
                   for f in ("a", "b")} for name, value in mask.items()}


def _qv_pairs(module_names) -> dict[int, tuple[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for name in module_names:
        prefix, target = name.rsplit(".", 1)
        if target not in ("query", "value"):
            raise ValueError("layer/tied selection requires named query/value modules")
        groups.setdefault(prefix, {})[target] = name
    if any(set(pair) != {"query", "value"} for pair in groups.values()):
        raise ValueError("incomplete Q/V layer pair")
    return {i: (groups[prefix]["query"], groups[prefix]["value"])
            for i, prefix in enumerate(module_order(groups))}


def _cyclic_offsets(seed: int, cid: int, factor: str,
                    names: list[str], rank: int) -> dict[str, int]:
    client_seed = keyed_seed(seed, "selection", cid)
    return {name: keyed_seed(client_seed, "cyclic", name, factor) % rank
            for name in names}


def probe_seed(run_seed: int, round_id: int, client_id: int, factor: str) -> int:
    """One method-independent two-batch probe stream, separate from training."""
    if run_seed < 0 or round_id < 1 or client_id < 0 or factor not in ("a", "b"):
        raise ValueError("invalid probe key")
    return keyed_seed(run_seed, "v7_probe", round_id, client_id, factor)


def _mask_notice(round_id: int, cid: int, masks: Mapping[str, Tensor], factor: str) -> ControlPacket:
    names = module_order(masks)
    rank = len(masks[names[0]])
    indices = [i * rank + j for i, name in enumerate(names)
               for j in masks[name].nonzero().flatten().tolist()]
    return _packet("coord_support", "download", round_id, cid, {},
                   {"factor": factor, "module_order": names,
                    "parent_version": round_id - 1},
                   mask=indices, mask_size=len(names) * rank)


@timed("budget_and_return_control")
def preflight_control_bytes(model: nn.Module, parent: Mapping[str, Tensor],
                            cohort: list[int], *, round_id: int, method: str,
                            representation: str, period: int, factor: str,
                            cache: dict | None = None, h: int = 32) -> int:
    """Exact fixed-size coordination charge before computing any probe gradient.

    The returned count includes current-parent downloads, all sketches and
    support/rotation notices. Actual tensor values and selected mask bits
    cannot change the encoded length under this wire format.
    """
    if not cohort or len(set(cohort)) != len(cohort) or factor not in ("a", "b"):
        raise ValueError("invalid preflight cohort")
    coordinated = method in ("JL", "JQ") and coordination_due(round_id, period)
    t_refresh = representation == "T" and refresh_due(round_id, period)
    if not coordinated and not t_refresh:
        return 0
    if coordinated and representation != "N":
        raise NotImplementedError("J coordination requires N coordinates")
    cache = cache or {}
    probe_parent = {name: value.detach().cpu().clone() for name, value in parent.items()}
    modules = lora_modules(model)
    names = module_order(modules)
    if t_refresh:
        for name in names:
            result = map_pair("D", parent[name + ".lora_b"], parent[name + ".lora_a"])
            probe_parent[name + ".lora_b"] = result.b
            probe_parent[name + ".lora_a"] = result.a
    kind = "t_d_parent" if t_refresh else "coord_parent"
    total = sum(packet.nbytes for packet in _download_packets(kind, probe_parent, cohort, cache, round_id))
    ranks = {modules[name].rank for name in names}
    if len(ranks) != 1:
        raise ValueError("coordination requires common projection rank")
    rank = ranks.pop()
    if t_refresh:
        sketch_specs = {name: TensorSpec((h, rank)) for name in names}
        rotation_specs = {name: TensorSpec((rank, rank), "float64") for name in names}
        md = {"from_digest": tensor_digest(probe_parent),
              "to_digest": "0" * 64, "parent_version": round_id - 1,
              "factor": factor}
        for cid in cohort:
            total += packet_length("t_grad_sketch", round_id - 1, cid, sketch_specs,
                                   {"parent_version": round_id - 1, "factor": factor})
            total += packet_length("t_rotation", round_id - 1, cid, rotation_specs, md)
    else:
        specs = {f"{name}.{suffix}": TensorSpec((rank,) if suffix == "score" else (h, rank))
                 for name in names for suffix in ("gradient", "update", "score")}
        mask_size = len(names) * rank
        for cid in cohort:
            total += packet_length("coord_probe", round_id - 1, cid, specs,
                                   {"parent_version": round_id - 1, "factor": factor})
            total += packet_length("coord_support", round_id - 1, cid, {},
                                   {"factor": factor, "module_order": names,
                                    "parent_version": round_id - 1},
                                   mask=[], mask_size=mask_size)
    return total


def preflight_round_upper_bound(model: nn.Module, parent: Mapping[str, Tensor],
                                cohort: list[int], budgets: Mapping[int, int], *,
                                round_id: int, method: str, representation: str,
                                period: int, factor: str, cache: dict | None = None,
                                h: int = 32,
                                force_full_download: bool = False) -> int:
    """Safe exact-wire upper bound before any coordinated probe or message.

    For J/T control rounds the probe-parent, sketch, support/rotation packet
    lengths are exact; the following train download is an empty encoded
    notification because clients reconstruct/cache the final parent. The
    sparse delta's tensor names and row counts depend on the selected support,
    so its header is bounded by including every active module at full rank,
    while its payload uses the exact K times the largest projection width.
    This may reject a borderline round early but can never treat a sent probe
    packet as free. The returned bound covers all download and upload bytes.
    """
    if set(budgets) != set(cohort):
        raise ValueError("preflight budget/cohort mismatch")
    cache = cache or {}
    modules = lora_modules(model)
    names = module_order(modules)
    ranks = {modules[name].rank for name in names}
    if len(ranks) != 1 or factor not in ("a", "b"):
        raise ValueError("preflight requires common rank and one active factor")
    rank = ranks.pop()
    for cid in cohort:
        if budgets[cid] < 0 or budgets[cid] > len(names) * rank:
            raise ValueError("preflight budget exceeds position capacity")
    control = preflight_control_bytes(model, parent, cohort, round_id=round_id,
                                      method=method, representation=representation,
                                      period=period, factor=factor, cache=cache, h=h)
    coordinated = method in ("JL", "JQ") and coordination_due(round_id, period)
    t_refresh = representation == "T" and refresh_due(round_id, period)
    total = control
    head_specs = {name: TensorSpec(tuple(value.shape), str(value.detach().cpu().numpy().dtype))
                  for name, value in parent.items() if name.startswith("classifier.")}
    if not head_specs:
        raise ValueError("head state required for upload accounting")
    active_specs = {}
    max_position_bytes = 0
    for name in names:
        tensor = parent[name + ".lora_" + factor]
        width = tensor.shape[1] if factor == "a" else tensor.shape[0]
        max_position_bytes = max(max_position_bytes, width * tensor.element_size())
        active_specs[name + ".lora_" + factor] = TensorSpec(tuple(tensor.shape), str(tensor.detach().cpu().numpy().dtype))
    full_specs = {**active_specs, **head_specs}
    full_factor_bytes = sum(spec.nbytes for spec in active_specs.values())
    mask_size = len(names) * 2 * rank
    for cid in cohort:
        if force_full_download:
            changed = parent
            task_init = str(cid) not in cache.get("clients", {})
        elif coordinated or t_refresh:
            changed = {}
            task_init = False
        else:
            known = cache.get("clients", {}).get(str(cid), {}).get("digests", {})
            changed = {name: value for name, value in parent.items()
                       if known.get(name) != _tensor_sha(value)}
            task_init = str(cid) not in cache.get("clients", {})
        total += packet_length("model_download", round_id - 1, cid, changed,
                               {"task_init": task_init, "parent_version": round_id - 1})
        # All names present and all count fields at their maximal digit width
        # bound any support-specific tensor descriptor header.
        full_length = packet_length("delta_upload", round_id - 1, cid, full_specs,
                                    {"parent_version": round_id - 1, "delta": True},
                                    mask=[], mask_size=mask_size)
        total += full_length - full_factor_bytes + budgets[cid] * max_position_bytes
    return total


def assert_preflight_covers_actual(upper_bound: int, actual_charge: int,
                                   *, round_id: int) -> None:
    """Fail closed if an actual coordinated wire charge exceeds its admission bound."""
    if upper_bound < 0 or actual_charge < 0 or round_id < 1:
        raise ValueError("invalid preflight charge")
    if actual_charge > upper_bound:
        raise AssertionError(
            f"round {round_id} coordinated charge {actual_charge} exceeds preflight {upper_bound}")


@timing_context("coordinate_and_server_decision")
def prepare_cohort(model: nn.Module, parent: Mapping[str, Tensor], clients: Mapping[int, object],
                   cohort: list[int], budgets: Mapping[int, int], visits: Mapping[int, int],
                   *, round_id: int, seed: int, factor: str, method: str,
                   representation: str = "N", period: int = 1,
                   learning_rate: float, cache: dict | None = None,
                   weights: Mapping[int, float] | None = None,
                   xi: float = 1., h: int = 32, eps: float = 1e-8,
                   weight_decay: float = .01, batch_size: int = 16,
                   on_control_packet: Callable[[ControlPacket], None] | None = None,
                   timing=None) -> CohortPlan:
    """Construct one executable R/J cohort decision and actual control packets.

    This changes the supplied model's LoRA factors to the returned parent, but
    leaves all base W0 weights byte-identical. It never performs local training
    and never reads development/diagnostic examples.
    """
    if not cohort or len(set(cohort)) != len(cohort) or set(cohort) - set(clients):
        raise ValueError("invalid cohort")
    if set(cohort) != set(budgets) or set(cohort) != set(visits):
        raise ValueError("budget/visit cohort mismatch")
    if round_id < 1 or period < 1 or seed < 0 or factor not in ("a", "b"):
        raise ValueError("invalid round metadata")
    if method not in ("local", "cyclic", "tied_local", "layer", "energy",
                      "equal_quota_cyclic", "JL", "JQ") or representation not in ("N", "D", "S", "T"):
        raise NotImplementedError(f"cohort bridge lacks {method}/{representation}")
    if method in ("JL", "JQ") and representation != "N":
        raise NotImplementedError("v7 J line uses N coordinates")
    if weights is None:
        sizes = {cid: len(clients[cid]) for cid in cohort}
        total = sum(sizes.values())
        weights = {cid: sizes[cid] / total for cid in cohort}
    if set(weights) != set(cohort):
        raise ValueError("aggregation weights must cover cohort")
    cache = cache or {}
    expected_control_bytes = verification(preflight_control_bytes,
        model, parent, cohort, round_id=round_id, method=method,
        representation=representation, period=period, factor=factor,
        cache=cache, h=h)
    _load_state(model, parent)
    w0_before = _base_digest(model)
    packets: list[ControlPacket] = []
    def append_packet(packet: ControlPacket) -> None:
        if on_control_packet is not None:
            on_control_packet(packet)
        packets.append(packet)
    probe_examples: dict[int, int] = {}
    error = 0.
    phase = "ordinary"
    coordinated = method in ("JL", "JQ") and coordination_due(round_id, period)
    t_refresh = representation == "T" and refresh_due(round_id, period)
    fixed_logits_before = (_fixed_train_logits(model, fixed_coordinate_verification_dataset(clients))
                           if t_refresh else None)
    if representation in ("D", "S") and refresh_due(round_id, period):
        error = _apply_map(model, representation)
        phase = "representation_refresh"
    if t_refresh:
        error = _apply_map(model, "D")
        phase = "T_refresh"
    probe_parent = _state(model)

    # Probe parents are sent before client-side gradient collection. Ordinary
    # rounds let RawTensorWire perform the model download after mask selection.
    if coordinated or t_refresh:
        kind = "t_d_parent" if t_refresh else "coord_parent"
        packets += _download_packets(kind, probe_parent, cohort, cache, round_id,
                                     on_packet=on_control_packet)

    gradients: dict[int, dict[str, Tensor]] = {}
    factors = _factors(model)
    for cid in cohort:
        if coordinated or t_refresh or method in ("local", "tied_local", "layer") or (method in ("JL", "JQ") and not coordinated):
            _load_state(model, probe_parent)
            gradients[cid], probe_examples[cid] = probe_full_gradients(
                model, clients[cid], factor,
                seed=probe_seed(seed, round_id, cid, factor),
                batch_size=batch_size)

    decision = {}
    masks: dict[int, dict[str, Tensor]] = {}
    if t_refresh:
        # Only one gradient sketch is uploaded. The same D parent and same
        # projection are used by all clients in this round.
        t_payloads = []
        for cid in cohort:
            c = {}
            for name, gradient in gradients[cid].items():
                width = gradient.shape[0 if factor == "b" else 1]
                s = measured("scoring_and_shared_information", projection, seed, round_id, name, factor, width, h)
                c[name] = measured("scoring_and_shared_information", sketch, gradient, s, factor)
            append_packet(_packet("t_grad_sketch", "upload", round_id, cid, c,
                                  {"parent_version": round_id - 1, "factor": factor}))
            # Aggregate only G sketches; T does not send predicted-U sketches.
            t_payloads.append(ClientPayload(cid, c, {}, {}, round_id - 1))
        cbar = {name: sum((float(weights[p.client_id]) * p.gradient_sketches[name].double()
                           for p in t_payloads), torch.zeros_like(t_payloads[0].gradient_sketches[name], dtype=torch.float64))
                for name in factors}
        _load_state(model, probe_parent)
        rotations = {}
        for name, module in lora_modules(model).items():
            result = rotate_balanced_t(module.lora_b, module.lora_a, cbar[name])
            with torch.no_grad():
                module.lora_b.copy_(result.b.to(module.lora_b.device))
                module.lora_a.copy_(result.a.to(module.lora_a.device))
            rotations[name] = result.rotation
            error = max(error, result.relative_error)
        final_parent = _state(model)
        with span("technical_logits_check"):
            for name, rotation in rotations.items():
                reconstructed = rotate_with_matrix(probe_parent[name + ".lora_b"],
                                                     probe_parent[name + ".lora_a"], rotation)
                if (not torch.equal(reconstructed.b, final_parent[name + ".lora_b"])
                        or not torch.equal(reconstructed.a, final_parent[name + ".lora_a"])):
                    raise AssertionError("client/server T adapter digest mismatch")
        with span("message_processing"):
            md = {"from_digest": tensor_digest(probe_parent),
                  "to_digest": tensor_digest(final_parent),
                  "parent_version": round_id - 1, "factor": factor}
        for cid in cohort:
            append_packet(_packet("t_rotation", "download", round_id, cid, rotations, md))
        # Client-side chain-rule conversion and a new virtual AdamW step.
        factors = _factors(model)
        for cid in cohort:
            transformed = {name: measured("scoring_and_shared_information", rotate_gradient, gradients[cid][name], rotations[name], factor)
                           for name in factors}
            gradients[cid] = transformed
            for name, (b, a) in factors.items():
                parameter = a if factor == "a" else b
                measured("scoring_and_shared_information", virtual_adamw_step, parameter, transformed[name], learning_rate,
                                   eps=eps, weight_decay=weight_decay)
        with span("technical_logits_check"):
            decision["rotation_digest"] = tensor_digest(rotations)
            fixed_logits_after = _fixed_train_logits(model, fixed_coordinate_verification_dataset(clients))
            logit_error = float((fixed_logits_after - fixed_logits_before).abs().max())
            if logit_error > 1e-4:
                raise FloatingPointError(f"T refresh logits differ by {logit_error}")
            decision["logits_max_absolute_error"] = logit_error
    else:
        final_parent = probe_parent

    if coordinated:
        phase = method + "_coordination"
        payloads = []
        for cid in cohort:
            payload = measured("scoring_and_shared_information", client_payload, cid, round_id - 1, factors, gradients[cid], factor,
                                     learning_rate, seed, round_id, h=h,
                                     eps=eps, weight_decay=weight_decay)
            payloads.append(payload)
            tensors = {f"{name}.gradient": value for name, value in payload.gradient_sketches.items()}
            tensors.update({f"{name}.update": value for name, value in payload.update_sketches.items()})
            tensors.update({f"{name}.score": value for name, value in payload.local_scores.items()})
            append_packet(_packet("coord_probe", "upload", round_id, cid, tensors,
                                  {"parent_version": round_id - 1, "factor": factor}))
        names = module_order(factors)
        rank = next(iter(factors.values()))[0].shape[1]
        offsets = {cid: _cyclic_offsets(seed, cid, factor, names, rank) for cid in cohort}
        result = coordinate_masks(payloads, weights, budgets, factors, factor, visits,
                                  method, xi=xi, offsets_by_client=offsets,
                                  expected_parent_version=round_id - 1)
        masks = result.masks
        for cid in cohort:
            append_packet(_mask_notice(round_id, cid, masks[cid], factor))
        decision.update({"lambda": result.lambda_value,
                         "objective_initial": result.objective_initial,
                         "objective_final": result.objective_final,
                         "choices": result.choices})
    else:
        names = module_order(factors)
        ranks = {len(factors[n][1]) for n in names}
        if len(ranks) != 1:
            raise ValueError("cyclic currently requires a common projection rank")
        rank = ranks.pop()
        allocation = equalized_quotas(budgets, round_id) if method == "equal_quota_cyclic" else budgets
        pairs = _qv_pairs(names) if method in ("layer", "tied_local") else None
        for cid in cohort:
            if method in ("cyclic", "equal_quota_cyclic"):
                masks[cid] = cyclic_mask(names, rank, allocation[cid], visits[cid], factor=factor,
                                         offsets=_cyclic_offsets(seed, cid, factor, names, rank))
            elif method == "energy":
                masks[cid] = topk_mask(energy_scores(factors), allocation[cid])
            else:
                scores = measured("scoring_and_shared_information", normalized_scores, factors, gradients[cid], factor)
                if method == "layer":
                    masks[cid] = layer_mask(scores, allocation[cid], pairs)
                elif method == "tied_local":
                    masks[cid] = tied_local_mask(scores, allocation[cid], pairs)
                else:
                    masks[cid] = topk_mask(scores, allocation[cid])

    if _base_digest(model) != w0_before:
        raise AssertionError("v7 bridge changed immutable W0")
    # A coordinating client already holds the final parent after applying the
    # rotation or after loading the JL/JQ probe parent. This updated cache
    # prevents a second full tensor download during ordinary wire.preview.
    after_cache = _cache_after(cache, final_parent, cohort, round_id - 1) if (coordinated or t_refresh) else cache
    plan = CohortPlan(final_parent, {cid: _support(masks[cid], factor) for cid in cohort},
                      tuple(packets), after_cache, round_id, round_id - 1,
                      phase, probe_examples, error, decision)
    with span("technical_equivalence_check"):
        if plan.control_bytes != expected_control_bytes:
            raise AssertionError("preflight coordination bytes differ from encoded packets")
    return plan


def committed_control_cache(plan: CohortPlan, *, diagnostic: bool) -> dict:
    """Apply actual control-packet events only after the round is admitted."""
    cache = {"clients": dict(plan.cache_after_controls.get("clients", {})),
             "events": list(plan.cache_after_controls.get("events", []))}
    existing = {event["event_id"] for event in cache["events"]}
    bucket = "diagnostic_after_budget" if diagnostic else "deployed_prefix"
    for packet in plan.control_packets:
        if packet.event_id in existing:
            raise ValueError("duplicate coordination packet event")
        cache["events"].append({"event_id": packet.event_id, "round": plan.round_id,
                                "direction": packet.direction, "bytes": packet.nbytes,
                                "bucket": bucket})
    return cache
