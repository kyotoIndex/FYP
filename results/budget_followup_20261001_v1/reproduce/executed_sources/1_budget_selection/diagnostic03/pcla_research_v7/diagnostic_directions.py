"""Pure fixed-parent direction variants and effective-update readout bridge.

The caller supplies the actual same-cohort h32 sketch measured on the D parent.
The S nonzero-space rotation uses a separate reproducible CPU generator. This
module does not choose a cohort, use readout labels, train, or mutate its parent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Callable, Mapping

import torch
from torch import Tensor

from pcla_repair.core import keyed_seed
from .coordinates import balance_d, rotate_balanced_t, rotate_with_matrix, split_s
from .data import PROTOCOL


VARIANTS = ("N", "D", "S", "T", "S_random_nonzero", "S_zero_removed")
N_MODULES = 24
HEAD_KEYS = ("classifier.weight", "classifier.bias")


@dataclass(frozen=True)
class RotationRecord:
    module: str
    seed: int
    matrix: Tensor
    matrix_sha256: str

    def as_dict(self) -> dict:
        """JSON-safe seed, exact Q and Q hash for durable recording."""
        return {"module": self.module, "seed": self.seed,
                "matrix_sha256": self.matrix_sha256, "matrix": self.matrix.tolist()}


@dataclass(frozen=True)
class DirectionVariant:
    name: str
    state: dict[str, Tensor]
    active: bool
    numerical_ranks: dict[str, int]
    rank_thresholds: dict[str, float]
    relative_errors: dict[str, float]
    random_rotations: dict[str, RotationRecord] = field(default_factory=dict)


@dataclass(frozen=True)
class NormMatchedReadout:
    status: str
    multiplier: float | None
    reference_norm: float
    candidate_norm: float
    matched_effective: dict[str, Tensor] | None
    head: dict[str, Tensor]


def _names(state: Mapping[str, Tensor]) -> tuple[str, ...]:
    names = tuple(sorted(key[:-7] for key in state if key.endswith(".lora_a")))
    expected = {key for name in names for key in (name + ".lora_a", name + ".lora_b")}
    expected.update(HEAD_KEYS)
    if len(names) != N_MODULES or set(state) != expected:
        raise ValueError("direction state requires exactly 24 LoRA pairs and the head")
    for name in names:
        b, a = state[name + ".lora_b"], state[name + ".lora_a"]
        if (b.ndim != 2 or a.ndim != 2 or b.shape[1] != a.shape[0]
                or b.shape[1] > min(b.shape[0], a.shape[1])
                or b.device.type != "cpu" or a.device.type != "cpu"
                or b.dtype != torch.float32 or a.dtype != torch.float32
                or not bool(torch.isfinite(b).all()) or not bool(torch.isfinite(a).all())):
            raise ValueError(f"invalid CPU float32 LoRA pair: {name}")
    for key in HEAD_KEYS:
        value = state[key]
        if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid head: {key}")
    return names


def _copy(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().clone() for key, value in state.items()}


def matched_training_seed(object_id: str, client_id: int) -> int:
    """Frozen diagnostic bank key, shared across local/cyclic and directions.

    The object ID already binds factor, anchor and seed. This is identical to
    diagnostic_bank._seed(object_id, client_id).
    """
    if not isinstance(object_id, str) or not object_id or type(client_id) is not int or client_id < 0:
        raise ValueError("need diagnostic object ID and nonnegative client ID")
    return keyed_seed(object_id, client_id)


def sample_s_nonzero_rotation(object_id: str, module: str, k: int) -> RotationRecord:
    """Frozen CPU Gaussian QR rule; leaves all global RNG streams untouched."""
    if (not isinstance(object_id, str) or not object_id or
            not isinstance(module, str) or not module or type(k) is not int or k < 0):
        raise ValueError("need object ID, module name and nonnegative numerical rank")
    seed = keyed_seed(PROTOCOL, "diagnostic_S_nonzero_rotation_v1", object_id, module)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if k == 0:
        q = torch.empty((0, 0), dtype=torch.float64, device="cpu")
    else:
        gaussian = torch.randn((k, k), dtype=torch.float64, device="cpu", generator=generator)
        q, r = torch.linalg.qr(gaussian)
        signs = torch.sign(torch.diag(r))
        signs[signs == 0] = 1.
        q = (q * signs.unsqueeze(0)).contiguous()
    raw = b"pcla-s-rotation-float64-v1\0" + k.to_bytes(8, "big") + q.numpy().tobytes(order="C")
    return RotationRecord(module, seed, q, hashlib.sha256(raw).hexdigest())


def build_direction_variants(
    parent: Mapping[str, Tensor], *,
    object_id: str,
    sketch_for_d_parent: Callable[[Mapping[str, Tensor]], Mapping[str, Tensor]],
) -> dict[str, DirectionVariant]:
    """Force one N/D/S/T intervention at this parent, independent of period.

    ``sketch_for_d_parent`` must compute the current ten-client aggregate h32
    on the supplied D state. The caller owns its probe seeds and provenance.
    S rotations are keyed by protocol/object/module, and each seed, Q and Q
    hash is returned for durable recording.
    """
    names = _names(parent)
    states = {name: _copy(parent) for name in VARIANTS}
    ranks: dict[str, dict[str, int]] = {name: {} for name in VARIANTS}
    thresholds: dict[str, dict[str, float]] = {name: {} for name in VARIANTS}
    errors: dict[str, dict[str, float]] = {name: {} for name in VARIANTS}
    s_results = {}
    for module in names:
        b, a = parent[module + ".lora_b"], parent[module + ".lora_a"]
        d = balance_d(b, a)
        s = split_s(b, a)
        s_results[module] = s
        for label, result in (("D", d), ("S", s)):
            states[label][module + ".lora_b"] = result.b
            states[label][module + ".lora_a"] = result.a
            ranks[label][module] = result.numerical_rank
            thresholds[label][module] = result.rank_threshold
            errors[label][module] = result.relative_error
        ranks["N"][module] = d.numerical_rank
        thresholds["N"][module] = 0.
        errors["N"][module] = 0.

    # T must probe the D parent, then rotate exactly those D factors.
    sketches = sketch_for_d_parent(_copy(states["D"]))
    if not isinstance(sketches, Mapping) or set(sketches) != set(names):
        raise ValueError("T needs one current-cohort h32 sketch per module")
    for module in names:
        cbar = sketches[module]
        rank = states["D"][module + ".lora_a"].shape[0]
        if (not isinstance(cbar, Tensor) or cbar.shape != (32, rank)
                or not bool(torch.isfinite(cbar).all())):
            raise ValueError(f"T requires finite h32 by rank sketch: {module}")
        t = rotate_balanced_t(states["D"][module + ".lora_b"],
                              states["D"][module + ".lora_a"], cbar)
        states["T"][module + ".lora_b"] = t.b
        states["T"][module + ".lora_a"] = t.a
        ranks["T"][module] = t.numerical_rank
        thresholds["T"][module] = t.rank_threshold
        errors["T"][module] = t.relative_error

    rotation_records = {}
    random_changed = False
    zero_changed = False
    for module in names:
        s = s_results[module]
        rank = s.a.shape[0]
        k = s.numerical_rank
        record = sample_s_nonzero_rotation(object_id, module, k)
        rotation_records[module] = record
        full = torch.eye(rank, dtype=torch.float64)
        full[:k, :k] = record.matrix
        rotated = rotate_with_matrix(s.b, s.a, full)
        states["S_random_nonzero"][module + ".lora_b"] = rotated.b
        states["S_random_nonzero"][module + ".lora_a"] = rotated.a
        random_changed |= not (torch.equal(rotated.b, s.b) and torch.equal(rotated.a, s.a))
        zero_b, zero_a = s.b.clone(), s.a.clone()
        zero_b[:, k:] = 0
        zero_a[k:, :] = 0
        states["S_zero_removed"][module + ".lora_b"] = zero_b
        states["S_zero_removed"][module + ".lora_a"] = zero_a
        zero_changed |= not (torch.equal(zero_b, s.b) and torch.equal(zero_a, s.a))
        for label, error in (("S_random_nonzero", rotated.relative_error),
                             ("S_zero_removed", s.relative_error)):
            ranks[label][module] = k
            thresholds[label][module] = s.rank_threshold
            errors[label][module] = error
    return {name: DirectionVariant(name, states[name],
            (random_changed if name == "S_random_nonzero" else
             zero_changed if name == "S_zero_removed" else True),
            ranks[name], thresholds[name], errors[name],
            rotation_records if name == "S_random_nonzero" else {}) for name in VARIANTS}


def norm_match_to_n(parent: Mapping[str, Tensor], n_aggregated: Mapping[str, Tensor],
                    candidate_aggregated: Mapping[str, Tensor], *,
                    scaling_by_module: Mapping[str, float] | None = None) -> NormMatchedReadout:
    """Match the *whole* 24-module effective delta to N with one multiplier.

    The returned effective matrices are temporary readout inputs (not factors
    or a trainable state). The candidate's actually aggregated head is copied
    unchanged. The caller applies these matrices over the unchanged W0 for
    model evaluation, then restores the model.
    """
    names = _names(parent)
    if _names(n_aggregated) != names or _names(candidate_aggregated) != names:
        raise ValueError("norm match states have different modules")
    scaling = {name: 1. for name in names} if scaling_by_module is None else dict(scaling_by_module)
    if set(scaling) != set(names) or any(not math.isfinite(float(v)) or float(v) <= 0
                                      for v in scaling.values()):
        raise ValueError("need one positive finite LoRA scale per module")
    parent_effective = {}
    n_delta = {}
    candidate_delta = {}
    n_sq = 0.
    candidate_sq = 0.
    for name in names:
        def effective(state: Mapping[str, Tensor]) -> Tensor:
            return (state[name + ".lora_b"].double() @
                    state[name + ".lora_a"].double()) * float(scaling[name])
        base = effective(parent)
        nd = effective(n_aggregated) - base
        cd = effective(candidate_aggregated) - base
        parent_effective[name], n_delta[name], candidate_delta[name] = base, nd, cd
        n_sq += float(nd.square().sum())
        candidate_sq += float(cd.square().sum())
    head = {key: candidate_aggregated[key].detach().clone() for key in HEAD_KEYS}
    reference_norm, candidate_norm = math.sqrt(n_sq), math.sqrt(candidate_sq)
    if n_sq == 0 or candidate_sq == 0:
        return NormMatchedReadout("NA_zero_reference" if n_sq == 0 else
                                  "NA_zero_candidate", None, reference_norm,
                                  candidate_norm, None, head)
    multiplier = math.sqrt(n_sq / candidate_sq)
    if not math.isfinite(multiplier):
        raise FloatingPointError("nonfinite norm-match multiplier")
    matched = {name: parent_effective[name] + multiplier * candidate_delta[name]
               for name in names}
    return NormMatchedReadout("matched", multiplier, reference_norm,
                              candidate_norm, matched, head)


def read_norm_matched(match: NormMatchedReadout,
                      evaluate_effective: Callable[[Mapping[str, Tensor],
                                                    Mapping[str, Tensor]], float]) -> float | None:
    """Inject the real registered-view evaluator; never evaluate an NA delta.

    The evaluator temporarily installs ``matched_effective`` over fixed W0
    and ``head`` as-is, computes one readout, and restores model/RNG state.
    That model-specific operation belongs to the caller's diagnostic bridge.
    """
    if match.status != "matched":
        return None
    if match.matched_effective is None:
        raise ValueError("matched readout lacks effective matrices")
    value = float(evaluate_effective(match.matched_effective, match.head))
    if not math.isfinite(value):
        raise ValueError("norm-matched readout is nonfinite")
    return value
