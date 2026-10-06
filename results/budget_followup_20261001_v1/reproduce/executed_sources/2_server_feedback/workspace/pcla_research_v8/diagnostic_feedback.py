"""One-cohort PostV server feedback, using actual factor-space states.

The caller supplies already decoded and aggregated states and an independent
readout NLL callback. No client training, bank loading, or job dispatch occurs.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from .server_control import (
    GRID4,
    _choose,
    _hash_state,
    _preserve_caller,
    _sync,
    apply_server_rule,
    interpolate,
)

RULES = ("ordinary_mean", "scalar4", "scalar17", "block16", "separable")
CORNER_COORDS = {"00": (0.0, 0.0), "10": (1.0, 0.0),
                 "01": (0.0, 1.0), "11": (1.0, 1.0)}


def _direction(gain: float, tolerance: float) -> int:
    return 1 if gain > tolerance else -1 if gain < -tolerance else 0


def _validate_config(config: Mapping[str, Any] | None) -> None:
    if config is None:
        return
    data = config.get("data", {})
    if data.get("selection_role", "public_training_512") != "public_training_512":
        raise ValueError("public selection role differs")
    if data.get("readout_role", "cohort_readout_2048") != "cohort_readout_2048":
        raise ValueError("independent readout role differs")
    if data.get("readout_must_not_select_or_fit", True) is not True:
        raise ValueError("independent readout may not participate in selection")
    controller = config.get("controller", config)
    if tuple(controller.get("grid4", GRID4)) != GRID4:
        raise ValueError("diagnostic grid4 differs from server controller")
    if tuple(controller.get("scalar17", (i / 16 for i in range(17)))) != tuple(i / 16 for i in range(17)):
        raise ValueError("diagnostic scalar17 differs from server controller")
    candidates = controller.get("candidate_rules", RULES)
    if tuple(candidates[:5]) != RULES:
        raise ValueError("diagnostic controller rules differ")
    if controller.get("public_selection_then_independent_readout", True) is not True:
        raise ValueError("independent readout may not participate in selection")


def _json_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("independent readout contains nonfinite output")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"independent readout value is not JSON-compatible: {type(value).__name__}")


def evaluate_feedback(
    model: nn.Module,
    old_parent: Mapping[str, Tensor],
    proposal: Mapping[str, Tensor],
    children: Sequence[Mapping[str, Tensor]],
    weights: Sequence[float],
    public_dataset: Any,
    independent_reader: Callable[[Mapping[str, Tensor]], Any],
    seed: int,
    round_id: int,
    *,
    config: Mapping[str, Any] | None = None,
    posthoc_readout_grid: bool = True,
) -> tuple[dict[str, dict[str, Tensor]], dict[str, Any]]:
    """Return selected CPU states and a JSON-serializable diagnostic record.

    `independent_reader(state)` must evaluate mean NLL on an independent,
    fixed-order readout set. It may return a float or a JSON-compatible mapping
    with `loss` or `nll`, plus accuracy/predictions. It is invoked only after
    all five public selections complete.
    This function restores `model` state, modes, grads, buffers and RNG around
    every callback, so a callback may load the supplied state into `model`.
    """
    _validate_config(config)
    if len(public_dataset) != 512:
        raise ValueError("public training count differs from diagnostic contract")
    if not callable(independent_reader):
        raise TypeError("independent_reader must be callable")
    device = next(model.parameters()).device
    states: dict[str, dict[str, Tensor]] = {}
    controller_records: dict[str, dict[str, Any]] = {}
    server_wall: dict[str, float] = {}
    recipe_id = None if config is None else config.get("job", {}).get("job_id")
    config_sha256 = None if config is None else config.get("expanded_sha256")

    # Each rule gets its own real controller call and measured wall interval.
    # The mean does not consume public data; its public NLL is annotated later
    # from the independently executed block16 grid.
    for label in RULES:
        rule = "mean_r8" if label == "ordinary_mean" else label
        recipe = {"server_rule": rule, "recipe_id": recipe_id,
                  "config_sha256": config_sha256}
        _sync(device)
        started = time.perf_counter()
        accepted, controller = apply_server_rule(
            model, old_parent, proposal, list(children), list(weights),
            None if label == "ordinary_mean" else public_dataset,
            recipe, seed, round_id,
        )
        _sync(device)
        server_wall[label] = time.perf_counter() - started
        states[label] = accepted
        controller_records[label] = controller

    grid = controller_records["block16"]["grid_public_NLL"]
    grid_lookup = {(row["alpha"], row["beta"]): row["loss"] for row in grid}
    if len(grid) != 16 or len(grid_lookup) != 16 or set(grid_lookup) != {
        (a, b) for a in GRID4 for b in GRID4
    }:
        raise RuntimeError("block16 grid is incomplete")
    public_parent = grid_lookup[(0.0, 0.0)]
    public_mean = grid_lookup[(1.0, 1.0)]
    if _hash_state(states["ordinary_mean"]) != _hash_state(proposal):
        raise RuntimeError("ordinary mean differs from decoded proposal")

    readout_cache: dict[str, float] = {}
    readout_payloads: dict[str, dict[str, Any]] = {}
    readout_calls = 0
    readout_seconds = 0.0

    def evaluate(state: Mapping[str, Tensor]) -> tuple[float, str]:
        nonlocal readout_calls, readout_seconds
        digest = _hash_state(state)
        if digest not in readout_cache:
            with _preserve_caller(model):
                _sync(device)
                started = time.perf_counter()
                raw = independent_reader({key: value.detach().clone()
                                          for key, value in state.items()})
                _sync(device)
                readout_seconds += time.perf_counter() - started
            if isinstance(raw, Mapping):
                payload = _json_value(raw)
                if "nll" in payload:
                    value = float(payload["nll"])
                elif "loss" in payload:
                    value = float(payload["loss"])
                else:
                    raise ValueError("independent readout mapping lacks nll/loss")
            else:
                value = float(raw)
                payload = {"nll": value}
            if not math.isfinite(value) or value < 0:
                raise FloatingPointError("independent readout returned invalid mean NLL")
            readout_cache[digest] = value
            readout_payloads[digest] = payload
            readout_calls += 1
        return readout_cache[digest], digest

    # This is the first point where independent readout information enters.
    selected_readout = {rule: evaluate(states[rule])[0] for rule in RULES}
    corners = {}
    for name, (alpha, beta) in CORNER_COORDS.items():
        corner_state = interpolate(old_parent, proposal, alpha, beta)
        loss, digest = evaluate(corner_state)
        corners[name] = {"alpha": alpha, "beta": beta,
                         "public_NLL": grid_lookup[(alpha, beta)],
                         "independent_NLL": loss, "state_hash": digest}
    independent_parent = corners["00"]["independent_NLL"]
    independent_mean = corners["11"]["independent_NLL"]
    tolerance = float((config or {}).get("controller", {}).get("tie_abs_nll", 1e-12))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid gain-direction tolerance")

    rules = {}
    for label in RULES:
        controller = controller_records[label]
        public_loss = public_mean if label == "ordinary_mean" else controller["chosen_public_NLL"]
        public_gain = public_parent - public_loss
        independent_gain = independent_parent - selected_readout[label]
        rules[label] = {
            "alpha": controller["alpha"], "beta": controller["beta"],
            "selected_state_hash": controller["committed_state_hash"],
            "public_NLL": public_loss, "independent_NLL": selected_readout[label],
            "public_NLL_source": "block16_grid_diagnostic_annotation" if label == "ordinary_mean"
                else "own_standalone_public_grid",
            "public_gain_vs_parent": public_gain,
            "independent_gain_vs_parent": independent_gain,
            "public_independent_gain_direction_agrees":
                _direction(public_gain, tolerance) == _direction(independent_gain, tolerance),
            "public_gain_direction": _direction(public_gain, tolerance),
            "independent_gain_direction": _direction(independent_gain, tolerance),
            "standalone_measured_server_wall_seconds": server_wall[label],
            "controller": controller,
        }

    posthoc = None
    if posthoc_readout_grid:
        rows = []
        for alpha in GRID4:
            for beta in GRID4:
                state = interpolate(old_parent, proposal, alpha, beta)
                loss, digest = evaluate(state)
                rows.append({"alpha": alpha, "beta": beta, "loss": loss,
                             "state_hash": digest})
        best = _choose(rows)
        posthoc = {
            "role": "diagnostic_only_after_public_selection",
            "grid_independent_NLL": rows,
            "best_alpha": best["alpha"], "best_beta": best["beta"],
            "best_independent_NLL": best["loss"],
            "best_minus_public_selected_block_independent_NLL":
                best["loss"] - selected_readout["block16"],
        }

    def interaction(key: str) -> float:
        return (corners["10"][key] + corners["01"][key]
                - corners["11"][key] - corners["00"][key])

    record = {
        "format": "pcla-v8-diagnostic-feedback-1",
        "seed": int(seed), "round_id": int(round_id),
        "config_sha256": config_sha256,
        "selection_role": "public_training_512",
        "independent_readout_role": "cohort_readout_2048",
        "selection_completed_before_independent_readout": True,
        "parent_state_hash": _hash_state(old_parent),
        "proposal_state_hash": _hash_state(proposal),
        "rules": rules,
        "four_corners": corners,
        "interaction_I": {
            "definition": "G_joint-G_head-G_LoRA = L10+L01-L11-L00",
            "public": interaction("public_NLL"),
            "independent": interaction("independent_NLL"),
        },
        "comparisons": {
            "block_minus_scalar17_independent_NLL":
                selected_readout["block16"] - selected_readout["scalar17"],
            "block_minus_ordinary_mean_independent_NLL":
                selected_readout["block16"] - independent_mean,
            "block_minus_scalar17_public_NLL":
                rules["block16"]["public_NLL"] - rules["scalar17"]["public_NLL"],
            "block_minus_ordinary_mean_public_NLL":
                rules["block16"]["public_NLL"] - public_mean,
        },
        "posthoc_readout_grid": posthoc,
        "independent_readout_by_state_hash": readout_payloads,
        "counters": {
            "controller_public_feature_examples": sum(
                c["features_count"] for c in controller_records.values()),
            "controller_encoder_batches": sum(
                c["encoder_batches"] for c in controller_records.values()),
            "controller_head_evals": sum(
                c["head_evals"] for c in controller_records.values()),
            "controller_public_dataset_passes": sum(
                c["public_dataset_passes"] for c in controller_records.values()),
            "independent_readout_unique_state_evals": readout_calls,
        },
        "timers": {
            "measurement": "five actual standalone server calls; no shared-grid time attribution",
            "standalone_server_wall_seconds_by_rule": server_wall,
            "standalone_server_wall_seconds_total": sum(server_wall.values()),
            "independent_readout_callback_seconds": readout_seconds,
        },
    }
    return states, record


evaluate_diagnostic_feedback = evaluate_feedback
