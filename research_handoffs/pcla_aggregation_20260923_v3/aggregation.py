from __future__ import annotations

import math
from typing import Mapping

from common import DIAGNOSTIC_RULES, FULL_RULES, require


def accumulator_type(original_class, torch, scaling_by_module: Mapping[str, float]):
    """Return a drop-in accumulator with four aggregation readouts.

    The original runtime calls `DeltaAccumulator(base, total_weight, mode)` before
    it computes the current A/B factor. The factor is therefore inferred at
    `finish()` from coordinate coverage. The batch rejects simultaneous A and B
    coverage because the effective update would contain a dB@dA cross term.
    """

    class StudyAccumulator(original_class):
        def __init__(self, base, total_weight, mode="fixed"):
            require(mode in DIAGNOSTIC_RULES, f"unknown aggregation rule: {mode}")
            super().__init__(base, total_weight, "fixed")
            self.requested_rule = mode

        def _lora_names(self):
            return sorted(n for n in self.base
                          if n.endswith((".lora_a", ".lora_b")))

        def _active_factor(self):
            a = sum(int(torch.count_nonzero(self.coverage[n]))
                    for n in self.base if n.endswith(".lora_a"))
            b = sum(int(torch.count_nonzero(self.coverage[n]))
                    for n in self.base if n.endswith(".lora_b"))
            require(bool(a) ^ bool(b),
                    "aggregation study requires exactly one active LoRA factor")
            return "a" if a else "b"

        def _parameter_norm_sq(self, update):
            return sum(float(update[n].square().sum()) for n in self._lora_names())

        def _function_norm_sq(self, update, factor):
            result = 0.0
            modules = sorted({n.rsplit(".", 1)[0] for n in self._lora_names()})
            require(set(modules) == set(scaling_by_module),
                    "LoRA scaling map differs from adapter state")
            for module in modules:
                scale = float(scaling_by_module[module])
                if factor == "b":
                    delta = update[module + ".lora_b"].double()
                    partner = self.base[module + ".lora_a"].double()
                    matrix = scale * (delta @ partner)
                else:
                    partner = self.base[module + ".lora_b"].double()
                    delta = update[module + ".lora_a"].double()
                    matrix = scale * (partner @ delta)
                result += float(matrix.square().sum())
            require(math.isfinite(result), "nonfinite effective-matrix norm")
            return result

        def _scale_fixed(self, fixed, gain):
            return {
                name: (value * gain
                       if name.endswith((".lora_a", ".lora_b")) else value)
                for name, value in fixed.items()
            }

        def increments(self):
            require(self.seen == self.total_weight, "missing client updates")
            fixed = {n: self.sums[n] / self.total_weight for n in self.base}
            contributors = {
                n: self.sums[n] / self.coverage[n].clamp(min=1)
                for n in self.base
            }
            factor = self._active_factor()
            fp = self._parameter_norm_sq(fixed)
            cp = self._parameter_norm_sq(contributors)
            fw = self._function_norm_sq(fixed, factor)
            cw = self._function_norm_sq(contributors, factor)

            def gain(numerator, denominator, label):
                require(math.isfinite(numerator) and math.isfinite(denominator),
                        f"nonfinite {label} norm")
                if denominator == 0:
                    require(numerator == 0, f"zero fixed but nonzero {label} norm")
                    return 1.0
                value = math.sqrt(numerator / denominator)
                require(math.isfinite(value), f"nonfinite {label} multiplier")
                return value

            gp = gain(cp, fp, "parameter")
            gw = gain(cw, fw, "function")
            parameter_matched = self._scale_fixed(fixed, gp)
            function_matched = self._scale_fixed(fixed, gw)

            for name in self.base:
                if name.startswith("classifier."):
                    require(torch.equal(fixed[name], contributors[name]),
                            f"classifier coverage is incomplete: {name}")
                    require(torch.equal(fixed[name], parameter_matched[name]) and
                            torch.equal(fixed[name], function_matched[name]),
                            "scale control altered classifier increment")

            updates = {
                "fixed": fixed,
                "contributors": contributors,
                "parameter_norm_matched": parameter_matched,
                "function_norm_matched": function_matched,
            }
            diagnostics = {
                "factor": factor,
                "fixed_parameter_norm": math.sqrt(fp),
                "contributors_parameter_norm": math.sqrt(cp),
                "parameter_gain": gp,
                "fixed_function_norm": math.sqrt(fw),
                "contributors_function_norm": math.sqrt(cw),
                "function_gain": gw,
                "parameter_match_relative_error": (
                    abs(self._parameter_norm_sq(parameter_matched) - cp) /
                    max(cp, 1e-30)
                ),
                "function_match_relative_error": (
                    abs(self._function_norm_sq(function_matched, factor) - cw) /
                    max(cw, 1e-30)
                ),
            }
            return updates, diagnostics

        def result_for(self, rule):
            require(rule in DIAGNOSTIC_RULES, "unknown aggregation rule")
            updates, _ = self.increments()
            return {
                name: (self.base[name].double() + updates[rule][name]).to(value.dtype)
                for name, value in self.base.items()
            }

        def readout(self):
            updates, diagnostics = self.increments()
            names = self._lora_names()
            f2 = self._parameter_norm_sq(updates["fixed"])
            c2 = self._parameter_norm_sq(updates["contributors"])
            dot = sum(float((updates["fixed"][n] *
                             updates["contributors"][n]).sum()) for n in names)
            coverage = {}
            for name in names:
                value = self.coverage[name]
                if value.ndim == 2:
                    value = (value[:, 0] if name.endswith(".lora_a")
                             else value[0, :])
                coverage[name] = (value / self.total_weight).tolist()
            head_hashes = {}
            for rule, update in updates.items():
                payload = []
                for name in sorted(n for n in update if n.startswith("classifier.")):
                    payload.append(update[name].detach().cpu().contiguous()
                                   .reshape(-1).view(torch.uint8).numpy().tobytes())
                import hashlib
                h = hashlib.sha256()
                for block in payload:
                    h.update(block)
                head_hashes[rule] = h.hexdigest()
            require(len(set(head_hashes.values())) == 1,
                    "classifier increments differ across aggregation controls")
            return {
                "rule": self.requested_rule,
                **diagnostics,
                "parameter_cosine_F_C": (
                    dot / math.sqrt(f2 * c2) if f2 > 0 and c2 > 0 else None
                ),
                "coverage_fraction": coverage,
                "classifier_increment_sha256": next(iter(head_hashes.values())),
            }

        def finish(self):
            require(self.requested_rule in FULL_RULES,
                    "parameter-norm matching is diagnostic-only")
            return self.result_for(self.requested_rule)

    return StudyAccumulator


def effective_inner(torch, left, right, base, scaling_by_module, factor):
    """Frobenius inner product of represented linear LoRA increments."""
    value = 0.0
    for module, scaling in sorted(scaling_by_module.items()):
        if factor == "b":
            x = left[module + ".lora_b"].double()
            y = right[module + ".lora_b"].double()
            a = base[module + ".lora_a"].double()
            term = ((x.T @ y) * (a @ a.T).T).sum()
        elif factor == "a":
            x = left[module + ".lora_a"].double()
            y = right[module + ".lora_a"].double()
            b = base[module + ".lora_b"].double()
            term = ((b.T @ b) * (x @ y.T)).sum()
        else:
            raise ValueError("effective_inner requires one alternating factor")
        value += float(term) * float(scaling) ** 2
    require(math.isfinite(value), "nonfinite effective-matrix inner product")
    return value


def effective_distance(torch, left, right, base, scaling_by_module, factor):
    difference = {name: left[name] - right[name] for name in left}
    return math.sqrt(max(0.0, effective_inner(
        torch, difference, difference, base, scaling_by_module, factor
    )))
