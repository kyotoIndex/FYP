"""Masked aggregation and an effective-weight (not output-logit) norm control."""
from __future__ import annotations
import math
import torch
from common import require

RULES = ('fixed', 'contributors', 'parameter_norm_matched', 'function_norm_matched')


def prefixes(base):
    return sorted(n[:-7] for n in base if n.endswith('.lora_a'))


def validate_schema(base, scalings):
    names = prefixes(base)
    require(names and set(names) == set(scalings), 'LoRA module/scaling inventory differs')
    allowed = set()
    for name in names:
        a, b = base[name + '.lora_a'], base[name + '.lora_b']
        require(a.ndim == b.ndim == 2 and a.shape[0] == b.shape[1], 'A/B shapes differ')
        require(math.isfinite(scalings[name]) and scalings[name] > 0, 'invalid LoRA scaling')
        allowed.update((name + '.lora_a', name + '.lora_b'))
    require(all(n in allowed or n.startswith('classifier.') for n in base), 'unexpected parameter')
    require(all(v.is_floating_point() and bool(torch.isfinite(v).all()) for v in base.values()),
            'invalid base state')


def effective_inner(left, right, base, scalings, factor):
    """Exact trace identity in float64; O(d*r²), not dense O(d²*r)."""
    require(factor in ('a', 'b'), 'effective norm is defined for alternating updates')
    terms = []
    for name in prefixes(base):
        x, y = left[name + '.lora_' + factor].double(), right[name + '.lora_' + factor].double()
        if factor == 'b':
            a = base[name + '.lora_a'].double()
            value = ((x.T @ y) * (a @ a.T).T).sum()
        else:
            b = base[name + '.lora_b'].double()
            value = ((b.T @ b) * (x @ y.T)).sum()
        terms.append(float(scalings[name]) ** 2 * float(value))
    result = math.fsum(terms)
    require(math.isfinite(result), 'nonfinite effective inner product')
    return result


def effective_norm_sq(update, base, scalings, factor):
    """Use a dense fallback only for numerical cancellation near a null direction."""
    result = effective_inner(update, update, base, scalings, factor)
    scale = 0.0
    for name in prefixes(base):
        active = update[name + '.lora_' + factor].double()
        other = base[name + ('.lora_a' if factor == 'b' else '.lora_b')].double()
        scale += float(active.square().sum()) * float(other.square().sum()) * scalings[name] ** 2
    if result <= 64 * torch.finfo(torch.float64).eps * scale:
        result = 0.0
        for name in prefixes(base):
            u = update[name + '.lora_' + factor].double()
            w = (u @ base[name + '.lora_a'].double() if factor == 'b'
                 else base[name + '.lora_b'].double() @ u)
            result += float(w.square().sum()) * scalings[name] ** 2
    require(result >= 0 and math.isfinite(result), 'invalid norm')
    return result


def compare_updates(update, reference, base, scalings, factor):
    x2 = effective_norm_sq(update, base, scalings, factor)
    y2 = effective_norm_sq(reference, base, scalings, factor)
    xy = effective_inner(update, reference, base, scalings, factor)
    diff = {n: update[n] - reference[n] for n in base}
    error2 = effective_norm_sq(diff, base, scalings, factor)
    return {'norm': math.sqrt(x2), 'reference_norm': math.sqrt(y2),
            'cosine': max(-1.0, min(1.0, xy / math.sqrt(x2*y2))) if x2 and y2 else None,
            'relative_error': math.sqrt(error2/y2) if y2 else None,
            'norm_ratio': math.sqrt(x2/y2) if y2 else None}


class AggregationSnapshot:
    def __init__(self, base, total_weight, scalings, factor):
        require(isinstance(total_weight, int) and total_weight > 0, 'invalid total weight')
        require(factor in ('a', 'b'), 'alternating factor required')
        validate_schema(base, scalings)
        self.base = {n: v.detach().cpu().clone() for n, v in base.items()}
        self.sums = {n: torch.zeros_like(v, dtype=torch.float64) for n, v in self.base.items()}
        self.coverage = {n: torch.zeros_like(v, dtype=torch.float64) for n, v in self.base.items()}
        self.total_weight, self.scalings, self.factor = total_weight, dict(scalings), factor
        self.seen, self.clients, self._cache = 0, set(), None

    def add(self, local, masks, weight, client_id=None):
        require(isinstance(weight, int) and weight > 0 and self.seen+weight <= self.total_weight,
                'invalid client weight')
        require(set(local) == set(self.base) and set(masks) <= set(self.base), 'state keys differ')
        if client_id is not None:
            require(client_id not in self.clients, 'duplicate client update'); self.clients.add(client_id)
        for n, before in self.base.items():
            after = local[n].detach().cpu()
            require(after.shape == before.shape and after.dtype == before.dtype, 'state shape/dtype differs')
            require(bool(torch.isfinite(after).all()), 'nonfinite local state')
            if n not in masks:
                require(torch.equal(before, after), 'inactive tensor changed: ' + n)
                require(not n.startswith('classifier.'), 'classifier missing participant contribution')
                continue
            mask = masks[n].detach().cpu()
            require(mask.dtype == torch.bool and mask.shape == before.shape, 'invalid mask')
            if '.lora_' in n:
                require(n.endswith('.lora_' + self.factor), 'both LoRA factors active')
            else:
                require(bool(mask.all()), 'classifier only partially covered')
            delta = after.double() - before.double()
            require(not torch.count_nonzero(delta[~mask]), 'inactive slice changed: ' + n)
            self.sums[n] += weight * delta * mask
            self.coverage[n] += weight * mask
        self.seen += weight; self._cache = None

    def increments(self):
        require(self.seen == self.total_weight, 'partial round')
        if self._cache is not None:
            return self._cache
        f = {n: self.sums[n]/self.total_weight for n in self.base}
        c = {n: self.sums[n]/self.coverage[n].clamp(min=1) for n in self.base}
        fn2 = effective_norm_sq(f, self.base, self.scalings, self.factor)
        cn2 = effective_norm_sq(c, self.base, self.scalings, self.factor)
        fp2 = sum(float(f[n].square().sum()) for n in f if '.lora_' in n)
        cp2 = sum(float(c[n].square().sum()) for n in c if '.lora_' in n)
        pscale = math.sqrt(cp2/fp2) if fp2 else 1.0
        # A nonzero C cannot be matched by scaling a functionally zero F.
        # Preserve a defined algorithm without pretending that a match occurred.
        matched = fn2 > 0 or cn2 == 0
        gscale = math.sqrt(cn2/fn2) if fn2 else 1.0
        reason = None if matched else 'zero_F_nonzero_C_function_update; G_falls_back_to_F'
        p = {n: v*pscale if '.lora_' in n else v for n, v in f.items()}
        g = {n: v*gscale if '.lora_' in n else v for n, v in f.items()}
        for n in f:
            if n.startswith('classifier.'):
                require(torch.equal(f[n], c[n]), 'classifier differs across rules')
        result = {'fixed': f, 'contributors': c,
                  'parameter_norm_matched': p, 'function_norm_matched': g}
        require(all(bool(torch.isfinite(v).all()) for x in result.values() for v in x.values()),
                'nonfinite scaled update')
        meta = {'factor': self.factor, 'parameter_scale': pscale, 'function_scale': gscale,
                'fixed_function_norm': math.sqrt(fn2), 'contributors_function_norm': math.sqrt(cn2),
                'function_match_defined': matched, 'fallback': reason,
                'classifier_equal_across_rules': True}
        self._cache = result, meta
        return self._cache

    def state(self, rule):
        require(rule in RULES, 'unknown rule')
        updates, meta = self.increments()
        return {n: (v.double()+updates[rule][n]).to(v.dtype)
                for n, v in self.base.items()}, dict(meta)

    def readout(self, rule):
        state, meta = self.state(rule)
        inc, _ = self.increments()
        applied = {n: state[n].double()-self.base[n].double() for n in self.base}
        counts = {}
        for name in prefixes(self.base):
            key = name + '.lora_' + self.factor
            c = self.coverage[key]
            row = c[:, 0] if self.factor == 'a' else c[0, :]
            require(torch.equal(c, row[:, None].expand_as(c) if self.factor == 'a'
                                 else row[None, :].expand_as(c)), 'non-rank-structured mask')
            counts[name] = (row/self.total_weight).tolist()
        return {'rule': rule, **meta, 'coverage_fraction_by_module_rank': counts,
                'requested_effective_norm': math.sqrt(effective_norm_sq(inc[rule], self.base, self.scalings, self.factor)),
                'applied_effective_norm': math.sqrt(effective_norm_sq(applied, self.base, self.scalings, self.factor)),
                'applied_vs_requested': compare_updates(applied, inc[rule], self.base, self.scalings, self.factor)}
