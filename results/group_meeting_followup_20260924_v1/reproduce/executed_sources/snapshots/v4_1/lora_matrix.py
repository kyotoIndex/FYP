"""Exact effective-weight operations. These do not average whole-network logits."""
from __future__ import annotations
import math
import torch
from common import require


def validate_pair(b, a):
    b, a = b.detach().cpu().double(), a.detach().cpu().double()
    require(b.ndim == a.ndim == 2 and b.shape[1] == a.shape[0] and b.shape[1] > 0,
            'invalid low-rank factor shapes')
    require(bool(torch.isfinite(b).all() and torch.isfinite(a).all()), 'nonfinite factor')
    return b, a


def checked_entries(entries):
    values = []
    for weight, scaling, b, a in entries:
        require(math.isfinite(weight) and weight > 0, 'sample weights must be finite positive')
        require(math.isfinite(scaling) and scaling > 0, 'scaling must be finite positive')
        b, a = validate_pair(b, a)
        if values:
            require((b.shape[0], a.shape[1]) ==
                    (values[0][2].shape[0], values[0][3].shape[1]), 'effective dimensions differ')
        values.append((float(weight), float(scaling), b, a))
    require(values, 'empty aggregation')
    return values, math.fsum(v[0] for v in values)


def weighted_effective_mean(entries):
    values, total = checked_entries(entries)
    result = torch.zeros(values[0][2].shape[0], values[0][3].shape[1], dtype=torch.float64)
    for weight, scale, b, a in values:
        result += (weight/total) * scale * (b@a)
    return result


def stacked_exact(entries):
    values, total = checked_entries(entries)
    return torch.cat([w/total*s*b for w, s, b, a in values], 1), torch.cat([a for w, s, b, a in values], 0)


def factor_average_residual(entries):
    values, total = checked_entries(entries)
    _, scale, first_b, first_a = values[0]
    bbar, abar = torch.zeros_like(first_b), torch.zeros_like(first_a)
    for w, s, b, a in values:
        require(b.shape == first_b.shape and a.shape == first_a.shape and s == scale,
                'factor averaging requires equal rank and scaling')
        bbar += w/total*b; abar += w/total*a
    target = weighted_effective_mean(values)
    represented = scale*(bbar@abar)
    return target, represented, target-represented


def product_delta(destination_initial, source_final, source_initial):
    bs, aa, shape = [], [], None
    for sign, item in ((1, destination_initial), (1, source_final), (-1, source_initial)):
        scale, b, a = item
        require(math.isfinite(scale) and scale > 0, 'invalid LoRA scale')
        b, a = validate_pair(b, a)
        shape = shape or (b.shape[0], a.shape[1])
        require(shape == (b.shape[0], a.shape[1]), 'effective dimensions differ')
        bs.append(sign*scale*b); aa.append(a)
    return torch.cat(bs, 1), torch.cat(aa, 0)


def exact_server_correction(base, mean_b, mean_a, scale, target):
    """Residual uses the actual rounded mean factors that will be deployed."""
    represented = scale * (mean_b.double() @ mean_a.double())
    residual = target.double() - represented
    pre_cast = base.double() + residual
    corrected = pre_cast.to(base.dtype)
    achieved = corrected.double() + represented
    desired = base.double() + target.double()
    error = achieved - desired
    # Only the final cast of the base can cause representation error here.
    rounding = corrected.double() - pre_cast
    arithmetic = error-rounding
    bound = 64*torch.finfo(torch.float64).eps * max(1., float(desired.abs().max()),
                                                   float(represented.abs().max()))
    require(float(arithmetic.abs().max()) <= bound, 'residual identity failed beyond rounding')
    tnorm = float(target.norm())
    return corrected, {'target_norm': tnorm, 'residual_norm': float(residual.norm()),
                       'residual_ratio': float(residual.norm())/tnorm if tnorm else None,
                       'deployed_error_max_abs': float(error.abs().max()),
                       'base_cast_error_max_abs': float(rounding.abs().max()),
                       'arithmetic_error_bound': bound,
                       'corrected_base_bytes': corrected.numel()*corrected.element_size()}
