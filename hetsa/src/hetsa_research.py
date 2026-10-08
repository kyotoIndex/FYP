"""Exploratory HetSA variants. Independent of the frozen historical runner."""
from __future__ import annotations

import torch


@torch.no_grad()
def transport_private_b(old_a, new_a, old_b, strength=1.0, ridge_relative=1e-3):
    """Ridge least-squares in weight space, not a claim of input-aware optimality.

    Minimize ||B A_new - B_old A_old||_F^2 + lambda ||B-B_old||_F^2.
    Strength interpolates with original B; no full d_out x d_in matrix formed.
    """
    if not 0 <= strength <= 1 or ridge_relative <= 0:
        raise ValueError('strength in [0,1], positive ridge required')
    if old_a.shape != new_a.shape or old_b.shape[1] != old_a.shape[0]:
        raise ValueError('incompatible factor dimensions')
    if strength == 0 or torch.equal(old_a, new_a):
        return old_b.clone()
    a0, a1, b = old_a.double(), new_a.double(), old_b.double()
    gram = a1 @ a1.T
    ridge = ridge_relative * gram.trace() / a1.shape[0] + 1e-8
    lhs = gram + ridge * torch.eye(a1.shape[0], device=a1.device, dtype=a1.dtype)
    rhs = b @ (a0 @ a1.T) + ridge * b
    fitted = torch.linalg.solve(lhs, rhs.T).T
    result = ((1-strength)*b + strength*fitted).to(old_b.dtype)
    if not torch.isfinite(result).all():
        raise FloatingPointError('nonfinite transported B')
    return result


def shared_a_update_round(method, round_id, warmup=3, period=3):
    if method in {'local_only', 'fixed_a_local_b'}:
        return False
    return method != 'slow_a' or round_id <= warmup or (round_id-warmup) % period == 0


def suffix_layers(depth, capacity):
    if not 1 <= capacity <= depth:
        raise ValueError('capacity outside depth')
    return list(range(depth-capacity, depth))
