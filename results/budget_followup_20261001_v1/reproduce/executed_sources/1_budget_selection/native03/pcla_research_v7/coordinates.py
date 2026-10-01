"""V7 LoRA coordinate maps. All decompositions run on CPU in float64.

Callers own the model and W0. These functions return new float32 factors and
never write into a module, base weight, or optimizer state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class CoordinateResult:
    b: Tensor
    a: Tensor
    numerical_rank: int
    rank_threshold: float
    relative_error: float
    absolute_error: float
    rotation: Tensor | None = None


def refresh_due(round_id: int, period: int) -> bool:
    if round_id < 1 or period < 1:
        raise ValueError("round and period must be positive")
    return round_id >= 2 and (round_id - 2) % period == 0


def _pair(b: Tensor, a: Tensor) -> tuple[np.ndarray, np.ndarray]:
    if b.ndim != 2 or a.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError("B and A must be compatible matrices")
    if b.shape[1] > min(b.shape[0], a.shape[1]):
        raise ValueError("rank exceeds a matrix dimension")
    bb = b.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
    aa = a.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
    if not np.isfinite(bb).all() or not np.isfinite(aa).all():
        raise ValueError("nonfinite factors")
    return bb, aa


def _result(old_b: np.ndarray, old_a: np.ndarray, b: np.ndarray, a: np.ndarray,
            numerical_rank: int, threshold: float, rotation: np.ndarray | None = None) -> CoordinateResult:
    # Report error after the float32 deployment cast, not just float64 algebra.
    bt = torch.from_numpy(np.asarray(b, dtype=np.float32).copy())
    at = torch.from_numpy(np.asarray(a, dtype=np.float32).copy())
    if not torch.isfinite(bt).all() or not torch.isfinite(at).all():
        raise FloatingPointError("coordinate factors are nonfinite after float32 cast")
    actual = bt.double().numpy() @ at.double().numpy()
    target = old_b @ old_a
    absolute = float(np.linalg.norm(actual - target))
    relative = absolute / max(float(np.linalg.norm(target)), 1e-12)
    if not math.isfinite(relative) or relative > 1e-5:
        raise FloatingPointError(f"coordinate map exceeds effective-matrix tolerance: {relative}")
    ot = None if rotation is None else torch.from_numpy(np.asarray(rotation, dtype=np.float64).copy())
    return CoordinateResult(bt, at, numerical_rank, threshold, relative, absolute, ot)


def _small_rank(b: np.ndarray, a: np.ndarray) -> int:
    _, rb = np.linalg.qr(b, mode="reduced")
    _, ra = np.linalg.qr(a.T, mode="reduced")
    return int(np.linalg.matrix_rank(rb @ ra.T))


def balance_d(b: Tensor, a: Tensor) -> CoordinateResult:
    """Reciprocal per-component balance, with zero factors left trainable."""
    bb, aa = _pair(b, a)
    bn, an = np.linalg.norm(bb, axis=0), np.linalg.norm(aa, axis=1)
    c = np.ones_like(bn)
    nonzero = (bn > 0) & (an > 0)
    c[nonzero] = np.exp(.5 * (np.log(an[nonzero]) - np.log(bn[nonzero])))
    return _result(bb, aa, bb * c, aa / c[:, None], _small_rank(bb, aa), 0.)


def split_s(b: Tensor, a: Tensor) -> CoordinateResult:
    """Thin QR/SVD split; zero singular B columns retain orthogonal A rows."""
    bb, aa = _pair(b, a)
    qb, rb = np.linalg.qr(bb, mode="reduced")
    qa, ra = np.linalg.qr(aa.T, mode="reduced")
    uc, singular, vch = np.linalg.svd(rb @ ra.T, full_matrices=True)
    u, v = qb @ uc, qa @ vch.T
    threshold = float(np.finfo(np.float64).eps * max(bb.shape[0], aa.shape[1]) * (singular[0] if len(singular) else 0.))
    k = int((singular > threshold).sum())
    for j in range(k):
        pivot = int(np.abs(u[:, j]).argmax())
        if u[pivot, j] < 0:
            u[:, j] *= -1
            v[:, j] *= -1
    root = np.sqrt(singular[:k])
    new_b = np.zeros_like(bb)
    new_a = np.zeros_like(aa)
    new_b[:, :k] = u[:, :k] * root
    new_a[:k] = root[:, None] * v[:, :k].T
    if k < bb.shape[1]:
        # Match the historical svd_split trainable null-row construction.
        residual = aa - (aa @ v[:, :k]) @ v[:, :k].T
        _, null_s, null_vh = np.linalg.svd(residual, full_matrices=False)
        rows = null_s[:bb.shape[1] - k, None] * null_vh[:bb.shape[1] - k]
        for row in rows:
            pivot = int(np.abs(row).argmax())
            if row[pivot] < 0:
                row *= -1
        new_a[k:] = rows
    return _result(bb, aa, new_b, new_a, k, threshold)


def rotation_from_sketch(cbar: Tensor) -> Tensor:
    """O diagonalizes the same-cohort Cbar.T Cbar, descending, fixed signs."""
    c = cbar.detach().to(device="cpu", dtype=torch.float64).numpy()
    if c.ndim != 2 or not np.isfinite(c).all():
        raise ValueError("sketch must be a finite h by r matrix")
    if not np.any(c):
        return torch.eye(c.shape[1], dtype=torch.float64)
    _, o = np.linalg.eigh(c.T @ c)
    o = o[:, ::-1].copy()
    for j in range(o.shape[1]):
        k = int(np.abs(o[:, j]).argmax())
        if o[k, j] < 0:
            o[:, j] *= -1
    return torch.from_numpy(o)


def rotate_t(b: Tensor, a: Tensor, cbar: Tensor) -> CoordinateResult:
    """Offline convenience: D followed by rotation. Probe must use that D parent."""
    d = balance_d(b, a)
    return rotate_balanced_t(d.b, d.a, cbar)


def rotate_balanced_t(b: Tensor, a: Tensor, cbar: Tensor) -> CoordinateResult:
    """Live T phase 2: rotate the exact D parent already sent to clients."""
    return rotate_with_matrix(b, a, rotation_from_sketch(cbar))


def rotate_with_matrix(b: Tensor, a: Tensor, rotation: Tensor) -> CoordinateResult:
    """Server/client CPU float64 reconstruction from the same transmitted O."""
    bb, aa = _pair(b, a)
    o = rotation.detach().to(device="cpu", dtype=torch.float64).numpy()
    if o.shape != (bb.shape[1], bb.shape[1]):
        raise ValueError("sketch rank mismatch")
    if not np.isfinite(o).all() or not np.allclose(o.T @ o, np.eye(len(o)), atol=1e-10):
        raise ValueError("invalid orthogonal rotation")
    return _result(bb, aa, bb @ o, o.T @ aa, _small_rank(bb, aa), 0., o)


def rotate_gradient(gradient: Tensor, rotation: Tensor, factor: str) -> Tensor:
    """Chain rule for full retained G; virtual AdamW must be recomputed after this."""
    g = gradient.detach().to(device="cpu", dtype=torch.float64)
    o = rotation.detach().to(device="cpu", dtype=torch.float64)
    if o.ndim != 2 or o.shape[0] != o.shape[1] or factor not in ("a", "b"):
        raise ValueError("invalid rotation or factor")
    if factor == "b" and g.ndim == 2 and g.shape[1] == len(o):
        return g @ o
    if factor == "a" and g.ndim == 2 and g.shape[0] == len(o):
        return o.T @ g
    raise ValueError("gradient shape mismatch")


def map_pair(kind: str, b: Tensor, a: Tensor, *, cbar: Tensor | None = None) -> CoordinateResult:
    if kind == "N":
        bb, aa = _pair(b, a)
        return _result(bb, aa, bb, aa, _small_rank(bb, aa), 0.)
    if kind == "D":
        return balance_d(b, a)
    if kind == "S":
        return split_s(b, a)
    if kind == "T":
        if cbar is None:
            raise ValueError("T requires current-cohort gradient sketch")
        return rotate_t(b, a, cbar)
    raise ValueError(f"unsupported representation: {kind}")
