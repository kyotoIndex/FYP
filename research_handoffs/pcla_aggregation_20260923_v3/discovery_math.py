from __future__ import annotations

import torch

from common import require


def validate_factors(b, a):
    b = b.detach().cpu().double()
    a = a.detach().cpu().double()
    require(b.ndim == 2 and a.ndim == 2 and b.shape[1] == a.shape[0],
            "incompatible low-rank factors")
    require(bool(torch.isfinite(b).all() and torch.isfinite(a).all()),
            "nonfinite factors")
    return b, a


def exact_sum(weighted_factors):
    """Return concatenated factors for a weighted sum of effective matrices.

    Each B factor must already include the adapter's alpha/r scaling. Negative
    weights are represented in B, so no square-root convention is required.
    """
    require(len(weighted_factors) > 0, "empty factor collection")
    bs, aa, shape = [], [], None
    for weight, b, a in weighted_factors:
        b, a = validate_factors(b, a)
        current = (b.shape[0], a.shape[1])
        shape = current if shape is None else shape
        require(current == shape, "effective matrix shapes differ")
        require(torch.isfinite(torch.tensor(float(weight))).item(),
                "nonfinite merge weight")
        bs.append(float(weight) * b)
        aa.append(a)
    return torch.cat(bs, dim=1), torch.cat(aa, dim=0)


def task_increment(destination_initial, source_final, source_initial):
    """Destination initialization plus the source's true effective increment."""
    return exact_sum([
        (1.0, *destination_initial),
        (1.0, *source_final),
        (-1.0, *source_initial),
    ])


def projected_moment(a, activation_batches):
    """Compute E[(A x)(A x)^T] from unpadded activation rows."""
    a = a.detach().cpu().double()
    result = torch.zeros(a.shape[0], a.shape[0], dtype=torch.float64)
    count = 0
    for x in activation_batches:
        x = x.detach().cpu().double()
        require(x.ndim == 2 and x.shape[1] == a.shape[1],
                "activation dimensions differ")
        require(bool(torch.isfinite(x).all()), "nonfinite activations")
        z = x @ a.T
        result += z.T @ z
        count += len(z)
    require(count > 0, "no valid activation tokens")
    return result / count, count


def compress(b, a, rank, projected=None):
    """Parameter-SVD or projected-activation low-rank compression."""
    b, a = validate_factors(b, a)
    maximum = min(b.shape[0], b.shape[1], a.shape[1])
    require(isinstance(rank, int) and 0 <= rank <= maximum, "invalid rank")
    if rank == 0:
        return b[:, :0].clone(), a[:0].clone()
    c = a @ a.T if projected is None else projected.detach().cpu().double()
    require(c.shape == (a.shape[0], a.shape[0]) and
            bool(torch.isfinite(c).all()), "invalid projected moment")
    c = (c + c.T) / 2
    eigenvalues = torch.linalg.eigvalsh(c)
    scale = max(1.0, float(c.abs().max()))
    require(float(eigenvalues.min()) >= -1e-10 * scale,
            "projected moment is not positive semidefinite")
    if projected is not None:
        require(float(c.trace()) > 0,
                "zero activation energy must be reported, not silently replaced")
    q, r = torch.linalg.qr(b, mode="reduced")
    h = r @ c @ r.T
    _, vectors = torch.linalg.eigh((h + h.T) / 2)
    vectors = vectors[:, -rank:]
    return q @ vectors, vectors.T @ r @ a


def group_reference(b, a, moments, rank, iterations=20, step=0.5):
    """Bounded calibration heuristic, not a proven global minimax solver."""
    b, a = validate_factors(b, a)
    require(0 < rank <= min(b.shape[0], b.shape[1], a.shape[1]),
            "invalid reference rank")
    require(len(moments) > 0 and iterations > 0, "empty reference problem")
    q, r = torch.linalg.qr(b, mode="reduced")
    hs, energies, floors = [], [], []
    for moment in moments:
        moment = moment.detach().cpu().double()
        require(moment.shape == (a.shape[0], a.shape[0]) and
                bool(torch.isfinite(moment).all()), "invalid group moment")
        moment = (moment + moment.T) / 2
        require(float(torch.linalg.eigvalsh(moment).min()) >=
                -1e-10 * max(1.0, float(moment.abs().max())),
                "non-PSD group moment")
        h = r @ moment @ r.T
        h = (h + h.T) / 2
        energy = float(h.trace())
        require(energy > 0, "zero-energy group requires explicit handling")
        values = torch.linalg.eigvalsh(h)
        hs.append(h)
        energies.append(energy)
        floors.append(max(0.0, float(values[:-rank].sum())) / energy)
    weights = torch.ones(len(hs), dtype=torch.float64) / len(hs)
    best, best_vectors, history = float("inf"), None, []
    for iteration in range(iterations):
        mixed = sum(w * h / energy
                    for w, h, energy in zip(weights, hs, energies))
        _, vectors = torch.linalg.eigh((mixed + mixed.T) / 2)
        vectors = vectors[:, -rank:]
        excess = torch.tensor([
            max(0.0, float(h.trace() - (vectors.T @ h @ vectors).trace())) /
            energy - floor
            for h, energy, floor in zip(hs, energies, floors)
        ], dtype=torch.float64)
        objective = float(excess.max())
        history.append({
            "iteration": iteration,
            "weights": weights.tolist(),
            "group_excess": excess.tolist(),
            "objective": objective,
        })
        if objective < best:
            best, best_vectors = objective, vectors.clone()
        weights *= torch.exp(step * (excess - excess.max()))
        weights /= weights.sum()
    return q @ best_vectors, best_vectors.T @ r @ a, history
