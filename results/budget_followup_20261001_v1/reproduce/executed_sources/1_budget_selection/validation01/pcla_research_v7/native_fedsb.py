"""Fed-SB task-gradient initialization and fixed A/B, trainable R adapter.

Based on the paper's B R A model and the author's separated/r_squared path.
No official repository code is copied; its implementation has no LICENSE file
at the pinned commit. Runtime supplies permitted public-training gradients.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

OFFICIAL_REPO = "https://github.com/CERT-Lab/fed-sb"
OFFICIAL_COMMIT = "e9e649822ce63437535f72fbf06c146a22b9e527"
OFFICIAL_GRADIENT = "fed_sb/utils/gradient_utils.py:estimate_and_process_grads_torch"
OFFICIAL_INIT = "fed_sb/utils/initialization_utils.py:find_and_initialize_grad"


def signed_task_step(gradient: Tensor, effective_lr: float) -> Tensor:
    """Author preprocessing: -effective_lr*sign(accumulated task gradient)."""
    if effective_lr <= 0 or not torch.isfinite(gradient).all():
        raise ValueError("valid signed-gradient initialization required")
    return -float(effective_lr) * torch.sign(gradient)


def effective_init_lr(adapter_lr: float, local_optimizer_steps=16) -> float:
    """Author denominator is len(client train loader) * local_epochs.

    Official DP/SNLI/fed_trainer.py:224-230 computes this before calling the
    public-gradient estimator. Here 256 client rows / batch16 * one epoch=16.
    """
    if adapter_lr <= 0 or local_optimizer_steps <= 0:
        raise ValueError("positive optimizer setup required")
    return adapter_lr / local_optimizer_steps


def estimate_signed_steps(model: nn.Module, public_dataset, target_names: list[str],
                          adapter_lr: float, *, device: torch.device,
                          batch_size=16, local_optimizer_steps=16) -> dict[str, Tensor]:
    """Compute the author's signed task update on exactly the allowed public role.

    The caller must supply `TokenCache.dataset(..., 'public_training',
    'algorithm')`. The 512-row length check prevents a silently reduced or
    expanded initialization set. No benchmark evaluation loader is accepted.
    """
    if (getattr(public_dataset, "role", None) != "public_training" or
            getattr(public_dataset, "actor", None) != "algorithm" or
            len(public_dataset) != 512 or not target_names or batch_size <= 0):
        raise ValueError("Fed-SB initialization requires public_training512")
    params = dict(model.named_parameters())
    if len(target_names) != len(set(target_names)) or any(name not in params for name in target_names):
        raise ValueError("unknown target projection")
    if any(not name.endswith((".query.weight", ".value.weight", ".query.base.weight", ".value.base.weight"))
           for name in target_names):
        raise ValueError("Fed-SB initialization target must be BERT Q/V base weight")
    old_mode = model.training
    old_flags = {name: parameter.requires_grad for name, parameter in params.items()}
    old_grads = {name: parameter.grad for name, parameter in params.items()}
    devices = [device.index] if device.type == "cuda" else []
    try:
        for name, parameter in params.items():
            parameter.requires_grad_(name in target_names)
            parameter.grad = None
        model.train()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(42)
            if devices:
                torch.cuda.manual_seed_all(42)
            for batch in DataLoader(public_dataset, batch_size=batch_size, shuffle=False):
                batch = {key: value.to(device) for key, value in batch.items()}
                loss = model(**batch).loss
                # The author's gradient_utils.py divides the model's mean loss
                # by batch size. All 32 public batches are full-sized, so this
                # preserves its gradient weighting and exact sign convention.
                (loss / len(batch["labels"])).backward()
        eff_lr = effective_init_lr(adapter_lr, local_optimizer_steps)
        return {name: signed_task_step(params[name].grad.detach(), eff_lr)
                for name in target_names}
    finally:
        for name, parameter in params.items():
            parameter.requires_grad_(old_flags[name])
            parameter.grad = old_grads[name]
        model.train(old_mode)


def initialize_pair(task_step: Tensor, rank: int, *, svd_seed=42,
                    n_iter=10) -> tuple[Tensor, Tensor, Tensor, dict]:
    """Author orientation: SVD(step.T), A=U.T, B=V, R=diag(s).

    This task's deployed dtype is float32. The author implementation casts
    to bf16 after `svd_lowrank`; here the same decomposition is kept float32.
    """
    if task_step.ndim != 2 or not task_step.is_floating_point() or not torch.isfinite(task_step).all():
        raise ValueError("finite 2D task update required")
    out_features, in_features = task_step.shape
    if rank <= 0 or rank > min(in_features, out_features):
        raise ValueError("invalid Fed-SB rank")
    # The author's code resets seed 42 before each module's low-rank SVD.
    # fork_rng confines this to an independent initialization substream.
    devices = [task_step.device.index] if task_step.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(svd_seed)
        if devices:
            torch.cuda.manual_seed_all(svd_seed)
        u, singular, v = torch.svd_lowrank(task_step.T.float(), q=rank, niter=n_iter)
    a, b, r = u.T.contiguous(), v.contiguous(), torch.diag(singular).contiguous()
    reconstruction = b @ r @ a
    err = torch.linalg.matrix_norm((reconstruction-task_step.float()).double()).item()
    total = max(torch.linalg.matrix_norm(task_step.double()).item(), 1e-12)
    return a.to(task_step.dtype), b.to(task_step.dtype), r.to(task_step.dtype), {
        "rank": rank, "svd_seed": svd_seed, "svd_iterations": n_iter,
        "signed_step_relative_reconstruction_error": err / total,
        "signed_step_absolute_reconstruction_error": err}


class FedSBLinear(nn.Module):
    """Frozen base/A/B with one r×r trainable latent map."""
    def __init__(self, base: nn.Linear, a: Tensor, b: Tensor, r: Tensor,
                 *, scaling=1.0, dropout=0.1):
        super().__init__()
        if a.ndim != 2 or b.ndim != 2 or r.ndim != 2 or (
                a.shape[1] != base.in_features or b.shape[0] != base.out_features or
                a.shape[0] != b.shape[1] or r.shape != (a.shape[0], a.shape[0])):
            raise ValueError("Fed-SB factor shapes do not match base projection")
        self.base = base
        self.a = nn.Parameter(a.detach().clone(), requires_grad=False)
        self.b = nn.Parameter(b.detach().clone(), requires_grad=False)
        self.r = nn.Parameter(r.detach().clone())
        self.scaling = float(scaling)
        self.dropout = nn.Dropout(float(dropout))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(F.linear(self.dropout(x), self.a), self.r), self.b) * self.scaling


def mutable_names(state: Mapping[str, Tensor]) -> set[str]:
    if not any(name.endswith(".r") for name in state):
        raise ValueError("Fed-SB state has no latent maps")
    return {name for name in state if name.endswith(".r") or name.startswith("classifier.")}


def aggregate_r_head(parent: Mapping[str, Tensor], children: list[Mapping[str, Tensor]],
                     weights: list[int]) -> dict[str, Tensor]:
    if not children or len(children) != len(weights) or any(w <= 0 for w in weights):
        raise ValueError("invalid Fed-SB aggregation")
    mutable = mutable_names(parent)
    total = sum(weights)
    result = {name: tensor.clone() for name, tensor in parent.items()}
    for child in children:
        if set(child) != set(parent):
            raise ValueError("Fed-SB client keys differ")
        for name in parent:
            if name not in mutable and not torch.equal(child[name], parent[name]):
                raise ValueError(f"Fed-SB frozen basis changed: {name}")
    for name in mutable:
        delta = sum((child[name].double()-parent[name].double()) * (weight/total)
                    for child, weight in zip(children, weights))
        result[name] = (parent[name].double()+delta).to(parent[name].dtype)
    return result
