"""RoBERTa classifier with a fixed rank-R LoRA on query/value and per-client trainable rank.

Every method builds the same rank-R model from the same seed. A client with training
rank r < R only updates LoRA components 0..r-1; the rest stay frozen in the forward.
Because B starts at zero, an all-rank-2 federation is exactly a rank-2 LoRA with the
same per-component scale, so rank-2 and rank-8 runs share initialisation and scaling.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from transformers import AutoModelForSequenceClassification


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, scale: float):
        super().__init__()
        self.base = base
        self.scale = float(scale)
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, inputs: Tensor) -> Tensor:
        update = nn.functional.linear(nn.functional.linear(inputs, self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.scale


class TaskModel(nn.Module):
    def __init__(self, hf: nn.Module, rank: int, scale: float):
        super().__init__()
        if hf.config.model_type != "roberta":
            raise ValueError("Only RoBERTa sequence classification is supported")
        self.hf = hf
        self.rank = int(rank)
        hf.requires_grad_(False)
        self.adapters: list[LoRALinear] = []
        for layer in hf.roberta.encoder.layer:
            attention = layer.attention.self
            for name in ("query", "value"):
                adapter = LoRALinear(getattr(attention, name), rank, scale)
                setattr(attention, name, adapter)
                self.adapters.append(adapter)
        hf.classifier.requires_grad_(True)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    @classmethod
    def build(cls, model_cfg: dict, num_labels: int, rank: int, scale: float, seed: int) -> "TaskModel":
        # The classifier head and LoRA A are drawn from the torch RNG in a fixed
        # order, so every method with the same seed starts from identical weights.
        torch.manual_seed(seed)
        hf = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["name"],
            revision=model_cfg.get("revision", "main"),
            local_files_only=model_cfg.get("local_files_only", True),
            attn_implementation="eager",
            torch_dtype=torch.float32,
            num_labels=num_labels,
        )
        return cls(hf, rank, scale)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return self.hf(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

    def task_parameters(self) -> dict[str, nn.Parameter]:
        return {n: p for n, p in self.named_parameters() if p.requires_grad}

    def task_state(self) -> dict[str, Tensor]:
        return {n: p.detach().cpu().clone() for n, p in self.task_parameters().items()}

    @torch.no_grad()
    def load_task_state(self, state: dict[str, Tensor]) -> None:
        params = self.task_parameters()
        if params.keys() != state.keys():
            raise ValueError("Task state keys differ from model task parameters")
        for name, param in params.items():
            param.copy_(state[name])

    def mask_gradients(self, train_rank: int) -> None:
        """Zero gradients of LoRA components >= train_rank (frozen for this client)."""
        if train_rank >= self.rank:
            return
        for adapter in self.adapters:
            if adapter.lora_a.grad is not None:
                adapter.lora_a.grad[train_rank:].zero_()
            if adapter.lora_b.grad is not None:
                adapter.lora_b.grad[:, train_rank:].zero_()


def is_lora_a(name: str) -> bool:
    return name.endswith("lora_a")


def is_lora_b(name: str) -> bool:
    return name.endswith("lora_b")


def upload_parameter_count(state: dict[str, Tensor], train_rank: int) -> int:
    """Parameters a client actually trained: r rows of A, r columns of B, the head."""
    total = 0
    for name, value in state.items():
        if is_lora_a(name):
            total += train_rank * value.shape[1]
        elif is_lora_b(name):
            total += value.shape[0] * train_rank
        else:
            total += value.numel()
    return total
