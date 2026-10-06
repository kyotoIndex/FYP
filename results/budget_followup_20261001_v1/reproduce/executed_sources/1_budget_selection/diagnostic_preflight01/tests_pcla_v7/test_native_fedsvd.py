import pytest
import torch

from pcla_research_v7.native_fedsvd import (aggregate_b_head, refresh_due,
                                            refresh_state, svd_pair, trainable_names)


def state():
    torch.manual_seed(19)
    return {"query.lora_a": torch.randn(3, 7),
            "query.lora_b": torch.randn(5, 3),
            "classifier.weight": torch.randn(2, 5),
            "classifier.bias": torch.randn(2)}


def test_fedsvd_matches_full_svd_and_preserves_effective_model():
    current = state()
    b, a = current["query.lora_b"], current["query.lora_a"]
    new_b, new_a, audit = svd_pair(b, a)
    u, s, vh = torch.linalg.svd(b @ a, full_matrices=False)
    assert torch.allclose(new_b, u[:, :3] * s[:3], atol=1e-6)
    assert torch.allclose(new_a, vh[:3], atol=1e-6)
    assert torch.allclose(new_b @ new_a, b @ a, atol=1e-5)
    assert audit["relative_error"] < 1e-5
    probe = lambda parent: (parent["query.lora_b"] @ parent["query.lora_a"]) @ torch.ones(7, 4)
    refreshed, record = refresh_state(current, 2, 5, probe)
    assert record["refreshed"] is True
    assert torch.equal(refreshed["classifier.weight"], current["classifier.weight"])
    assert torch.allclose(probe(refreshed), probe(current), atol=1e-5)
    assert not refresh_due(1, 1) and refresh_due(2, 1)
    assert refresh_due(7, 5) and not refresh_due(6, 5)


def test_fedsvd_full_cpu_float64_svd_and_float32_deployment(monkeypatch):
    """The server must not depend on accelerator float32 SVD accuracy."""
    torch.manual_seed(311)
    rank, width = 16, 64
    a = torch.randn(rank, width, dtype=torch.float32)
    b = torch.randn(width, rank, dtype=torch.float32)
    b[:, -2:] *= 1e-4  # two weak singular directions, as in a trained adapter
    old_math = b.double() @ a.double()
    original_svd = torch.linalg.svd
    calls = []
    def checked_svd(value, *args, **kwargs):
        calls.append((value.shape, value.dtype, value.device.type,
                      kwargs.get("full_matrices")))
        assert value.shape == (width, width)
        assert value.dtype == torch.float64 and value.device.type == "cpu"
        return original_svd(value, *args, **kwargs)
    monkeypatch.setattr(torch.linalg, "svd", checked_svd)
    # A requested accelerator is deliberately not used for the reference
    # solve; the deployed factors still return to the input device/dtype.
    next_b, next_a, audit = svd_pair(b, a, compute_device=torch.device("cuda:0"))
    assert calls == [((width, width), torch.float64, "cpu", False)]
    assert next_b.dtype == next_a.dtype == torch.float32
    assert next_b.device.type == next_a.device.type == "cpu"
    assert next_b.shape == (width, rank) and next_a.shape == (rank, width)
    u, singular, vh = original_svd(old_math, full_matrices=False)
    assert torch.allclose(next_b, (u[:, :rank] * singular[:rank]).float(), atol=1e-6)
    assert torch.allclose(next_a, vh[:rank].float(), atol=1e-6)
    mathematical_error = torch.linalg.matrix_norm(
        next_b.double() @ next_a.double() - old_math).item() / torch.linalg.matrix_norm(old_math).item()
    deployed_error = torch.linalg.matrix_norm(
        ((next_b @ next_a) - (b @ a)).double()).item() / torch.linalg.matrix_norm((b @ a).double()).item()
    assert mathematical_error == pytest.approx(audit["relative_error"], abs=1e-12)
    assert deployed_error == pytest.approx(audit["deployed_relative_error"], abs=1e-12)
    assert max(mathematical_error, deployed_error) < 1e-5
    assert audit["compute_device"] == "cpu" and audit["svd_dtype"] == "float64"


def test_fedsvd_preserves_full_svd_null_basis_for_rank_deficient_product():
    torch.manual_seed(312)
    b = torch.randn(20, 8)
    a = torch.randn(8, 20)
    b[:, -2:] = 0  # rank < requested adapter rank
    next_b, next_a, audit = svd_pair(b, a)
    u, singular, vh = torch.linalg.svd(b.double() @ a.double(), full_matrices=False)
    assert torch.allclose(next_a, vh[:8].float(), atol=1e-6)
    assert torch.allclose(next_b, (u[:, :8] * singular[:8]).float(), atol=1e-6)
    assert audit["relative_error"] < 1e-5
    assert audit["deployed_relative_error"] < 1e-5


def test_fedsvd_only_b_and_head_aggregate():
    parent = state()
    assert trainable_names(parent) == {"query.lora_b", "classifier.weight", "classifier.bias"}
    first = {k: v.clone() for k, v in parent.items()}
    second = {k: v.clone() for k, v in parent.items()}
    first["query.lora_b"] += 2
    second["query.lora_b"] -= 1
    first["classifier.bias"] += 3
    result = aggregate_b_head(parent, [first, second], [1, 3])
    assert torch.allclose(result["query.lora_b"], parent["query.lora_b"]-.25)
    assert torch.equal(result["query.lora_a"], parent["query.lora_a"])
    assert torch.allclose(result["classifier.bias"], parent["classifier.bias"]+.75)
    first["query.lora_a"] += 1
    with pytest.raises(ValueError, match="frozen factor"):
        aggregate_b_head(parent, [first, second], [1, 3])
