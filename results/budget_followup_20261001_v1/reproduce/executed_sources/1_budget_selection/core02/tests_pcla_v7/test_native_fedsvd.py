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
