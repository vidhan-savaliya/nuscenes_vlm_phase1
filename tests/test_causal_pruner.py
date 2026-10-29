"""
test_causal_pruner.py — Unit tests for E2C-VLM Causal Token Pruning engine.

Paper:
E2C-VLM: Development of Edge Intelligence-based Energy-Efficient Causal VLM
for Robust Autonomous Systems (IIT Jodhpur)
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from causal_pruner import CausalTokenPruner, PruningResult, compute_kl_divergence
from causal_vlm import CausalVLMWrapper


# ─────────────────────────────────────────────────────────────────────────────
# 1. KL Divergence Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_kl_divergence_identity():
    """KL divergence between identical distributions must be exactly zero."""
    logits = torch.randn(2, 5, 50)
    kl = compute_kl_divergence(logits, logits)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_kl_divergence_non_negative():
    """KL divergence must always be non-negative (Gibbs inequality)."""
    p = torch.randn(4, 10, 100)
    q = torch.randn(4, 10, 100)
    kl = compute_kl_divergence(p, q)
    assert torch.all(kl >= -1e-6)


def test_kl_divergence_perturbation_sensitivity():
    """Perturbing a distribution increases KL divergence."""
    p = torch.randn(2, 50)
    q1 = p + torch.randn_like(p) * 0.1
    q2 = p + torch.randn_like(p) * 2.0
    kl1 = compute_kl_divergence(p, q1)
    kl2 = compute_kl_divergence(p, q2)
    assert kl2.mean() > kl1.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Counterfactual Null Token Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_null_token_modes():
    """Verify spatial centroid, zero, and gaussian null token generation."""
    v = torch.ones(1, 10, 32) * 3.0

    pruner_mean = CausalTokenPruner(null_token_mode="mean")
    null_mean = pruner_mean.compute_null_token(v)
    assert torch.allclose(null_mean, torch.ones_like(null_mean) * 3.0)

    pruner_zero = CausalTokenPruner(null_token_mode="zero")
    null_zero = pruner_zero.compute_null_token(v)
    assert torch.allclose(null_zero, torch.zeros_like(null_zero))

    pruner_gauss = CausalTokenPruner(null_token_mode="gaussian")
    null_gauss = pruner_gauss.compute_null_token(v)
    assert null_gauss.shape == (1, 1, 32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pruning Ratio & Retention Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("prune_ratio", [0.0, 0.25, 0.5, 0.75])
def test_pruning_ratios(prune_ratio):
    """Verify that the exact expected fraction of tokens is pruned."""
    N = 64
    D = 16
    tokens = torch.randn(1, N, D)
    pruner = CausalTokenPruner(
        method="causal",
        pruning_ratio=prune_ratio,
        group_size=8,
        keep_minimum=4,
    )
    result = pruner.prune(tokens)

    assert result.original_count == N
    expected_keep = max(4, int(round(N * (1.0 - prune_ratio))))
    assert result.retained_count == expected_keep
    assert result.pruned_count == N - expected_keep
    assert abs(result.compression_ratio - (result.pruned_count / N)) < 1e-5


def test_safety_minimum_retention_guard():
    """Minimum retention guard prevents pruning all tokens."""
    N = 32
    tokens = torch.randn(1, N, 8)
    pruner = CausalTokenPruner(
        method="causal",
        pruning_ratio=0.99,  # Would otherwise leave ~0 tokens
        keep_minimum=10,
    )
    result = pruner.prune(tokens)
    assert result.retained_count >= 10


def test_epsilon_threshold_pruning():
    """Pruning by epsilon threshold J(v_i) < epsilon."""
    N = 40
    tokens = torch.randn(1, N, 16)
    # Set high epsilon to trigger pruning
    pruner = CausalTokenPruner(
        method="causal",
        threshold_epsilon=0.01,
        keep_minimum=8,
    )
    result = pruner.prune(tokens)
    assert result.retained_count >= 8
    assert result.original_count == N


# ─────────────────────────────────────────────────────────────────────────────
# 4. Comparative Methods Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["causal", "fastv_attention", "similarity", "random", "none"])
def test_all_pruning_methods_execute(method):
    """All supported pruning methods must execute without error and return PruningResult."""
    tokens = torch.randn(1, 32, 16)
    pruner = CausalTokenPruner(method=method, pruning_ratio=0.5, keep_minimum=4)
    result = pruner.prune(tokens)

    assert isinstance(result, PruningResult)
    assert result.method == method
    if method == "none":
        assert result.retained_count == 32
        assert result.compression_ratio == 0.0
    else:
        assert result.retained_count == 16
        assert result.pruned_count == 16


def test_unbatched_input_handling():
    """Pruner must handle unbatched [N, D] tensors seamlessly."""
    tokens = torch.randn(48, 16)
    pruner = CausalTokenPruner(method="causal", pruning_ratio=0.5, keep_minimum=4)
    result = pruner.prune(tokens)

    assert result.pruned_tokens.dim() == 2
    assert result.pruned_tokens.shape == (48, 16)
    assert result.retention_mask.dim() == 1
    assert result.retention_mask.shape == (48,)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CausalVLMWrapper Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class MockVisionBackbone(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.proj = nn.Linear(3, embed_dim)

    def forward(self, x):
        return self.proj(x)


class MockVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = MockVisionBackbone(32)
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        v = self.visual(x)
        return self.head(v.mean(dim=1))


def test_causal_vlm_wrapper_hook_lifecycle():
    """Verify forward hook attaches, intercepts, prunes, and detaches cleanly."""
    model = MockVLM()
    pruner = CausalTokenPruner(method="causal", pruning_ratio=0.5, keep_minimum=4)
    wrapper = CausalVLMWrapper(model, pruner=pruner, model_key="test_mock")

    assert wrapper._is_hooked is True

    # Run dummy forward pass
    x = torch.randn(1, 32, 3)
    _ = model(x)

    # Check that pruning occurred and was recorded
    last_res = wrapper.last_pruning_result
    assert last_res is not None
    assert last_res.original_count == 32
    assert last_res.retained_count == 16
    assert last_res.pruned_count == 16

    # Test clean detachment
    wrapper.remove_hook()
    assert wrapper._is_hooked is False
