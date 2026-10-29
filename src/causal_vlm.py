"""
causal_vlm.py — Causal Vision-Language Model Wrapper for E2C-VLM Phase 1.

Paper:
E2C-VLM: Development of Edge Intelligence-based Energy-Efficient Causal VLM
for Robust Autonomous Systems
Ayanabha Ghosh, Vidhan Pravinbhai Savaliya, Debasis Das
Centre for AIoT and Applications & Department of CSE, IIT Jodhpur

This module wraps a pretrained Vision-Language Model (such as Qwen2.5-VL or
SmolVLM2) and integrates the CausalTokenPruner. It intercepts visual tokens
produced by the vision backbone (e.g. Qwen2_5_VisionTransformerPretrainedModel),
computes Pearl's interventional causal importance scores J(v_i), and performs
SCM graph surgery / pruning before language decoder reasoning.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from causal_pruner import CausalTokenPruner, PruningResult

logger = logging.getLogger(__name__)


class CausalVLMWrapper:
    """Wraps a pretrained Hugging Face VLM to apply causality-grounded token pruning.

    Supports Qwen2.5-VL (`Qwen2_5_VLForConditionalGeneration`) and SmolVLM2.
    Uses forward hooks on the visual encoder to intercept and prune visual tokens
    without modifying the model's core generation logic or breaking 3D RoPE.

    Args:
        model: HuggingFace vision-language model instance.
        pruner: CausalTokenPruner instance.
        model_key: Identifier string ('qwen2_5_vl_3b' or 'smolvlm2_2b').
    """

    def __init__(
        self,
        model: nn.Module,
        pruner: Optional[CausalTokenPruner] = None,
        model_key: str = "qwen2_5_vl_3b",
    ) -> None:
        self.model = model
        self.pruner = pruner or CausalTokenPruner(method="none", pruning_ratio=0.0)
        self.model_key = model_key

        self._hook_handle = None
        self._last_pruning_result: Optional[PruningResult] = None
        self._is_hooked = False

        if self.pruner.method != "none" and self.pruner.pruning_ratio > 0.0:
            self.attach_hook()

    @property
    def last_pruning_result(self) -> Optional[PruningResult]:
        """Return the PruningResult from the most recent forward/generate call."""
        return self._last_pruning_result

    def attach_hook(self) -> None:
        """Attach forward hook to the vision encoder to intercept visual tokens."""
        if self._is_hooked:
            return

        target_module = self._find_vision_module()
        if target_module is None:
            logger.warning(
                "Could not find vision encoder module on model '%s'. "
                "Causal pruning hook not attached.",
                self.model_key,
            )
            return

        def _visual_forward_hook(module: nn.Module, inputs: Any, output: Any) -> Any:
            """Hook called immediately after visual encoder completes."""
            return self._apply_pruning_hook(output)

        self._hook_handle = target_module.register_forward_hook(_visual_forward_hook)
        self._is_hooked = True
        logger.info(
            "Causal pruning hook successfully attached to '%s' (%s, ratio=%.2f).",
            target_module.__class__.__name__,
            self.pruner.method,
            self.pruner.pruning_ratio,
        )

    def remove_hook(self) -> None:
        """Remove the forward hook if attached."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            self._is_hooked = False
            logger.info("Causal pruning hook removed.")

    def _find_vision_module(self) -> Optional[nn.Module]:
        """Locate the vision backbone module inside the VLM architecture."""
        # Qwen2-VL / Qwen2.5-VL: self.model.visual
        if hasattr(self.model, "visual"):
            return self.model.visual

        # Some architectures: self.model.model.visual
        if hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            return self.model.model.visual

        # SmolVLM2 / Idefics: self.model.vision_model or self.model.model.vision_model
        if hasattr(self.model, "vision_model"):
            return self.model.vision_model
        if hasattr(self.model, "model") and hasattr(self.model.model, "vision_model"):
            return self.model.model.vision_model

        # Fallback search by module name
        for name, module in self.model.named_modules():
            if name.endswith("visual") or name.endswith("vision_tower") or name.endswith("vision_model"):
                return module

        return None

    def _apply_pruning_hook(self, visual_output: Any) -> Any:
        """Intercept visual output tensor, execute pruning, and return pruned tensor."""
        if self.pruner.method == "none" or self.pruner.pruning_ratio == 0.0:
            return visual_output

        # Handle various output formats: tensor directly or tuple / BaseModelOutput
        is_tuple = isinstance(visual_output, (tuple, list))
        tokens_tensor = visual_output[0] if is_tuple else visual_output

        if not isinstance(tokens_tensor, torch.Tensor):
            return visual_output

        # Run causal token pruning
        pruning_result = self.pruner.prune(tokens_tensor)
        self._last_pruning_result = pruning_result

        logger.debug(
            "Causal pruning executed: %d -> %d tokens (pruned %d, ratio %.1f%%) in %.3f s",
            pruning_result.original_count,
            pruning_result.retained_count,
            pruning_result.pruned_count,
            pruning_result.compression_ratio * 100.0,
            pruning_result.pruning_latency_sec,
        )

        if is_tuple:
            return (pruning_result.pruned_tokens,) + visual_output[1:]
        return pruning_result.pruned_tokens

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Call underlying model.generate() with causal pruning active."""
        return self.model.generate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying model."""
        return getattr(self.model, name)
