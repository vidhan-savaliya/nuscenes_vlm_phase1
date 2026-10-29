"""
model_loader.py — VLM loading and inference interface for Phase 1.

Supported models
----------------
* ``qwen2_5_vl_3b`` → ``Qwen/Qwen2.5-VL-3B-Instruct``
* ``smolvlm2_2b``   → ``HuggingFaceTB/SmolVLM2-2.2B-Instruct``

Design decisions
----------------
* Only ONE model is loaded at a time (enforced at class level).
* Model weights are loaded lazily on the first call to ``load()``.
* GPU is preferred; falls back to CPU when CUDA is unavailable.
* ``dtype`` is configurable (bfloat16, float16, float32).
* Flash-attention is used opportunistically when available.
* Generation is deterministic by default (greedy decoding).
* OOM errors are caught and reported without crashing the pipeline.

The class follows a simple lifecycle::

    loader = ModelLoader(cfg)
    loader.load()
    result = loader.generate(image_path, prompt)
    loader.unload()   # frees GPU memory

"""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from causal_pruner import CausalTokenPruner
from causal_vlm import CausalVLMWrapper

logger = logging.getLogger(__name__)

# Mapping of friendly model keys to HuggingFace model IDs
_MODEL_IDS: Dict[str, str] = {
    "qwen2_5_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "smolvlm2_2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
}

_DTYPE_MAP = {
    "bfloat16": None,  # resolved lazily after torch import
    "float16": None,
    "float32": None,
}


def _resolve_dtype(dtype_str: str):
    """Resolve dtype string to torch.dtype."""
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        logger.warning(
            "Unknown dtype '%s', defaulting to float32.", dtype_str
        )
        return torch.float32
    return mapping[dtype_str]


def _detect_device(preferred: Optional[str] = None) -> str:
    """Return the best available compute device.

    Args:
        preferred: Override device string (e.g. ``"cuda:0"``).  ``None``
            triggers auto-detection.

    Returns:
        Device string: ``"cuda"`` or ``"cpu"``.
    """
    if preferred is not None:
        return preferred
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _try_flash_attention() -> bool:
    """Return True if flash_attn is importable (for Qwen models)."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


class ModelLoader:
    """Manages loading and greedy inference for a single VLM.

    Args:
        cfg: Top-level configuration dict (parsed from ``config.yaml``).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        model_cfg = cfg.get("model", {})
        self.model_key: str = model_cfg.get("name", "qwen2_5_vl_3b")
        self.model_ids: Dict[str, str] = model_cfg.get(
            "model_ids", _MODEL_IDS
        )
        self.model_id: str = self.model_ids.get(self.model_key, "")
        if not self.model_id:
            raise ValueError(
                f"Unknown model key '{self.model_key}'.  "
                f"Supported: {list(self.model_ids)}"
            )

        self.dtype_str: str = model_cfg.get("dtype", "bfloat16")
        self.max_new_tokens: int = model_cfg.get("max_new_tokens", 512)
        self.do_sample: bool = model_cfg.get("do_sample", False)
        self.temperature: float = float(model_cfg.get("temperature", 1.0))
        self.top_p: float = float(model_cfg.get("top_p", 1.0))
        self.device: str = _detect_device(model_cfg.get("device"))
        self.cache_dir: Optional[str] = model_cfg.get("cache_dir")

        # Causal Token Pruning configuration (E2C-VLM Stage 1)
        causal_cfg = cfg.get("causal_pruning", {})
        self.causal_enabled: bool = bool(causal_cfg.get("enabled", False))
        self.causal_method: str = str(causal_cfg.get("method", "causal")).lower()
        self.pruning_ratio: float = float(causal_cfg.get("pruning_ratio", 0.5))
        self.threshold_epsilon: Optional[float] = causal_cfg.get("threshold_epsilon")
        if self.threshold_epsilon is not None:
            self.threshold_epsilon = float(self.threshold_epsilon)
        self.intervention_mode: str = str(causal_cfg.get("intervention_mode", "group"))
        self.group_size: int = int(causal_cfg.get("group_size", 16))
        self.null_token_mode: str = str(causal_cfg.get("null_token_mode", "mean"))
        self.keep_minimum: int = int(causal_cfg.get("keep_minimum", 16))
        self.mask_in_place: bool = bool(causal_cfg.get("mask_in_place", True))

        # Internal state
        self._model = None
        self._processor = None
        self._pruner: Optional[CausalTokenPruner] = None
        self._causal_wrapper: Optional[CausalVLMWrapper] = None
        self._load_time_sec: float = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Download (if needed) and load the model and processor into memory.

        This method is idempotent — calling it twice has no effect.

        Raises:
            ImportError: If required packages (transformers, torch) are missing.
            RuntimeError: If model loading fails.
        """
        if self._model is not None:
            logger.debug("Model '%s' already loaded.", self.model_key)
            return

        logger.info(
            "Loading model '%s' (%s) onto %s with dtype=%s …",
            self.model_key,
            self.model_id,
            self.device,
            self.dtype_str,
        )
        t0 = time.perf_counter()

        try:
            import torch
            dtype = _resolve_dtype(self.dtype_str)

            if self.model_key == "qwen2_5_vl_3b":
                self._load_qwen(dtype, torch)
            elif self.model_key == "smolvlm2_2b":
                self._load_smolvlm(dtype, torch)
            else:
                raise ValueError(f"Unsupported model key: {self.model_key}")

        except Exception as exc:
            logger.error("Failed to load model '%s': %s", self.model_key, exc)
            raise

        self._load_time_sec = time.perf_counter() - t0
        logger.info(
            "Model loaded in %.1f s.  Device: %s.", self._load_time_sec, self.device
        )

    def _load_qwen(self, dtype, torch) -> None:
        """Load Qwen2.5-VL model and processor."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        use_flash = _try_flash_attention() and self.device.startswith("cuda")
        attn_impl = "flash_attention_2" if use_flash else "eager"
        if use_flash:
            logger.info("Flash Attention 2 enabled.")

        load_kwargs = dict(
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )
        # Use device_map="auto" only on CUDA to allow multi-GPU or CPU offload
        if self.device.startswith("cuda"):
            load_kwargs["device_map"] = "auto"

        if self.cache_dir:
            load_kwargs["cache_dir"] = self.cache_dir

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, **load_kwargs
        )
        if not self.device.startswith("cuda"):
            self._model = self._model.to(self.device)
        self._model.eval()

        self._setup_causal_pruning()

        processor_kwargs = {}
        if self.cache_dir:
            processor_kwargs["cache_dir"] = self.cache_dir
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, **processor_kwargs
        )

    def _load_smolvlm(self, dtype, torch) -> None:
        """Load SmolVLM2 model and processor."""
        from transformers import AutoProcessor, AutoModelForVision2Seq

        load_kwargs = dict(torch_dtype=dtype)
        if self.device.startswith("cuda"):
            load_kwargs["device_map"] = "auto"
        if self.cache_dir:
            load_kwargs["cache_dir"] = self.cache_dir

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            **({"cache_dir": self.cache_dir} if self.cache_dir else {}),
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_id, **load_kwargs
        )
        if not self.device.startswith("cuda"):
            self._model = self._model.to(self.device)
        self._model.eval()

        self._setup_causal_pruning()

    def _setup_causal_pruning(self) -> None:
        """Initialize CausalTokenPruner and wrap model if causal pruning is enabled."""
        if self.causal_enabled and self.causal_method != "none" and self.pruning_ratio > 0.0:
            logger.info(
                "Initializing E2C-VLM CausalTokenPruner (method=%s, ratio=%.2f, mode=%s, group_size=%d) …",
                self.causal_method,
                self.pruning_ratio,
                self.intervention_mode,
                self.group_size,
            )
            self._pruner = CausalTokenPruner(
                method=self.causal_method,
                pruning_ratio=self.pruning_ratio,
                threshold_epsilon=self.threshold_epsilon,
                intervention_mode=self.intervention_mode,
                group_size=self.group_size,
                null_token_mode=self.null_token_mode,
                keep_minimum=self.keep_minimum,
                mask_in_place=self.mask_in_place,
            )
            self._causal_wrapper = CausalVLMWrapper(
                self._model,
                pruner=self._pruner,
                model_key=self.model_key,
            )
            self._model = self._causal_wrapper

    def unload(self) -> None:
        """Delete model from memory and free GPU cache."""
        if self._causal_wrapper is not None:
            self._causal_wrapper.remove_hook()
            self._causal_wrapper = None
        self._pruner = None
        self._model = None
        self._processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()
        logger.info("Model '%s' unloaded.", self.model_key)

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
    ) -> Dict[str, Any]:
        """Run VLM inference on a single image.

        Args:
            image_path: Absolute path to the image file.
            prompt: Text prompt to send alongside the image.

        Returns:
            Dictionary with keys:
              - ``text``: Generated description string.
              - ``n_tokens``: Number of generated tokens.
              - ``latency_sec``: Wall-clock inference time.
              - ``device``: Device used.
              - ``dtype``: Precision string.
              - ``error``: Error message if inference failed, else None.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError(
                "Model not loaded.  Call ModelLoader.load() first."
            )

        image_path = Path(image_path)
        if not image_path.is_file():
            return self._error_record(f"Image file not found: {image_path}")

        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return self._error_record(f"Failed to open image: {exc}")

        try:
            if self.model_key == "qwen2_5_vl_3b":
                return self._generate_qwen(image, prompt)
            elif self.model_key == "smolvlm2_2b":
                return self._generate_smolvlm(image, prompt)
            else:
                return self._error_record(f"Unknown model key: {self.model_key}")
        except MemoryError as exc:
            return self._error_record(f"OOM error: {exc}")
        except Exception as exc:
            return self._error_record(f"Inference error: {exc}")

    def _generate_qwen(self, image, prompt: str) -> Dict[str, Any]:
        """Run Qwen2.5-VL inference."""
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to the model's device
        target_device = next(self._model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
        )
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - t0

        # Trim input tokens from generated output
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        n_tokens = trimmed[0].shape[0]

        pruning_res = getattr(self._causal_wrapper, "last_pruning_result", None) if self._causal_wrapper else None
        causal_meta = {
            "enabled": self.causal_enabled,
            "method": self.causal_method,
            "pruning_ratio": self.pruning_ratio,
            "original_visual_tokens": pruning_res.original_count if pruning_res else None,
            "retained_visual_tokens": pruning_res.retained_count if pruning_res else None,
            "pruned_visual_tokens": pruning_res.pruned_count if pruning_res else None,
            "compression_ratio": pruning_res.compression_ratio if pruning_res else 0.0,
            "pruning_latency_sec": pruning_res.pruning_latency_sec if pruning_res else 0.0,
            "mean_causal_score": pruning_res.diagnostics.get("mean_causal_score") if pruning_res else None,
        }

        return {
            "text": output_text,
            "n_tokens": int(n_tokens),
            "latency_sec": latency,
            "device": str(target_device),
            "dtype": self.dtype_str,
            "error": None,
            "causal_pruning": causal_meta,
        }

    def _generate_smolvlm(self, image, prompt: str) -> Dict[str, Any]:
        """Run SmolVLM2 inference."""
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        input_text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._processor(
            text=input_text,
            images=[image],
            return_tensors="pt",
        )

        target_device = next(self._model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
        )
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - t0

        trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        output_text = self._processor.decode(trimmed[0], skip_special_tokens=True)
        n_tokens = trimmed.shape[1]

        pruning_res = getattr(self._causal_wrapper, "last_pruning_result", None) if self._causal_wrapper else None
        causal_meta = {
            "enabled": self.causal_enabled,
            "method": self.causal_method,
            "pruning_ratio": self.pruning_ratio,
            "original_visual_tokens": pruning_res.original_count if pruning_res else None,
            "retained_visual_tokens": pruning_res.retained_count if pruning_res else None,
            "pruned_visual_tokens": pruning_res.pruned_count if pruning_res else None,
            "compression_ratio": pruning_res.compression_ratio if pruning_res else 0.0,
            "pruning_latency_sec": pruning_res.pruning_latency_sec if pruning_res else 0.0,
            "mean_causal_score": pruning_res.diagnostics.get("mean_causal_score") if pruning_res else None,
        }

        return {
            "text": output_text,
            "n_tokens": int(n_tokens),
            "latency_sec": latency,
            "device": str(target_device),
            "dtype": self.dtype_str,
            "error": None,
            "causal_pruning": causal_meta,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Properties & helpers
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def load_time_sec(self) -> float:
        """Model loading wall-clock time in seconds."""
        return self._load_time_sec

    @property
    def is_loaded(self) -> bool:
        """True if the model and processor are currently in memory."""
        return self._model is not None

    def peak_gpu_memory_mb(self) -> Optional[float]:
        """Return peak GPU memory allocated in MB, or None on CPU."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024 ** 2)
        except ImportError:
            pass
        return None

    def reset_peak_gpu_memory(self) -> None:
        """Reset PyTorch peak GPU memory counter."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    def _error_record(self, message: str) -> Dict[str, Any]:
        """Return a standardised error result dict."""
        logger.error(message)
        return {
            "text": "",
            "n_tokens": 0,
            "latency_sec": 0.0,
            "device": self.device,
            "dtype": self.dtype_str,
            "error": message,
            "causal_pruning": {
                "enabled": self.causal_enabled,
                "method": self.causal_method,
                "pruning_ratio": self.pruning_ratio,
                "original_visual_tokens": None,
                "retained_visual_tokens": None,
                "pruned_visual_tokens": None,
                "compression_ratio": 0.0,
                "pruning_latency_sec": 0.0,
                "mean_causal_score": None,
            },
        }
