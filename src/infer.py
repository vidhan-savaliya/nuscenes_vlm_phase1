"""
infer.py — Main inference pipeline for Phase 1.

Workflow
--------
nuScenes dataset
  → metadata inspection
  → scene selection (train / val)
  → annotated sample traversal
  → CAM_FRONT image retrieval
  → image validation
  → VLM inference
  → natural-language scene description
  → incremental JSONL prediction storage

Features
--------
* Supports both v1.0-mini (pipeline validation) and v1.0-trainval (pilot).
* Skips already-processed samples to support interrupted-run resumption.
* Saves each prediction immediately after generation (no batch buffering).
* Captures inference latency, token count, device, and error information.
* Graceful handling of missing image files and GPU OOM.

Usage (CLI)
-----------
    # Mini experiment
    python src/infer.py --config configs/config.yaml --version v1.0-mini

    # Trainval research pilot
    python src/infer.py --config configs/config.yaml --version v1.0-trainval

    # Override model
    python src/infer.py --config configs/config.yaml --model smolvlm2_2b
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    load_config,
    load_prompt,
    resolve_output_path,
    load_completed_tokens,
    append_jsonl,
    set_seed,
    setup_logging,
    get_logger,
)
from dataset_loader import NuScenesLoader
from scene_selector import select_mini_samples, select_trainval_samples
from model_loader import ModelLoader

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction record construction
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_record(
    *,
    version: str,
    split: str,
    scene_token: str,
    sample_token: str,
    timestamp: int,
    image_path: str,
    model_id: str,
    model_key: str,
    prompt: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct a flat JSONL prediction record.

    Args:
        version: Dataset version string.
        split: Split label (``"train"``, ``"val"``, or ``"mini"``).
        scene_token: nuScenes scene token.
        sample_token: nuScenes sample token.
        timestamp: Sample timestamp (microseconds).
        image_path: Absolute path to the image file.
        model_id: HuggingFace model ID string.
        model_key: Short model key (e.g. ``"qwen2_5_vl_3b"``).
        prompt: Exact prompt text sent to the model.
        result: Output dict from ``ModelLoader.generate()``.

    Returns:
        Flat dictionary ready for JSONL serialisation.
    """
    return {
        # Dataset metadata
        "dataset_version": version,
        "split": split,
        "scene_token": scene_token,
        "sample_token": sample_token,
        "timestamp_us": timestamp,
        "image_path": image_path,
        # Model metadata
        "model_id": model_id,
        "model_key": model_key,
        "prompt": prompt,
        # Inference outputs
        "generated_description": result.get("text", ""),
        "latency_sec": result.get("latency_sec", 0.0),
        "n_tokens_generated": result.get("n_tokens", 0),
        "tokens_per_sec": (
            result["n_tokens"] / result["latency_sec"]
            if result.get("latency_sec", 0) > 0 and result.get("n_tokens", 0) > 0
            else 0.0
        ),
        "device": result.get("device", "unknown"),
        "dtype": result.get("dtype", "unknown"),
        # Error field
        "error": result.get("error"),
        # Causal Token Pruning metadata (E2C-VLM Stage 1)
        "causal_pruning": result.get("causal_pruning", {}),
        # Wall-clock timestamp of this record
        "recorded_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_on_samples(
    *,
    samples: List[Dict[str, Any]],
    loader: NuScenesLoader,
    model: ModelLoader,
    prompt: str,
    output_path: Path,
    version: str,
    split: str,
    checkpoint_interval: int = 10,
) -> Dict[str, Any]:
    """Run VLM inference on a list of nuScenes sample records.

    Skips samples whose ``sample_token`` already appears in *output_path*.

    Args:
        samples: List of sample record dicts.
        loader: Initialised ``NuScenesLoader``.
        model: Loaded ``ModelLoader``.
        prompt: Fixed prompt text.
        output_path: JSONL file to append predictions to.
        version: Dataset version (for record metadata).
        split: Split label (for record metadata).
        checkpoint_interval: Log progress every N samples.

    Returns:
        Dict with keys ``n_total``, ``n_skipped``, ``n_processed``,
        ``n_errors``, ``latencies_sec``.
    """
    completed = load_completed_tokens(output_path)
    logger.info(
        "Starting inference on %d samples (%d already completed) → %s",
        len(samples),
        len(completed),
        output_path,
    )

    latencies: List[float] = []
    n_skipped = 0
    n_errors = 0
    n_processed = 0

    for idx, sample in enumerate(samples):
        sample_token = sample["token"]

        # ── Resume support ────────────────────────────────────────────────────
        if sample_token in completed:
            n_skipped += 1
            continue

        # ── Retrieve CAM_FRONT data ───────────────────────────────────────────
        cam_data = loader.get_front_camera_data(sample)
        if cam_data is None:
            record = build_prediction_record(
                version=version,
                split=split,
                scene_token=sample.get("scene_token", ""),
                sample_token=sample_token,
                timestamp=sample.get("timestamp", 0),
                image_path="",
                model_id=model.model_id,
                model_key=model.model_key,
                prompt=prompt,
                result={"error": "No CAM_FRONT channel for this sample."},
            )
            append_jsonl(output_path, record)
            n_errors += 1
            n_processed += 1
            continue

        image_path = loader.get_image_path(cam_data)

        # ── Validate image file ───────────────────────────────────────────────
        if not image_path.is_file():
            logger.warning("Image not found: %s", image_path)
            record = build_prediction_record(
                version=version,
                split=split,
                scene_token=sample.get("scene_token", ""),
                sample_token=sample_token,
                timestamp=sample.get("timestamp", 0),
                image_path=str(image_path),
                model_id=model.model_id,
                model_key=model.model_key,
                prompt=prompt,
                result={"error": f"Image file not found: {image_path}"},
            )
            append_jsonl(output_path, record)
            n_errors += 1
            n_processed += 1
            continue

        # ── VLM inference ─────────────────────────────────────────────────────
        result = model.generate(image_path=image_path, prompt=prompt)

        if result.get("error") is None:
            latencies.append(result["latency_sec"])
        else:
            n_errors += 1

        # ── Save record ───────────────────────────────────────────────────────
        record = build_prediction_record(
            version=version,
            split=split,
            scene_token=sample.get("scene_token", ""),
            sample_token=sample_token,
            timestamp=sample.get("timestamp", 0),
            image_path=str(image_path),
            model_id=model.model_id,
            model_key=model.model_key,
            prompt=prompt,
            result=result,
        )
        append_jsonl(output_path, record)
        n_processed += 1

        # ── Progress logging ──────────────────────────────────────────────────
        if (idx + 1) % checkpoint_interval == 0 or idx == len(samples) - 1:
            elapsed = result.get("latency_sec", 0.0)
            logger.info(
                "[%s|%s] %d/%d processed | errors=%d | last_latency=%.2fs",
                version,
                split,
                n_processed + n_skipped,
                len(samples),
                n_errors,
                elapsed,
            )

    return {
        "n_total": len(samples),
        "n_skipped": n_skipped,
        "n_processed": n_processed,
        "n_errors": n_errors,
        "latencies_sec": latencies,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for the inference pipeline."""
    parser = argparse.ArgumentParser(
        description="Phase 1 VLM inference on nuScenes."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config.yaml (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--dataroot",
        help="Override nuScenes dataroot directory path.",
    )
    parser.add_argument(
        "--version",
        choices=["v1.0-mini", "v1.0-trainval"],
        help="Override dataset version from config.",
    )
    parser.add_argument(
        "--model",
        choices=["qwen2_5_vl_3b", "smolvlm2_2b"],
        help="Override model key from config.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="both",
        help="Which trainval split to process (ignored for mini).",
    )
    # E2C-VLM Causal Token Pruning CLI arguments
    parser.add_argument(
        "--causal-pruning",
        action="store_true",
        default=None,
        help="Enable causality-grounded token pruning (E2C-VLM Stage 1).",
    )
    parser.add_argument(
        "--no-causal-pruning",
        action="store_true",
        help="Explicitly disable causal token pruning.",
    )
    parser.add_argument(
        "--pruning-method",
        choices=["causal", "fastv_attention", "similarity", "random", "none"],
        help="Pruning method ('causal' Eq. 1, 'fastv_attention', 'similarity', 'random', 'none').",
    )
    parser.add_argument(
        "--pruning-ratio",
        type=float,
        help="Fraction of visual tokens to prune (e.g. 0.3, 0.5, 0.7).",
    )
    parser.add_argument(
        "--intervention-mode",
        choices=["group", "taylor", "amortized"],
        help="SCM interventional distribution approximation mode ('group', 'taylor', 'amortized').",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        help="Group size for spatial super-token SCM surgery.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Causal score threshold epsilon for filtering J(v_i) < epsilon.",
    )
    args = parser.parse_args(argv)

    # ── Load configuration ─────────────────────────────────────────────────
    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file") if log_cfg.get("log_to_file") else None,
    )
    logger.info("Phase 1 Inference Pipeline — starting.")

    # CLI overrides
    if args.dataroot:
        cfg["dataset"]["dataroot"] = args.dataroot
    if args.version:
        cfg["dataset"]["version"] = args.version
    if args.model:
        cfg["model"]["name"] = args.model

    # Causal pruning overrides
    causal_cfg = cfg.setdefault("causal_pruning", {})
    if args.causal_pruning is not None:
        causal_cfg["enabled"] = True
    if args.no_causal_pruning:
        causal_cfg["enabled"] = False
    if args.pruning_method:
        causal_cfg["method"] = args.pruning_method
        if args.pruning_method != "none":
            causal_cfg["enabled"] = True
    if args.pruning_ratio is not None:
        causal_cfg["pruning_ratio"] = args.pruning_ratio
    if args.intervention_mode:
        causal_cfg["intervention_mode"] = args.intervention_mode
    if args.group_size is not None:
        causal_cfg["group_size"] = args.group_size
    if args.epsilon is not None:
        causal_cfg["threshold_epsilon"] = args.epsilon

    version = cfg["dataset"]["version"]
    seed = cfg["selection"].get("mini", {}).get("seed", 42)
    set_seed(seed if seed is not None else 42)

    prompt = load_prompt(cfg)
    checkpoint_interval = cfg.get("inference", {}).get("checkpoint_interval", 10)

    # ── Load dataset ───────────────────────────────────────────────────────
    loader = NuScenesLoader(
        dataroot=cfg["dataset"]["dataroot"],
        version=version,
        verbose=cfg["dataset"].get("verbose", False),
    )
    logger.info("Dataset summary: %s", loader.summarise())

    # ── Load model ─────────────────────────────────────────────────────────
    model = ModelLoader(cfg)
    model.reset_peak_gpu_memory()
    model.load()

    all_stats: List[Dict[str, Any]] = []

    # Tag output filename when causal pruning is active
    prune_tag = ""
    if model.causal_enabled and model.causal_method != "none" and model.pruning_ratio > 0.0:
        prune_tag = f"_{model.causal_method}_p{int(round(model.pruning_ratio * 100))}"

    # ── Run inference ──────────────────────────────────────────────────────
    if "mini" in version:
        sel_cfg = cfg["selection"]["mini"]
        samples = select_mini_samples(
            loader=loader,
            max_samples=sel_cfg.get("max_samples", 10),
            seed=sel_cfg.get("seed", 42),
        )
        output_path = resolve_output_path(
            cfg, "predictions",
            f"predictions_{version}_{model.model_key}{prune_tag}.jsonl",
        )
        stats = run_inference_on_samples(
            samples=samples,
            loader=loader,
            model=model,
            prompt=prompt,
            output_path=output_path,
            version=version,
            split="mini",
            checkpoint_interval=checkpoint_interval,
        )
        all_stats.append({"split": "mini", **stats})

    else:  # trainval
        sel_cfg = cfg["selection"]["trainval"]
        train_samples, val_samples = select_trainval_samples(
            loader=loader,
            max_train_scenes=sel_cfg.get("max_train_scenes", 50),
            max_val_scenes=sel_cfg.get("max_val_scenes", 10),
            seed=sel_cfg.get("seed", 42),
            one_sample_per_scene=sel_cfg.get("one_sample_per_scene", False),
        )

        run_splits = args.split  # "train", "val", or "both"

        if run_splits in ("train", "both") and train_samples:
            out_train = resolve_output_path(
                cfg, "predictions",
                f"predictions_{version}_train_{model.model_key}{prune_tag}.jsonl",
            )
            stats_train = run_inference_on_samples(
                samples=train_samples,
                loader=loader,
                model=model,
                prompt=prompt,
                output_path=out_train,
                version=version,
                split="train",
                checkpoint_interval=checkpoint_interval,
            )
            all_stats.append({"split": "train", **stats_train})

        if run_splits in ("val", "both") and val_samples:
            out_val = resolve_output_path(
                cfg, "predictions",
                f"predictions_{version}_val_{model.model_key}{prune_tag}.jsonl",
            )
            stats_val = run_inference_on_samples(
                samples=val_samples,
                loader=loader,
                model=model,
                prompt=prompt,
                output_path=out_val,
                version=version,
                split="val",
                checkpoint_interval=checkpoint_interval,
            )
            all_stats.append({"split": "val", **stats_val})

    # ── Summary ────────────────────────────────────────────────────────────
    peak_gpu = model.peak_gpu_memory_mb()
    logger.info("=" * 60)
    logger.info("Inference complete.")
    if peak_gpu is not None:
        logger.info("Peak GPU memory: %.1f MB", peak_gpu)
    for stat in all_stats:
        lats = stat.get("latencies_sec", [])
        import statistics
        mean_lat = statistics.mean(lats) if lats else 0.0
        logger.info(
            "Split %-10s | total=%d | processed=%d | errors=%d | "
            "mean_latency=%.2fs",
            stat["split"],
            stat["n_total"],
            stat["n_processed"],
            stat["n_errors"],
            mean_lat,
        )

    model.unload()
    logger.info("Done.")


if __name__ == "__main__":
    main()
