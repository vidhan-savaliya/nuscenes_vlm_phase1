"""
dataset_audit.py — Dataset inspection, manifest generation, and reporting.

Produces:
  1. A CSV manifest listing every selected sample with image availability,
     annotation counts, and visibility distribution.
  2. A printed / logged summary report covering scene counts, object-category
     distribution, visibility distribution, and missing-file counts.
  3. An object-category frequency table.

Usage (CLI)
-----------
    python src/dataset_audit.py --config configs/config.yaml --version v1.0-mini
    python src/dataset_audit.py --config configs/config.yaml --version v1.0-trainval
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_config, resolve_output_path, setup_logging, get_logger
from dataset_loader import NuScenesLoader
from scene_selector import select_mini_samples, select_trainval_samples

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample audit record
# ─────────────────────────────────────────────────────────────────────────────

def audit_sample(
    sample: Dict[str, Any],
    loader: NuScenesLoader,
    version: str,
    split: str,
) -> Dict[str, Any]:
    """Collect audit information for a single sample.

    Args:
        sample: nuScenes sample record.
        loader: Initialised ``NuScenesLoader``.
        version: Dataset version string.
        split: Split label.

    Returns:
        Flat dict suitable for CSV / reporting.
    """
    sample_token = sample["token"]
    scene_token = sample.get("scene_token", "")
    timestamp_us = sample.get("timestamp", 0)

    # Scene name
    try:
        scene = loader.get_scene(scene_token)
        scene_name = scene.get("name", "")
        scene_description = scene.get("description", "")
    except Exception:
        scene_name = ""
        scene_description = ""

    # CAM_FRONT data
    cam_data = loader.get_front_camera_data(sample)
    if cam_data is not None:
        image_path = loader.get_image_path(cam_data)
        image_available = image_path.is_file()
        image_path_str = str(image_path)
        image_width = cam_data.get("width", 0)
        image_height = cam_data.get("height", 0)
    else:
        image_path_str = ""
        image_available = False
        image_width = 0
        image_height = 0

    # Annotations
    try:
        annotations = loader.get_annotations(sample_token)
    except Exception as exc:
        logger.warning("Could not load annotations for %s: %s", sample_token, exc)
        annotations = []

    n_annotations = len(annotations)
    visibility_counts = Counter(
        loader.get_visibility_level(ann) for ann in annotations
    )
    category_counts = Counter(ann.get("category_name", "unknown") for ann in annotations)

    return {
        "dataset_version": version,
        "split": split,
        "scene_token": scene_token,
        "scene_name": scene_name,
        "scene_description": scene_description,
        "sample_token": sample_token,
        "timestamp_us": timestamp_us,
        "image_path": image_path_str,
        "image_available": image_available,
        "image_width": image_width,
        "image_height": image_height,
        "n_annotations": n_annotations,
        "vis_1": visibility_counts.get(1, 0),
        "vis_2": visibility_counts.get(2, 0),
        "vis_3": visibility_counts.get(3, 0),
        "vis_4": visibility_counts.get(4, 0),
        "categories_json": str(dict(category_counts)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Manifest generation
# ─────────────────────────────────────────────────────────────────────────────

_MANIFEST_FIELDS = [
    "dataset_version",
    "split",
    "scene_token",
    "scene_name",
    "scene_description",
    "sample_token",
    "timestamp_us",
    "image_path",
    "image_available",
    "image_width",
    "image_height",
    "n_annotations",
    "vis_1",
    "vis_2",
    "vis_3",
    "vis_4",
    "categories_json",
]


def write_manifest(
    records: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write audit records to a CSV manifest file.

    Args:
        records: List of audit record dicts.
        output_path: Destination CSV file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    logger.info("Manifest written → %s  (%d rows)", output_path, len(records))


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_audit_report(
    records: List[Dict[str, Any]],
    loader: NuScenesLoader,
    selected_scene_tokens: List[str],
    version: str,
) -> None:
    """Log a structured audit report to the logger.

    Args:
        records: List of audit record dicts.
        loader: Loaded ``NuScenesLoader`` (for total dataset counts).
        selected_scene_tokens: Tokens of all selected scenes.
        version: Dataset version string.
    """
    total_scenes = len(loader.get_all_scenes())
    total_samples = sum(1 for _ in loader.nusc.sample)
    selected_scenes = len(selected_scene_tokens)
    selected_samples = len(records)
    available_images = sum(1 for r in records if r["image_available"])
    missing_images = selected_samples - available_images

    # Visibility distribution
    vis_counter: Counter = Counter()
    cat_counter: Counter = Counter()
    for r in records:
        for lv in range(1, 5):
            vis_counter[lv] += r.get(f"vis_{lv}", 0)
        # Parse categories_json (simple string representation)
        import ast
        try:
            cats = ast.literal_eval(r.get("categories_json", "{}"))
            cat_counter.update(cats)
        except Exception:
            pass

    total_anns = sum(r["n_annotations"] for r in records)

    sep = "=" * 64
    logger.info(sep)
    logger.info("DATASET AUDIT REPORT — %s", version)
    logger.info(sep)
    logger.info("Total scenes in dataset   : %d", total_scenes)
    logger.info("Total samples in dataset  : %d", total_samples)
    logger.info("Selected scenes           : %d", selected_scenes)
    logger.info("Selected samples          : %d", selected_samples)
    logger.info("Images available          : %d", available_images)
    logger.info("Images MISSING            : %d", missing_images)
    logger.info("Total annotations         : %d", total_anns)
    logger.info("")
    logger.info("── Visibility distribution (across selected annotations) ──")
    vis_total = sum(vis_counter.values())
    for lv in range(1, 5):
        n = vis_counter[lv]
        pct = 100 * n / vis_total if vis_total else 0
        logger.info("  Level %d (%.0f–%.0f%% visible) : %d  (%.1f%%)",
                    lv, (lv - 1) * 20, lv * 20, n, pct)
    logger.info("")
    logger.info("── Top-20 object categories ──────────────────────────────")
    for cat, n in cat_counter.most_common(20):
        logger.info("  %-50s : %d", cat, n)
    logger.info(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="nuScenes dataset audit and manifest generation."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--dataroot",
        help="Override nuScenes dataroot directory path.",
    )
    parser.add_argument(
        "--version",
        choices=["v1.0-mini", "v1.0-trainval"],
        help="Override dataset version from config.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file") if log_cfg.get("log_to_file") else None,
    )

    if args.dataroot:
        cfg["dataset"]["dataroot"] = args.dataroot
    if args.version:
        cfg["dataset"]["version"] = args.version

    version = cfg["dataset"]["version"]

    loader = NuScenesLoader(
        dataroot=cfg["dataset"]["dataroot"],
        version=version,
        verbose=cfg["dataset"].get("verbose", False),
    )

    records: List[Dict[str, Any]] = []
    selected_scene_tokens: List[str] = []

    if "mini" in version:
        sel_cfg = cfg["selection"]["mini"]
        samples = select_mini_samples(
            loader=loader,
            max_samples=sel_cfg.get("max_samples", 10),
            seed=sel_cfg.get("seed", 42),
        )
        for s in samples:
            records.append(audit_sample(s, loader, version, "mini"))
            if s.get("scene_token") not in selected_scene_tokens:
                selected_scene_tokens.append(s["scene_token"])
    else:
        sel_cfg = cfg["selection"]["trainval"]
        train_samples, val_samples = select_trainval_samples(
            loader=loader,
            max_train_scenes=sel_cfg.get("max_train_scenes", 50),
            max_val_scenes=sel_cfg.get("max_val_scenes", 10),
            seed=sel_cfg.get("seed", 42),
            one_sample_per_scene=sel_cfg.get("one_sample_per_scene", False),
        )
        for s in train_samples:
            records.append(audit_sample(s, loader, version, "train"))
            if s.get("scene_token") not in selected_scene_tokens:
                selected_scene_tokens.append(s["scene_token"])
        for s in val_samples:
            records.append(audit_sample(s, loader, version, "val"))
            if s.get("scene_token") not in selected_scene_tokens:
                selected_scene_tokens.append(s["scene_token"])

    # Save manifest
    manifest_path = resolve_output_path(
        cfg, "manifests", f"manifest_{version}.csv"
    )
    write_manifest(records, manifest_path)

    # Print report
    print_audit_report(records, loader, selected_scene_tokens, version)


if __name__ == "__main__":
    main()
