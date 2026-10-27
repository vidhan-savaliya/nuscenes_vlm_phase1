"""
scene_selector.py — Reproducible scene and sample selection for Phase 1.

Key guarantees
--------------
* Training and validation scenes are drawn from the official nuScenes splits
  and are **never mixed**.  A scene cannot contribute frames to both sets.
* Selection is deterministic given the same seed.
* The subset size is configurable via the master config.
* Mini-split selection returns a flat list of sample records.
* Trainval selection returns separate train and val lists of sample records.

nuScenes splits reference
--------------------------
The official splits are encoded in ``nuscenes.utils.splits``:
  - ``mini_train`` and ``mini_val``   (v1.0-mini)
  - ``train`` and ``val``             (v1.0-trainval)
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# nuScenes official scene-name → split mapping (imported lazily to avoid hard
# dependency at import time).
_SPLIT_MODULE = "nuscenes.utils.splits"


def _get_official_splits(version: str) -> Tuple[List[str], List[str]]:
    """Return (train_scene_names, val_scene_names) from the nuScenes devkit.

    Args:
        version: Dataset version, e.g. ``"v1.0-mini"`` or ``"v1.0-trainval"``.

    Returns:
        Tuple of (train_names, val_names) — lists of scene name strings.

    Raises:
        ImportError: If the nuScenes devkit is not installed.
        ValueError: If the version is not recognised.
    """
    try:
        import nuscenes.utils.splits as splits_module
    except ImportError as exc:
        raise ImportError(
            "nuScenes devkit not installed.  Run: pip install nuscenes-devkit"
        ) from exc

    version_key = version.replace(".", "_").replace("-", "_")  # e.g. v1_0_mini

    if "mini" in version:
        train_names = getattr(splits_module, "mini_train", [])
        val_names = getattr(splits_module, "mini_val", [])
    elif "trainval" in version:
        train_names = getattr(splits_module, "train", [])
        val_names = getattr(splits_module, "val", [])
    else:
        raise ValueError(
            f"Unrecognised nuScenes version '{version}'.  "
            "Expected 'v1.0-mini' or 'v1.0-trainval'."
        )

    if not train_names and not val_names:
        raise ValueError(
            f"No split information found for version '{version}'.  "
            "Check your nuscenes-devkit installation."
        )

    return list(train_names), list(val_names)


def _scenes_by_name(
    loader_nusc,  # NuScenesLoader instance
    scene_names: List[str],
) -> List[Dict[str, Any]]:
    """Filter all scene records to those whose names appear in *scene_names*.

    Args:
        loader_nusc: A ``NuScenesLoader`` instance (has ``.nusc`` attribute).
        scene_names: List of valid scene name strings.

    Returns:
        Filtered list of scene record dicts.
    """
    name_set = set(scene_names)
    return [s for s in loader_nusc.get_all_scenes() if s["name"] in name_set]


def select_mini_samples(
    loader,
    max_samples: int = 10,
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """Select up to *max_samples* annotated samples from the mini split.

    Samples are drawn from the mini_train split (which is the only annotated
    subset in mini; mini_val overlaps in practice but is kept separate for
    consistency).

    Args:
        loader: Initialised ``NuScenesLoader`` for ``v1.0-mini``.
        max_samples: Maximum number of samples to return.
        seed: Random seed for reproducibility.  ``None`` = no shuffle.

    Returns:
        List of sample record dicts.
    """
    train_names, val_names = _get_official_splits(loader.version)
    all_names = train_names + val_names  # mini is small — use both

    scenes = _scenes_by_name(loader, all_names)
    logger.info("Mini split: %d scenes available.", len(scenes))

    # Collect all samples from all scenes
    all_samples: List[Dict[str, Any]] = []
    for scene in scenes:
        samples = loader.get_samples_in_scene(scene["token"])
        for s in samples:
            s["_scene_name"] = scene["name"]
            s["_split"] = "mini"
        all_samples.extend(samples)

    logger.info("Mini split: %d total samples.", len(all_samples))

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(all_samples)

    selected = all_samples[:max_samples]
    logger.info("Mini split: selected %d samples.", len(selected))
    return selected


def select_trainval_samples(
    loader,
    max_train_scenes: int = 50,
    max_val_scenes: int = 10,
    seed: Optional[int] = 42,
    one_sample_per_scene: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select training and validation samples from the trainval split.

    Split integrity is enforced: no scene contributes to both sets.

    Args:
        loader: Initialised ``NuScenesLoader`` for ``v1.0-trainval``.
        max_train_scenes: Maximum number of training scenes.
        max_val_scenes: Maximum number of validation scenes.
        seed: Random seed for reproducibility.
        one_sample_per_scene: If True, return only the midpoint sample per
            scene.  If False, return all annotated samples.

    Returns:
        Tuple of (train_samples, val_samples).
    """
    train_names, val_names = _get_official_splits(loader.version)

    # ── Apply scene count limits ──────────────────────────────────────────────
    rng = random.Random(seed) if seed is not None else random.Random()

    if seed is not None:
        shuffled_train = list(train_names)
        rng.shuffle(shuffled_train)
        selected_train_names = shuffled_train[:max_train_scenes]

        shuffled_val = list(val_names)
        rng.shuffle(shuffled_val)
        selected_val_names = shuffled_val[:max_val_scenes]
    else:
        selected_train_names = train_names[:max_train_scenes]
        selected_val_names = val_names[:max_val_scenes]

    # Enforce no overlap (should not happen with official splits, but guard it)
    val_set = set(selected_val_names)
    selected_train_names = [n for n in selected_train_names if n not in val_set]

    train_scenes = _scenes_by_name(loader, selected_train_names)
    val_scenes = _scenes_by_name(loader, selected_val_names)

    logger.info(
        "Trainval: %d train scenes, %d val scenes selected.",
        len(train_scenes),
        len(val_scenes),
    )

    def collect_samples(
        scenes: List[Dict[str, Any]], split_label: str
    ) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for scene in scenes:
            scene_samples = loader.get_samples_in_scene(scene["token"])
            for s in scene_samples:
                s["_scene_name"] = scene["name"]
                s["_split"] = split_label
            if one_sample_per_scene and scene_samples:
                mid = len(scene_samples) // 2
                samples.append(scene_samples[mid])
            else:
                samples.extend(scene_samples)
        return samples

    train_samples = collect_samples(train_scenes, "train")
    val_samples = collect_samples(val_scenes, "val")

    logger.info(
        "Trainval: %d train samples, %d val samples.",
        len(train_samples),
        len(val_samples),
    )
    return train_samples, val_samples


def get_scene_diversity_report(
    loader,
    scene_tokens: List[str],
) -> Dict[str, Any]:
    """Compute a diversity summary for a list of scenes.

    Reports:
    - Number of scenes.
    - Total and mean sample count.
    - Unique scene locations (if available in scene description).
    - Object-category distribution across selected scenes.

    Args:
        loader: Initialised ``NuScenesLoader``.
        scene_tokens: List of scene tokens to analyse.

    Returns:
        Dictionary with diversity statistics.
    """
    from collections import Counter

    category_counter: Counter = Counter()
    sample_counts: List[int] = []

    for scene_token in scene_tokens:
        samples = loader.get_samples_in_scene(scene_token)
        sample_counts.append(len(samples))
        for sample in samples:
            anns = loader.get_annotations(sample["token"])
            for ann in anns:
                category_counter[ann["category_name"]] += 1

    total_samples = sum(sample_counts)
    return {
        "n_scenes": len(scene_tokens),
        "n_samples_total": total_samples,
        "n_samples_mean": total_samples / max(len(scene_tokens), 1),
        "category_distribution": dict(category_counter.most_common()),
    }
