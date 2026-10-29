"""
evaluation.py — Baseline evaluation of VLM scene descriptions.

Evaluates predictions stored in JSONL files against nuScenes ground-truth
annotations and computes:

  Scene understanding
  -------------------
  * Category recall: fraction of focus-category objects mentioned by name in
    the generated description.
  * Visibility-stratified recall: per visibility level (1–4).
  * Road-user detection recall (vehicles + pedestrians).

  Hallucination indicators (heuristic)
  -------------------------------------
  * Mentions of focus-category terms NOT supported by any annotation.
  * Mean unsupported-mention rate per sample.
  * A flag list of samples for manual review.

  Efficiency
  -----------
  * Model-loading time (from inference metadata if available).
  * First-inference latency.
  * Mean, median, P90, P95 latency.
  * Peak GPU memory (if recorded).
  * Tokens per second.

Evaluation assumptions (documented)
-------------------------------------
* nuScenes 3-D annotations are used as evidence.  An object may be annotated
  in 3-D but not visible from CAM_FRONT (behind ego, heavily occluded, etc.).
* Projection into the image plane is used to filter obviously invisible objects
  (depth <= 0), but occlusion is estimated only via the visibility_token field.
* Text matching is keyword-based (not semantically exact).  Fine-grained
  attribute errors (e.g. "bus" vs "truck") are counted as misses.
* Fluent or grammatically correct output is NOT treated as proof of correctness.

Usage (CLI)
-----------
    python src/evaluation.py \\
        --config configs/config.yaml \\
        --predictions outputs/predictions/predictions_v1.0-mini_qwen2_5_vl_3b.jsonl \\
        --version v1.0-mini
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    load_config,
    load_jsonl,
    resolve_output_path,
    setup_logging,
    get_logger,
    safe_divide,
)
from dataset_loader import NuScenesLoader

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Category keyword mapping
# ─────────────────────────────────────────────────────────────────────────────

# Maps nuScenes category name → list of keywords that a correct VLM description
# might include.  Used for recall estimation.
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "vehicle.car": ["car", "sedan", "suv", "hatchback", "vehicle"],
    "vehicle.truck": ["truck", "lorry", "pickup"],
    "vehicle.bus.bendy": ["bus", "articulated bus", "bendy bus"],
    "vehicle.bus.rigid": ["bus"],
    "vehicle.bicycle": ["bicycle", "bike", "cyclist"],
    "vehicle.motorcycle": ["motorcycle", "motorbike", "scooter"],
    "vehicle.trailer": ["trailer"],
    "vehicle.construction": ["construction vehicle", "excavator", "bulldozer"],
    "vehicle.emergency.ambulance": ["ambulance"],
    "vehicle.emergency.police": ["police car", "police vehicle"],
    "human.pedestrian.adult": ["pedestrian", "person", "people", "walker", "individual"],
    "human.pedestrian.child": ["child", "pedestrian", "kid"],
    "human.pedestrian.construction_worker": ["construction worker", "worker", "pedestrian"],
    "human.pedestrian.police_officer": ["police officer", "police", "officer"],
    "movable_object.barrier": ["barrier", "fence", "divider"],
    "movable_object.trafficcone": ["traffic cone", "cone"],
    "movable_object.pushable_pullable": ["cart", "trolley", "stroller"],
    "static_object.bicycle_rack": ["bicycle rack", "bike rack"],
    "animal": ["animal", "dog", "cat"],
}


def _text_mentions_category(text: str, category: str) -> bool:
    """Return True if *text* contains at least one keyword for *category*.

    Args:
        text: Generated description (lowercased internally).
        category: nuScenes category name (e.g. ``"vehicle.car"``).

    Returns:
        Boolean.
    """
    lower_text = text.lower()
    for kw in CATEGORY_KEYWORDS.get(category, []):
        if kw in lower_text:
            return True
    return False


def _all_keywords_in_text(text: str) -> List[str]:
    """Return all category keywords found in text."""
    lower_text = text.lower()
    found: List[str] = []
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower_text:
                found.append(cat)
                break
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sample(
    prediction: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    loader: NuScenesLoader,
    focus_categories: List[str],
    max_projection_depth_m: float = 50.0,
) -> Dict[str, Any]:
    """Evaluate a single prediction against its annotations.

    Args:
        prediction: JSONL prediction record.
        annotations: List of sample_annotation records for this sample.
        loader: ``NuScenesLoader`` (for projection).
        focus_categories: Category names to evaluate recall on.
        max_projection_depth_m: Skip annotations deeper than this.

    Returns:
        Per-sample evaluation dict.
    """
    text = prediction.get("generated_description", "") or ""
    sample_token = prediction.get("sample_token", "")
    error = prediction.get("error")

    if error or not text:
        return {
            "sample_token": sample_token,
            "skipped": True,
            "reason": error or "empty description",
        }

    # ── Filter annotations to focus categories and visible objects ────────────
    focus_anns = [
        ann for ann in annotations
        if ann.get("category_name", "") in focus_categories
    ]

    # Visibility-stratified recall
    vis_tp: Counter = Counter()   # true positives per visibility level
    vis_gt: Counter = Counter()   # ground-truth count per visibility level

    for ann in focus_anns:
        cat = ann["category_name"]
        vis = loader.get_visibility_level(ann)
        vis_gt[vis] += 1
        if _text_mentions_category(text, cat):
            vis_tp[vis] += 1

    # Overall recall for focus categories
    n_focus_gt = len(focus_anns)
    n_focus_mentioned = sum(
        1 for ann in focus_anns
        if _text_mentions_category(text, ann["category_name"])
    )
    recall = safe_divide(n_focus_mentioned, n_focus_gt)

    # Road-user recall (vehicles + pedestrians)
    road_user_cats = {
        c for c in focus_categories
        if c.startswith("vehicle.") or c.startswith("human.pedestrian.")
    }
    road_user_anns = [a for a in focus_anns if a["category_name"] in road_user_cats]
    n_ru_gt = len(road_user_anns)
    n_ru_mentioned = sum(
        1 for ann in road_user_anns
        if _text_mentions_category(text, ann["category_name"])
    )
    road_user_recall = safe_divide(n_ru_mentioned, n_ru_gt)

    # ── Hallucination heuristic ───────────────────────────────────────────────
    # Categories mentioned by the model
    mentioned_categories = set(_all_keywords_in_text(text))
    # Categories actually present in annotations (any visibility)
    present_categories = {ann["category_name"] for ann in annotations}
    # Unsupported mentions: model mentioned a category with NO annotation
    unsupported = mentioned_categories - present_categories
    n_unsupported = len(unsupported)

    return {
        "sample_token": sample_token,
        "skipped": False,
        "reason": None,
        # Recall
        "n_focus_gt": n_focus_gt,
        "n_focus_mentioned": n_focus_mentioned,
        "recall_focus": recall,
        "n_road_user_gt": n_ru_gt,
        "n_road_user_mentioned": n_ru_mentioned,
        "recall_road_users": road_user_recall,
        # Visibility-stratified recall
        "vis_gt": dict(vis_gt),
        "vis_tp": dict(vis_tp),
        # Hallucination
        "n_unsupported_mentions": n_unsupported,
        "unsupported_categories": sorted(unsupported),
        # Efficiency (from inference record)
        "latency_sec": prediction.get("latency_sec", 0.0),
        "n_tokens": prediction.get("n_tokens_generated", 0),
        "tokens_per_sec": prediction.get("tokens_per_sec", 0.0),
        "device": prediction.get("device", ""),
        "dtype": prediction.get("dtype", ""),
        "causal_pruning": prediction.get("causal_pruning", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_metrics(
    per_sample_results: List[Dict[str, Any]],
    hallucination_review_n: int = 20,
) -> Dict[str, Any]:
    """Aggregate per-sample evaluation results into final metrics.

    Args:
        per_sample_results: List of dicts from ``evaluate_sample()``.
        hallucination_review_n: Number of samples to flag for manual review.

    Returns:
        Aggregated metrics dictionary.
    """
    evaluated = [r for r in per_sample_results if not r.get("skipped")]
    skipped = [r for r in per_sample_results if r.get("skipped")]

    if not evaluated:
        logger.warning("No successfully evaluated samples.")
        return {
            "n_total": len(per_sample_results),
            "n_evaluated": 0,
            "n_skipped": len(skipped),
        }

    # ── Recall ────────────────────────────────────────────────────────────────
    recalls = [r["recall_focus"] for r in evaluated if "recall_focus" in r]
    ru_recalls = [r["recall_road_users"] for r in evaluated if "recall_road_users" in r]

    # Visibility-stratified recall
    vis_gt_total: Counter = Counter()
    vis_tp_total: Counter = Counter()
    for r in evaluated:
        for lv, n in r.get("vis_gt", {}).items():
            vis_gt_total[int(lv)] += n
        for lv, n in r.get("vis_tp", {}).items():
            vis_tp_total[int(lv)] += n
    vis_recall = {
        lv: safe_divide(vis_tp_total[lv], vis_gt_total[lv])
        for lv in range(1, 5)
    }

    # ── Hallucination ─────────────────────────────────────────────────────────
    unsupported_counts = [r.get("n_unsupported_mentions", 0) for r in evaluated]
    mean_unsupported = safe_divide(sum(unsupported_counts), len(unsupported_counts))

    # Flag samples with most unsupported mentions for manual review
    flagged = sorted(
        [r for r in evaluated if r.get("n_unsupported_mentions", 0) > 0],
        key=lambda x: x["n_unsupported_mentions"],
        reverse=True,
    )[:hallucination_review_n]

    # ── Efficiency ────────────────────────────────────────────────────────────
    latencies = [r["latency_sec"] for r in evaluated if r.get("latency_sec", 0) > 0]
    tokens_per_sec = [r["tokens_per_sec"] for r in evaluated if r.get("tokens_per_sec", 0) > 0]

    def percentile(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        lo, hi = int(k), math.ceil(k)
        if lo == hi:
            return sorted_data[lo]
        return sorted_data[lo] * (hi - k) + sorted_data[hi] * (k - lo)

    first_latency = latencies[0] if latencies else 0.0

    # ── Causal Compression Metrics (E2C-VLM Stage 1) ─────────────
    causal_entries = [r.get("causal_pruning", {}) for r in evaluated if r.get("causal_pruning")]
    if causal_entries and any(e.get("enabled") for e in causal_entries):
        ratios = [
            e.get("compression_ratio")
            if (e.get("compression_ratio") or 0.0) > 0.0
            else e.get("pruning_ratio", 0.0)
            for e in causal_entries
            if e.get("compression_ratio") is not None or e.get("pruning_ratio") is not None
        ]
        orig_tokens = [e.get("original_visual_tokens") for e in causal_entries if e.get("original_visual_tokens")]
        retained_tokens = [e.get("retained_visual_tokens") for e in causal_entries if e.get("retained_visual_tokens")]
        prune_latencies = [e.get("pruning_latency_sec", 0.0) for e in causal_entries if e.get("pruning_latency_sec") is not None]
        methods = list({e.get("method") for e in causal_entries if e.get("method")})

        compression_summary = {
            "causal_pruning_enabled": True,
            "pruning_methods": methods,
            "mean_compression_ratio": safe_divide(sum(ratios), len(ratios)),
            "mean_original_visual_tokens": safe_divide(sum(orig_tokens), len(orig_tokens)) if orig_tokens else 0,
            "mean_retained_visual_tokens": safe_divide(sum(retained_tokens), len(retained_tokens)) if retained_tokens else 0,
            "mean_pruning_latency_sec": safe_divide(sum(prune_latencies), len(prune_latencies)) if prune_latencies else 0.0,
        }
    else:
        compression_summary = {
            "causal_pruning_enabled": False,
            "mean_compression_ratio": 0.0,
        }

    return {
        # Counts
        "n_total": len(per_sample_results),
        "n_evaluated": len(evaluated),
        "n_skipped": len(skipped),
        # Recall
        "mean_recall_focus_categories": safe_divide(sum(recalls), len(recalls)),
        "mean_recall_road_users": safe_divide(sum(ru_recalls), len(ru_recalls)),
        "recall_by_visibility": {
            f"level_{lv}": vis_recall[lv] for lv in range(1, 5)
        },
        # Hallucination
        "mean_unsupported_mentions_per_sample": mean_unsupported,
        "hallucination_review_samples": [
            {"sample_token": r["sample_token"],
             "n_unsupported": r["n_unsupported_mentions"],
             "unsupported_cats": r["unsupported_categories"]}
            for r in flagged
        ],
        # Efficiency
        "first_inference_latency_sec": first_latency,
        "mean_latency_sec": statistics.mean(latencies) if latencies else 0.0,
        "median_latency_sec": statistics.median(latencies) if latencies else 0.0,
        "p90_latency_sec": percentile(latencies, 90),
        "p95_latency_sec": percentile(latencies, 95),
        "mean_tokens_per_sec": safe_divide(sum(tokens_per_sec), len(tokens_per_sec)),
        # Causal Compression
        "compression_metrics": compression_summary,
        # Caveats
        "evaluation_notes": [
            "Recall is estimated using keyword matching; it may over- or under-count.",
            "3-D annotations include objects not visible from CAM_FRONT.",
            "Hallucination detection is heuristic; manual review is required.",
            "Fluent output is NOT evidence of factual correctness.",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VLM predictions against nuScenes annotations."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--dataroot",
        help="Override nuScenes dataroot directory path.",
    )
    parser.add_argument(
        "--predictions",
        help="Path to the JSONL predictions file.",
    )
    parser.add_argument(
        "--version",
        choices=["v1.0-mini", "v1.0-trainval"],
        help="Override dataset version.",
    )
    parser.add_argument(
        "--output-metrics",
        help="Override output metrics file path.",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Paths to multiple metrics JSON files to produce a comparison table.",
    )
    args = parser.parse_args(argv)

    if args.compare:
        print_comparison_report(args.compare)
        return

    if not args.predictions:
        parser.error("--predictions is required unless using --compare.")

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

    eval_cfg = cfg.get("evaluation", {})
    focus_categories: List[str] = eval_cfg.get("focus_categories", list(CATEGORY_KEYWORDS))
    max_depth: float = eval_cfg.get("max_projection_depth_m", 50.0)
    review_n: int = eval_cfg.get("hallucination_review_n", 20)

    # Load dataset
    loader = NuScenesLoader(
        dataroot=cfg["dataset"]["dataroot"],
        version=version,
        verbose=False,
    )

    # Load predictions
    predictions = load_jsonl(args.predictions)
    logger.info("Loaded %d prediction records from %s", len(predictions), args.predictions)

    if not predictions:
        logger.error("No predictions found.  Run infer.py first.")
        sys.exit(1)

    # ── Per-sample evaluation ────────────────────────────────────────────────
    per_sample: List[Dict[str, Any]] = []
    for pred in predictions:
        sample_token = pred.get("sample_token", "")
        try:
            annotations = loader.get_annotations(sample_token)
        except Exception as exc:
            logger.warning("Annotation load failed for %s: %s", sample_token, exc)
            annotations = []
        result = evaluate_sample(
            prediction=pred,
            annotations=annotations,
            loader=loader,
            focus_categories=focus_categories,
            max_projection_depth_m=max_depth,
        )
        per_sample.append(result)

    # ── Aggregate ────────────────────────────────────────────────────────────
    metrics = aggregate_metrics(per_sample, hallucination_review_n=review_n)
    metrics["predictions_file"] = str(args.predictions)
    metrics["dataset_version"] = version

    # ── Save metrics ─────────────────────────────────────────────────────────
    if args.output_metrics:
        metrics_path = Path(args.output_metrics)
    else:
        pred_stem = Path(args.predictions).stem
        metrics_path = resolve_output_path(
            cfg, "metrics", f"metrics_{pred_stem}.json"
        )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False, default=str)
    logger.info("Metrics saved → %s", metrics_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    sep = "=" * 60
    logger.info(sep)
    logger.info("EVALUATION SUMMARY — %s", version)
    logger.info(sep)
    logger.info("Samples evaluated   : %d / %d", metrics["n_evaluated"], metrics["n_total"])
    logger.info("Focus-category recall (mean) : %.3f", metrics["mean_recall_focus_categories"])
    logger.info("Road-user recall    (mean)   : %.3f", metrics["mean_recall_road_users"])
    logger.info("Unsupported mentions (mean)  : %.2f", metrics["mean_unsupported_mentions_per_sample"])
    logger.info("Mean latency                 : %.2f s", metrics["mean_latency_sec"])
    logger.info("Median latency               : %.2f s", metrics["median_latency_sec"])
    logger.info("P90 latency                  : %.2f s", metrics["p90_latency_sec"])
    logger.info("P95 latency                  : %.2f s", metrics["p95_latency_sec"])
    logger.info("Mean tokens / sec            : %.1f", metrics["mean_tokens_per_sec"])
    if metrics["compression_metrics"].get("causal_pruning_enabled"):
        cm = metrics["compression_metrics"]
        logger.info(
            "Causal Token Pruning         : %s (%.1f%% tokens pruned)",
            ", ".join(cm.get("pruning_methods", ["unknown"])),
            cm.get("mean_compression_ratio", 0.0) * 100.0,
        )
        logger.info(
            "Visual Tokens (orig->kept)   : %.1f -> %.1f",
            cm.get("mean_original_visual_tokens", 0),
            cm.get("mean_retained_visual_tokens", 0),
        )
    logger.info(sep)
    logger.info("NOTES:")
    for note in metrics.get("evaluation_notes", []):
        logger.info("  • %s", note)
    logger.info(sep)


def print_comparison_report(metrics_paths: List[str]) -> None:
    """Load multiple metrics JSON files and print a comparative markdown table."""
    records = []
    for p in metrics_paths:
        path = Path(p)
        if not path.is_file():
            print(f"File not found: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            data["_filename"] = path.stem
            records.append(data)

    if not records:
        print("No valid metrics files loaded.")
        return

    print("\n" + "=" * 90)
    print("E2C-VLM STAGE 1: CAUSAL TOKEN PRUNING COMPARATIVE BENCHMARK")
    print("=" * 90)
    header = (
        f"{'Experiment':<35} | {'Recall (Focus)':<14} | {'Recall (RU)':<12} | "
        f"{'Unsupported':<12} | {'Tokens Pruned':<14} | {'Latency':<10}"
    )
    print(header)
    print("-" * len(header))

    for r in records:
        raw_name = r.get("_filename", "unknown")
        clean_name = raw_name.replace("metrics_predictions_v1.0-mini_", "").replace("predictions_v1.0-mini_", "")
        if clean_name == "qwen2_5_vl_3b":
            clean_name = "baseline (unpruned)"
        focus_rec = r.get("mean_recall_focus_categories", 0.0)
        ru_rec = r.get("mean_recall_road_users", 0.0)
        unsupp = r.get("mean_unsupported_mentions_per_sample", 0.0)
        comp_info = r.get("compression_metrics", {})
        comp = comp_info.get("mean_compression_ratio", 0.0) * 100.0
        lat = r.get("mean_latency_sec", 0.0)

        print(
            f"{clean_name:<35} | {focus_rec:<14.3f} | {ru_rec:<12.3f} | "
            f"{unsupp:<12.2f} | {comp:<13.1f}% | {lat:<9.2f}s"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
