"""
test_output_schema.py — Tests validating the JSONL prediction record schema.

Verifies that:
  1. Required fields are present in every prediction record.
  2. Field types are correct.
  3. The resume mechanism correctly identifies completed sample tokens.
  4. build_prediction_record() produces valid records.
  5. Manifest CSV rows conform to the expected schema.

These tests do NOT require dataset downloads, model weights, or CUDA.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ─────────────────────────────────────────────────────────────────────────────
# Required fields for the prediction JSONL schema
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_PREDICTION_FIELDS = {
    "dataset_version": str,
    "split": str,
    "scene_token": str,
    "sample_token": str,
    "timestamp_us": int,
    "image_path": str,
    "model_id": str,
    "model_key": str,
    "prompt": str,
    "generated_description": str,
    "latency_sec": float,
    "n_tokens_generated": int,
    "tokens_per_sec": float,
    "device": str,
    "dtype": str,
    "recorded_at": str,
    # "error" may be None or str — checked separately
}

REQUIRED_MANIFEST_FIELDS = {
    "dataset_version", "split", "scene_token", "scene_name",
    "sample_token", "timestamp_us", "image_path", "image_available",
    "image_width", "image_height", "n_annotations",
    "vis_1", "vis_2", "vis_3", "vis_4",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_valid_result(error=None):
    return {
        "text": "A car is visible ahead on a straight road.",
        "n_tokens": 42,
        "latency_sec": 1.23,
        "device": "cpu",
        "dtype": "float32",
        "error": error,
    }


def _build_record(sample_token="tok001", error=None):
    from infer import build_prediction_record
    return build_prediction_record(
        version="v1.0-mini",
        split="mini",
        scene_token="scene001",
        sample_token=sample_token,
        timestamp=1000000,
        image_path="/data/nuscenes/samples/CAM_FRONT/img.jpg",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        model_key="qwen2_5_vl_3b",
        prompt="Describe the scene.",
        result=_make_valid_result(error=error),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionSchema(unittest.TestCase):
    """Verify that build_prediction_record() produces the required schema."""

    def test_all_required_fields_present(self):
        record = _build_record()
        for field in REQUIRED_PREDICTION_FIELDS:
            self.assertIn(field, record, f"Missing field: {field}")

    def test_field_types(self):
        record = _build_record()
        for field, expected_type in REQUIRED_PREDICTION_FIELDS.items():
            self.assertIsInstance(
                record[field], expected_type,
                f"Field '{field}' has wrong type: "
                f"got {type(record[field])}, expected {expected_type}",
            )

    def test_error_field_is_none_on_success(self):
        record = _build_record(error=None)
        self.assertIsNone(record["error"])

    def test_error_field_is_string_on_failure(self):
        record = _build_record(error="OOM error")
        self.assertIsInstance(record["error"], str)

    def test_tokens_per_sec_positive(self):
        record = _build_record()
        self.assertGreater(record["tokens_per_sec"], 0.0)

    def test_tokens_per_sec_zero_on_zero_latency(self):
        from infer import build_prediction_record
        result = {"text": "", "n_tokens": 0, "latency_sec": 0.0,
                  "device": "cpu", "dtype": "float32", "error": None}
        record = build_prediction_record(
            version="v1.0-mini", split="mini",
            scene_token="s", sample_token="t", timestamp=0,
            image_path="/img.jpg", model_id="m", model_key="k",
            prompt="p", result=result,
        )
        self.assertEqual(record["tokens_per_sec"], 0.0)

    def test_recorded_at_is_iso_format(self):
        import datetime
        record = _build_record()
        # Should parse without error
        dt_str = record["recorded_at"].replace("Z", "+00:00")
        try:
            datetime.datetime.fromisoformat(dt_str)
        except ValueError as exc:
            self.fail(f"recorded_at is not valid ISO format: {exc}")

    def test_record_is_json_serialisable(self):
        record = _build_record()
        try:
            json.dumps(record)
        except (TypeError, ValueError) as exc:
            self.fail(f"Record is not JSON serialisable: {exc}")

    def test_record_with_error_is_json_serialisable(self):
        record = _build_record(error="Something went wrong")
        try:
            json.dumps(record)
        except (TypeError, ValueError) as exc:
            self.fail(f"Error record is not JSON serialisable: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Resume mechanism tests
# ─────────────────────────────────────────────────────────────────────────────

class TestResumeSupport(unittest.TestCase):
    """Verify that completed sample tokens are correctly identified."""

    def test_resume_skips_existing_tokens(self):
        from utils import append_jsonl, load_completed_tokens
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.jsonl"
            # Write 3 completed records
            for i in range(3):
                append_jsonl(path, {"sample_token": f"tok{i}", "text": "done"})
            completed = load_completed_tokens(path)

        self.assertEqual(len(completed), 3)
        for i in range(3):
            self.assertIn(f"tok{i}", completed)

    def test_resume_handles_empty_file(self):
        from utils import load_completed_tokens
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.jsonl"
            completed = load_completed_tokens(path)
        self.assertEqual(len(completed), 0)

    def test_resume_ignores_records_without_token(self):
        from utils import append_jsonl, load_completed_tokens
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.jsonl"
            append_jsonl(path, {"sample_token": "tok1"})
            append_jsonl(path, {"no_token_key": "oops"})
            completed = load_completed_tokens(path)
        self.assertEqual(completed, {"tok1"})

    def test_resume_handles_malformed_lines_gracefully(self):
        """A corrupted JSONL line should be skipped, not crash."""
        from utils import load_jsonl
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrupted.jsonl"
            with path.open("w") as fh:
                fh.write('{"sample_token": "good1"}\n')
                fh.write("NOT VALID JSON{{{\n")
                fh.write('{"sample_token": "good2"}\n')
            records = load_jsonl(path)
        tokens = {r["sample_token"] for r in records if "sample_token" in r}
        self.assertIn("good1", tokens)
        self.assertIn("good2", tokens)
        self.assertEqual(len(records), 2)  # corrupted line dropped


# ─────────────────────────────────────────────────────────────────────────────
# Manifest CSV schema tests
# ─────────────────────────────────────────────────────────────────────────────

class TestManifestSchema(unittest.TestCase):
    """Verify the manifest CSV produced by dataset_audit.py."""

    def test_manifest_has_all_required_columns(self):
        from dataset_audit import write_manifest, _MANIFEST_FIELDS

        row = {
            "dataset_version": "v1.0-mini",
            "split": "mini",
            "scene_token": "scene001",
            "scene_name": "scene-0001",
            "scene_description": "Urban driving",
            "sample_token": "tok001",
            "timestamp_us": 1000000,
            "image_path": "/data/img.jpg",
            "image_available": True,
            "image_width": 1600,
            "image_height": 900,
            "n_annotations": 5,
            "vis_1": 0, "vis_2": 1, "vis_3": 2, "vis_4": 2,
            "categories_json": "{'vehicle.car': 3}",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.csv"
            write_manifest([row], path)
            with path.open("r", newline="") as fh:
                reader = csv.DictReader(fh)
                columns = set(reader.fieldnames or [])
                rows = list(reader)

        self.assertGreaterEqual(len(rows), 1)
        for field in REQUIRED_MANIFEST_FIELDS:
            self.assertIn(field, columns, f"Manifest missing column: {field}")

    def test_manifest_row_count(self):
        from dataset_audit import write_manifest

        rows = [
            {
                "dataset_version": "v1.0-mini", "split": "mini",
                "scene_token": f"s{i}", "scene_name": f"scene-{i:04d}",
                "scene_description": "", "sample_token": f"tok{i}",
                "timestamp_us": i, "image_path": "", "image_available": False,
                "image_width": 0, "image_height": 0, "n_annotations": 0,
                "vis_1": 0, "vis_2": 0, "vis_3": 0, "vis_4": 0,
                "categories_json": "{}",
            }
            for i in range(7)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.csv"
            write_manifest(rows, path)
            with path.open("r", newline="") as fh:
                data_rows = list(csv.DictReader(fh))

        self.assertEqual(len(data_rows), 7)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics JSON schema tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsSchema(unittest.TestCase):
    """Verify that aggregate_metrics() output contains required keys."""

    REQUIRED_KEYS = {
        "n_total", "n_evaluated", "n_skipped",
        "mean_recall_focus_categories",
        "mean_recall_road_users",
        "recall_by_visibility",
        "mean_unsupported_mentions_per_sample",
        "hallucination_review_samples",
        "first_inference_latency_sec",
        "mean_latency_sec",
        "median_latency_sec",
        "p90_latency_sec",
        "p95_latency_sec",
        "mean_tokens_per_sec",
        "evaluation_notes",
    }

    def _make_per_sample_results(self, n=5):
        return [
            {
                "skipped": False, "reason": None,
                "sample_token": f"tok{i}",
                "n_focus_gt": 2, "n_focus_mentioned": 1,
                "recall_focus": 0.5,
                "n_road_user_gt": 1, "n_road_user_mentioned": 0,
                "recall_road_users": 0.0,
                "vis_gt": {4: 2}, "vis_tp": {4: 1},
                "n_unsupported_mentions": 0,
                "unsupported_categories": [],
                "latency_sec": 1.0 + i * 0.2,
                "n_tokens": 80,
                "tokens_per_sec": 60.0,
                "device": "cpu",
                "dtype": "float32",
            }
            for i in range(n)
        ]

    def test_all_required_keys_present(self):
        from evaluation import aggregate_metrics
        m = aggregate_metrics(self._make_per_sample_results())
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, m, f"Missing metric key: '{key}'")

    def test_recall_by_visibility_has_four_levels(self):
        from evaluation import aggregate_metrics
        m = aggregate_metrics(self._make_per_sample_results())
        vis_recalls = m["recall_by_visibility"]
        for lv in range(1, 5):
            self.assertIn(f"level_{lv}", vis_recalls)

    def test_evaluation_notes_are_non_empty_strings(self):
        from evaluation import aggregate_metrics
        m = aggregate_metrics(self._make_per_sample_results())
        notes = m["evaluation_notes"]
        self.assertIsInstance(notes, list)
        self.assertGreater(len(notes), 0)
        for note in notes:
            self.assertIsInstance(note, str)
            self.assertGreater(len(note), 0)

    def test_metrics_json_serialisable(self):
        from evaluation import aggregate_metrics
        m = aggregate_metrics(self._make_per_sample_results())
        try:
            json.dumps(m)
        except (TypeError, ValueError) as exc:
            self.fail(f"Metrics are not JSON serialisable: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
