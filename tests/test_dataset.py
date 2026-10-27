"""
test_dataset.py — Unit tests for dataset utilities that do NOT require
downloading nuScenes data or model weights.

All tests use in-memory mocks or minimal synthetic fixtures.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make src/ importable
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ─────────────────────────────────────────────────────────────────────────────
# Utils tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUtils(unittest.TestCase):
    """Tests for src/utils.py."""

    def test_safe_divide_normal(self):
        from utils import safe_divide
        self.assertAlmostEqual(safe_divide(10.0, 4.0), 2.5)

    def test_safe_divide_by_zero(self):
        from utils import safe_divide
        self.assertEqual(safe_divide(5.0, 0.0), 0.0)
        self.assertEqual(safe_divide(5.0, 0.0, default=-1.0), -1.0)

    def test_flatten_list(self):
        from utils import flatten_list
        result = flatten_list([[1, 2], [3], [4, 5]])
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_flatten_list_empty(self):
        from utils import flatten_list
        self.assertEqual(flatten_list([]), [])

    def test_chunks(self):
        from utils import chunks
        result = list(chunks([1, 2, 3, 4, 5], 2))
        self.assertEqual(result, [[1, 2], [3, 4], [5]])

    def test_timer_measures_elapsed(self):
        from utils import Timer
        import time
        with Timer() as t:
            time.sleep(0.05)
        self.assertGreater(t.elapsed, 0.04)
        self.assertLess(t.elapsed, 1.0)

    def test_set_seed_no_crash(self):
        from utils import set_seed
        set_seed(42)  # Should not raise

    def test_load_config_missing_file(self):
        from utils import load_config
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_jsonl_missing_file(self):
        from utils import load_jsonl
        result = load_jsonl("/nonexistent/path/file.jsonl")
        self.assertEqual(result, [])

    def test_append_and_load_jsonl(self, tmp_path=None):
        """Test round-trip write and read of JSONL."""
        import tempfile, os
        from utils import append_jsonl, load_jsonl
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            append_jsonl(path, {"a": 1, "b": "hello"})
            append_jsonl(path, {"a": 2, "b": "world"})
            records = load_jsonl(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["a"], 1)
        self.assertEqual(records[1]["b"], "world")

    def test_load_completed_tokens(self):
        import tempfile
        from utils import append_jsonl, load_completed_tokens
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.jsonl"
            append_jsonl(path, {"sample_token": "abc123", "text": "foo"})
            append_jsonl(path, {"sample_token": "def456", "text": "bar"})
            tokens = load_completed_tokens(path)
        self.assertIn("abc123", tokens)
        self.assertIn("def456", tokens)
        self.assertEqual(len(tokens), 2)


# ─────────────────────────────────────────────────────────────────────────────
# NuScenesLoader tests (mocked)
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_nusc():
    """Create a minimal mock NuScenes object."""
    mock_nusc = MagicMock()
    mock_nusc.scene = [
        {"token": "scene001", "name": "scene-0001",
         "first_sample_token": "s001", "description": "Urban driving"},
    ]
    mock_nusc.sample = [
        {"token": "s001", "scene_token": "scene001",
         "timestamp": 1000000, "data": {"CAM_FRONT": "sd001"}, "anns": ["ann001"],
         "next": "s002", "prev": ""},
        {"token": "s002", "scene_token": "scene001",
         "timestamp": 1500000, "data": {"CAM_FRONT": "sd002"}, "anns": [],
         "next": "", "prev": "s001"},
    ]
    mock_nusc.sample_data = [
        {"token": "sd001", "filename": "samples/CAM_FRONT/img001.jpg",
         "width": 1600, "height": 900,
         "calibrated_sensor_token": "cs001", "ego_pose_token": "ep001"},
    ]
    mock_nusc.category = [
        {"token": "cat001", "name": "vehicle.car"},
    ]

    def mock_get(table, token):
        mapping = {
            "scene": {"scene001": {"token": "scene001", "name": "scene-0001",
                                   "first_sample_token": "s001",
                                   "description": "Urban driving"}},
            "sample": {
                "s001": {"token": "s001", "scene_token": "scene001",
                          "timestamp": 1000000,
                          "data": {"CAM_FRONT": "sd001"}, "anns": ["ann001"],
                          "next": "s002", "prev": ""},
                "s002": {"token": "s002", "scene_token": "scene001",
                          "timestamp": 1500000,
                          "data": {"CAM_FRONT": "sd002"}, "anns": [],
                          "next": "", "prev": "s001"},
            },
            "sample_data": {
                "sd001": {"token": "sd001",
                           "filename": "samples/CAM_FRONT/img001.jpg",
                           "width": 1600, "height": 900,
                           "calibrated_sensor_token": "cs001",
                           "ego_pose_token": "ep001"},
                "sd002": {"token": "sd002",
                           "filename": "samples/CAM_FRONT/img002.jpg",
                           "width": 1600, "height": 900,
                           "calibrated_sensor_token": "cs001",
                           "ego_pose_token": "ep001"},
            },
            "sample_annotation": {
                "ann001": {"token": "ann001",
                            "category_token": "cat001",
                            "visibility_token": "3",
                            "translation": [10.0, 0.0, 1.0],
                            "size": [1.8, 4.5, 1.5],
                            "num_lidar_pts": 20,
                            "num_radar_pts": 2},
            },
            "category": {
                "cat001": {"token": "cat001", "name": "vehicle.car"},
            },
            "calibrated_sensor": {
                "cs001": {"token": "cs001",
                           "translation": [1.7, 0.0, 1.5],
                           "rotation": [1.0, 0.0, 0.0, 0.0],
                           "camera_intrinsic": [
                               [1266.4, 0.0, 816.0],
                               [0.0, 1266.4, 491.5],
                               [0.0, 0.0, 1.0],
                           ]},
            },
            "ego_pose": {
                "ep001": {"token": "ep001",
                           "translation": [0.0, 0.0, 0.0],
                           "rotation": [1.0, 0.0, 0.0, 0.0]},
            },
        }
        return mapping.get(table, {}).get(token, {})

    mock_nusc.get.side_effect = mock_get
    return mock_nusc


class TestNuScenesLoaderMocked(unittest.TestCase):
    """Tests for NuScenesLoader using a mock NuScenes object."""

    def _make_loader(self, dataroot="/fake/root"):
        """Create a NuScenesLoader with a mock NuScenes backend."""
        # Patch nuscenes.nuscenes.NuScenes and the dataroot existence check
        import importlib
        import tempfile, os

        tmpdir = tempfile.mkdtemp()
        meta_dir = Path(tmpdir) / "v1.0-mini"
        meta_dir.mkdir(parents=True)

        # Create a mock nuscenes module
        nuscenes_mod = types.ModuleType("nuscenes")
        nuscenes_nuscenes = types.ModuleType("nuscenes.nuscenes")
        nuscenes_nuscenes.NuScenes = MagicMock(return_value=_make_mock_nusc())
        sys.modules.setdefault("nuscenes", nuscenes_mod)
        sys.modules["nuscenes.nuscenes"] = nuscenes_nuscenes
        nuscenes_mod.nuscenes = nuscenes_nuscenes

        from dataset_loader import NuScenesLoader
        loader = NuScenesLoader.__new__(NuScenesLoader)
        loader.nusc = _make_mock_nusc()
        loader.dataroot = tmpdir
        loader.version = "v1.0-mini"
        return loader, tmpdir

    def test_get_all_scenes(self):
        loader, _ = self._make_loader()
        scenes = loader.get_all_scenes()
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0]["name"], "scene-0001")

    def test_get_samples_in_scene(self):
        loader, _ = self._make_loader()
        samples = loader.get_samples_in_scene("scene001")
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["token"], "s001")
        self.assertEqual(samples[1]["token"], "s002")

    def test_get_front_camera_data(self):
        loader, _ = self._make_loader()
        sample = {"token": "s001", "data": {"CAM_FRONT": "sd001"}}
        cam_data = loader.get_front_camera_data(sample)
        self.assertIsNotNone(cam_data)
        self.assertEqual(cam_data["token"], "sd001")

    def test_get_front_camera_data_missing_channel(self):
        loader, _ = self._make_loader()
        sample = {"token": "s001", "data": {}}
        cam_data = loader.get_front_camera_data(sample)
        self.assertIsNone(cam_data)

    def test_get_image_path(self):
        loader, tmpdir = self._make_loader()
        cam_data = {"filename": "samples/CAM_FRONT/img001.jpg"}
        path = loader.get_image_path(cam_data)
        expected = Path(tmpdir) / "samples/CAM_FRONT/img001.jpg"
        self.assertEqual(path, expected)

    def test_image_exists_false_for_nonexistent(self):
        loader, tmpdir = self._make_loader()
        cam_data = {"filename": "samples/CAM_FRONT/does_not_exist.jpg"}
        self.assertFalse(loader.image_exists(cam_data))

    def test_get_visibility_level(self):
        loader, _ = self._make_loader()
        ann = {"visibility_token": "3"}
        self.assertEqual(loader.get_visibility_level(ann), 3)

    def test_get_visibility_level_invalid(self):
        loader, _ = self._make_loader()
        ann = {"visibility_token": "x"}
        self.assertEqual(loader.get_visibility_level(ann), 1)

    def test_summarise(self):
        loader, _ = self._make_loader()
        summary = loader.summarise()
        self.assertIn("n_scenes", summary)
        self.assertIn("version", summary)

    def test_get_annotations(self):
        loader, _ = self._make_loader()
        anns = loader.get_annotations("s001")
        self.assertEqual(len(anns), 1)
        self.assertEqual(anns[0]["category_name"], "vehicle.car")

    def test_get_intrinsic_matrix_shape(self):
        loader, _ = self._make_loader()
        cam_data = {"calibrated_sensor_token": "cs001",
                    "ego_pose_token": "ep001"}
        K = loader.get_intrinsic_matrix(cam_data)
        self.assertEqual(K.shape, (3, 3))
        self.assertAlmostEqual(K[0, 0], 1266.4)


# ─────────────────────────────────────────────────────────────────────────────
# SceneSelector tests (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestSceneSelector(unittest.TestCase):
    """Tests for scene_selector using mocked split data."""

    def _make_loader_with_scenes(self, scene_names):
        """Return a loader-like mock with the given scene names."""
        scenes = [
            {"token": f"tok_{n}", "name": n,
             "first_sample_token": f"s_{n}_0", "description": ""}
            for n in scene_names
        ]
        mock = MagicMock()
        mock.version = "v1.0-mini"
        mock.get_all_scenes.return_value = scenes

        def get_scene(token):
            return next(s for s in scenes if s["token"] == token)

        def get_samples(token):
            return [
                {"token": f"{token}_s0", "scene_token": token,
                 "timestamp": 0, "data": {}, "anns": [], "next": "", "prev": ""},
            ]

        mock.get_scene.side_effect = get_scene
        mock.get_samples_in_scene.side_effect = get_samples
        return mock

    def test_select_mini_returns_at_most_max_samples(self):
        scene_names = [f"scene-{i:04d}" for i in range(20)]
        loader = self._make_loader_with_scenes(scene_names)

        with patch("scene_selector._get_official_splits",
                   return_value=(scene_names[:15], scene_names[15:])):
            from scene_selector import select_mini_samples
            samples = select_mini_samples(loader, max_samples=5, seed=0)
        self.assertLessEqual(len(samples), 5)

    def test_select_trainval_split_integrity(self):
        """No scene_token should appear in both train and val."""
        train_names = [f"train-{i}" for i in range(10)]
        val_names = [f"val-{i}" for i in range(5)]
        all_names = train_names + val_names
        loader = self._make_loader_with_scenes(all_names)
        loader.version = "v1.0-trainval"

        with patch("scene_selector._get_official_splits",
                   return_value=(train_names, val_names)):
            from scene_selector import select_trainval_samples
            train, val = select_trainval_samples(
                loader, max_train_scenes=5, max_val_scenes=3, seed=7
            )

        train_scene_tokens = {s["scene_token"] for s in train}
        val_scene_tokens = {s["scene_token"] for s in val}
        overlap = train_scene_tokens & val_scene_tokens
        self.assertEqual(len(overlap), 0, f"Split overlap detected: {overlap}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation tests (no dataset needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluation(unittest.TestCase):
    """Tests for evaluation.py helper functions."""

    def test_text_mentions_car(self):
        from evaluation import _text_mentions_category
        self.assertTrue(
            _text_mentions_category("I see a car approaching.", "vehicle.car")
        )

    def test_text_no_mention(self):
        from evaluation import _text_mentions_category
        self.assertFalse(
            _text_mentions_category("The road is empty.", "vehicle.car")
        )

    def test_all_keywords_in_text(self):
        from evaluation import _all_keywords_in_text
        found = _all_keywords_in_text("A pedestrian crossed near a bicycle.")
        self.assertIn("human.pedestrian.adult", found)
        self.assertIn("vehicle.bicycle", found)

    def test_aggregate_metrics_empty(self):
        from evaluation import aggregate_metrics
        m = aggregate_metrics([])
        self.assertEqual(m["n_evaluated"], 0)

    def test_aggregate_metrics_skipped(self):
        from evaluation import aggregate_metrics
        results = [{"skipped": True, "reason": "no image", "sample_token": "a"}]
        m = aggregate_metrics(results)
        self.assertEqual(m["n_skipped"], 1)
        self.assertEqual(m["n_evaluated"], 0)

    def test_aggregate_metrics_recall(self):
        from evaluation import aggregate_metrics
        # Simulated per-sample results
        per_sample = [
            {
                "skipped": False, "reason": None,
                "sample_token": f"tok{i}",
                "n_focus_gt": 3, "n_focus_mentioned": 2,
                "recall_focus": 2 / 3,
                "n_road_user_gt": 2, "n_road_user_mentioned": 1,
                "recall_road_users": 0.5,
                "vis_gt": {3: 2, 4: 1}, "vis_tp": {3: 1, 4: 1},
                "n_unsupported_mentions": 1,
                "unsupported_categories": ["animal"],
                "latency_sec": 1.0 + i * 0.1,
                "n_tokens": 100,
                "tokens_per_sec": 90.0,
                "device": "cpu",
                "dtype": "float32",
            }
            for i in range(5)
        ]
        m = aggregate_metrics(per_sample)
        self.assertEqual(m["n_evaluated"], 5)
        self.assertAlmostEqual(m["mean_recall_focus_categories"], 2 / 3, places=4)
        self.assertAlmostEqual(m["mean_recall_road_users"], 0.5, places=4)
        self.assertGreater(m["mean_latency_sec"], 0.0)
        self.assertGreater(m["p90_latency_sec"], m["mean_latency_sec"] - 0.01)

    def test_evaluate_sample_skipped_on_error(self):
        from evaluation import evaluate_sample
        pred = {"sample_token": "x", "error": "image not found",
                "generated_description": ""}
        result = evaluate_sample(
            prediction=pred,
            annotations=[],
            loader=MagicMock(),
            focus_categories=["vehicle.car"],
        )
        self.assertTrue(result["skipped"])

    def test_evaluate_sample_no_annotations(self):
        from evaluation import evaluate_sample
        pred = {
            "sample_token": "tok1",
            "error": None,
            "generated_description": "There is a car ahead.",
            "latency_sec": 1.5,
            "n_tokens_generated": 80,
            "tokens_per_sec": 53.3,
            "device": "cpu",
            "dtype": "float32",
        }
        loader = MagicMock()
        loader.get_visibility_level.return_value = 4
        result = evaluate_sample(
            prediction=pred,
            annotations=[],
            loader=loader,
            focus_categories=["vehicle.car"],
        )
        self.assertFalse(result["skipped"])
        self.assertEqual(result["n_focus_gt"], 0)
        # "car" is mentioned but no annotation exists → unsupported mention
        self.assertGreater(result["n_unsupported_mentions"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# ModelLoader tests (no weights needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestModelLoader(unittest.TestCase):
    """Tests for model_loader.py that do not download weights."""

    def _make_cfg(self, model_key="qwen2_5_vl_3b", dtype="float32", device="cpu"):
        return {
            "model": {
                "name": model_key,
                "model_ids": {
                    "qwen2_5_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "smolvlm2_2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                },
                "dtype": dtype,
                "max_new_tokens": 16,
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "device": device,
                "cache_dir": None,
            }
        }

    def test_invalid_model_key(self):
        from model_loader import ModelLoader
        cfg = self._make_cfg(model_key="nonexistent_model")
        # Update model_ids to NOT include that key
        cfg["model"]["model_ids"] = {}
        with self.assertRaises(ValueError):
            ModelLoader(cfg)

    def test_is_loaded_false_before_load(self):
        from model_loader import ModelLoader
        loader = ModelLoader(self._make_cfg())
        self.assertFalse(loader.is_loaded)

    def test_generate_raises_before_load(self):
        from model_loader import ModelLoader
        loader = ModelLoader(self._make_cfg())
        with self.assertRaises(RuntimeError):
            loader.generate("/fake/image.jpg", "Describe this.")

    def test_generate_returns_error_for_missing_image(self):
        """generate() should return error dict (not raise) for missing image."""
        from model_loader import ModelLoader
        loader = ModelLoader(self._make_cfg())
        # Inject a fake loaded state
        loader._model = MagicMock()
        loader._processor = MagicMock()
        result = loader.generate("/nonexistent/image.jpg", "Describe.")
        self.assertIsNotNone(result["error"])
        self.assertIn("not found", result["error"])

    def test_resolve_dtype_bfloat16(self):
        from model_loader import _resolve_dtype
        try:
            import torch
            self.assertEqual(_resolve_dtype("bfloat16"), torch.bfloat16)
        except ImportError:
            self.skipTest("torch not available")

    def test_detect_device_override(self):
        from model_loader import _detect_device
        self.assertEqual(_detect_device("cpu"), "cpu")
        self.assertEqual(_detect_device("cuda:1"), "cuda:1")

    def test_peak_gpu_memory_cpu(self):
        from model_loader import ModelLoader
        loader = ModelLoader(self._make_cfg(device="cpu"))
        # On CPU, peak_gpu_memory_mb should return None or 0
        try:
            import torch
            if not torch.cuda.is_available():
                self.assertIsNone(loader.peak_gpu_memory_mb())
        except ImportError:
            self.assertIsNone(loader.peak_gpu_memory_mb())


if __name__ == "__main__":
    unittest.main(verbosity=2)
