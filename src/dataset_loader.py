"""
dataset_loader.py — nuScenes dataset interface for Phase 1.

Wraps the official nuScenes devkit to provide:
  - NuScenes object initialisation with validation.
  - Scene, sample, and sample_data record access.
  - CAM_FRONT image retrieval and validation.
  - Calibration record access for 3-D → 2-D projection.
  - Annotation retrieval per sample.
  - Ego-pose access.

Design decisions
----------------
* One NuScenesLoader per dataset version (mini or trainval).
* The devkit's ``NuScenes`` object is used directly; no parallel loading.
* Images are NOT buffered in memory; only paths are returned.
* ``sample_annotation`` tokens are resolved lazily (only when needed).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NuScenesLoader:
    """Thin wrapper around the nuScenes devkit.

    Args:
        dataroot: Root directory of the nuScenes dataset.
        version: Dataset version string, e.g. ``"v1.0-mini"`` or
            ``"v1.0-trainval"``.
        verbose: Whether to print devkit progress bars.

    Raises:
        ImportError: If ``nuscenes`` package is not installed.
        FileNotFoundError: If *dataroot* does not exist or the metadata
            JSON files are missing.
    """

    # nuScenes camera channel used for Phase 1
    FRONT_CAM = "CAM_FRONT"

    # nuScenes modalities available in the dataset
    ALL_CAMERAS = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        verbose: bool = False,
    ) -> None:
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise ImportError(
                "nuScenes devkit not installed.  "
                "Run: pip install nuscenes-devkit"
            ) from exc

        dataroot = str(Path(dataroot).resolve())
        if not Path(dataroot).is_dir():
            raise FileNotFoundError(
                f"nuScenes dataroot does not exist: {dataroot}"
            )

        metadata_dir = Path(dataroot) / version
        if not metadata_dir.is_dir():
            raise FileNotFoundError(
                f"nuScenes metadata directory not found: {metadata_dir}\n"
                f"Expected layout: <dataroot>/{version}/*.json"
            )

        logger.info("Loading nuScenes %s from %s …", version, dataroot)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        self.dataroot = dataroot
        self.version = version
        logger.info(
            "Loaded %d scenes, %d samples.",
            len(self.nusc.scene),
            len(self.nusc.sample),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Scene access
    # ─────────────────────────────────────────────────────────────────────────

    def get_all_scenes(self) -> List[Dict[str, Any]]:
        """Return all scene records in the dataset version."""
        return list(self.nusc.scene)

    def get_scene(self, scene_token: str) -> Dict[str, Any]:
        """Return a single scene record by token."""
        return self.nusc.get("scene", scene_token)

    # ─────────────────────────────────────────────────────────────────────────
    # Sample access
    # ─────────────────────────────────────────────────────────────────────────

    def get_samples_in_scene(self, scene_token: str) -> List[Dict[str, Any]]:
        """Return all annotated sample records belonging to a scene, ordered
        chronologically.

        Args:
            scene_token: Token of the target scene.

        Returns:
            List of sample records sorted by timestamp.
        """
        scene = self.get_scene(scene_token)
        samples: List[Dict[str, Any]] = []
        current_token = scene["first_sample_token"]
        while current_token:
            sample = self.nusc.get("sample", current_token)
            samples.append(sample)
            current_token = sample["next"]
        return samples

    def get_sample(self, sample_token: str) -> Dict[str, Any]:
        """Return a single sample record by token."""
        return self.nusc.get("sample", sample_token)

    # ─────────────────────────────────────────────────────────────────────────
    # Camera data
    # ─────────────────────────────────────────────────────────────────────────

    def get_front_camera_data(
        self, sample: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return the CAM_FRONT sample_data record for a given sample.

        Args:
            sample: A sample record dictionary.

        Returns:
            sample_data record or None if the channel is missing.
        """
        cam_token = sample["data"].get(self.FRONT_CAM)
        if cam_token is None:
            logger.warning(
                "Sample %s has no %s channel.", sample["token"], self.FRONT_CAM
            )
            return None
        return self.nusc.get("sample_data", cam_token)

    def get_image_path(self, sample_data: Dict[str, Any]) -> Path:
        """Return the absolute path to the image file.

        Args:
            sample_data: A sample_data record (must have ``filename`` key).

        Returns:
            Absolute Path to the image file.
        """
        return Path(self.dataroot) / sample_data["filename"]

    def image_exists(self, sample_data: Dict[str, Any]) -> bool:
        """Return True if the image file exists on disk."""
        return self.get_image_path(sample_data).is_file()

    # ─────────────────────────────────────────────────────────────────────────
    # Annotations
    # ─────────────────────────────────────────────────────────────────────────

    def get_annotations(self, sample_token: str) -> List[Dict[str, Any]]:
        """Return all sample_annotation records for a sample.

        Each annotation record includes:
        - ``category_name``: Human-readable category string.
        - ``visibility_token``: Visibility level ('1' through '4').
        - ``num_lidar_pts``: Points from LiDAR scan.
        - ``num_radar_pts``: Points from radar scan.
        - ``translation``: 3-D centre in ego-vehicle frame.
        - ``size``: Bounding box [W, L, H] in metres.

        Args:
            sample_token: Token of the annotated sample.

        Returns:
            List of annotation records.
        """
        sample = self.get_sample(sample_token)
        annotations: List[Dict[str, Any]] = []
        for ann_token in sample["anns"]:
            ann = dict(self.nusc.get("sample_annotation", ann_token))
            if "category_name" not in ann:
                if "category_token" in ann:
                    ann["category_name"] = self.nusc.get("category", ann["category_token"])["name"]
                elif "instance_token" in ann:
                    inst = self.nusc.get("instance", ann["instance_token"])
                    ann["category_name"] = self.nusc.get("category", inst["category_token"])["name"]
                else:
                    ann["category_name"] = "unknown"
            annotations.append(ann)
        return annotations

    def get_visibility_level(self, annotation: Dict[str, Any]) -> int:
        """Return the integer visibility level (1–4) for an annotation.

        nuScenes visibility tokens are '1', '2', '3', '4' where:
          1 = 0–40 % visible, 2 = 40–60 %, 3 = 60–80 %, 4 = 80–100 %.

        Args:
            annotation: A sample_annotation record.

        Returns:
            Integer in range [1, 4].
        """
        try:
            return int(annotation.get("visibility_token", "1"))
        except ValueError:
            return 1

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration & ego pose
    # ─────────────────────────────────────────────────────────────────────────

    def get_calibrated_sensor(
        self, sample_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return the calibrated_sensor record for a sample_data entry.

        Contains:
        - ``translation``: Sensor position in vehicle frame.
        - ``rotation``: Quaternion [w, x, y, z].
        - ``camera_intrinsic``: 3×3 intrinsic matrix (cameras only).

        Args:
            sample_data: A sample_data record.

        Returns:
            calibrated_sensor record.
        """
        return self.nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )

    def get_ego_pose(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return the ego_pose record at the time of a sample_data capture.

        Contains global pose of the ego vehicle:
        - ``translation``: [x, y, z] in global frame.
        - ``rotation``: Quaternion [w, x, y, z].

        Args:
            sample_data: A sample_data record.

        Returns:
            ego_pose record.
        """
        return self.nusc.get("ego_pose", sample_data["ego_pose_token"])

    def get_intrinsic_matrix(
        self, sample_data: Dict[str, Any]
    ) -> np.ndarray:
        """Return the 3×3 camera intrinsic matrix for a sample_data record.

        Args:
            sample_data: A sample_data record from a camera channel.

        Returns:
            NumPy array of shape (3, 3).

        Raises:
            KeyError: If the record has no intrinsic (e.g. LiDAR).
        """
        cs = self.get_calibrated_sensor(sample_data)
        K = cs.get("camera_intrinsic")
        if K is None:
            raise KeyError(
                f"sample_data {sample_data['token']} has no camera_intrinsic "
                "(may be a non-camera channel)."
            )
        return np.array(K, dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────────────
    # Category utilities
    # ─────────────────────────────────────────────────────────────────────────

    def get_all_categories(self) -> List[Dict[str, Any]]:
        """Return all category records defined in the dataset."""
        return list(self.nusc.category)

    def get_category_name(self, category_token: str) -> str:
        """Return the category name string for a given token."""
        return self.nusc.get("category", category_token)["name"]

    # ─────────────────────────────────────────────────────────────────────────
    # 3-D → 2-D projection helper
    # ─────────────────────────────────────────────────────────────────────────

    def project_box_to_image(
        self,
        annotation: Dict[str, Any],
        sample_data: Dict[str, Any],
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Project the 3-D annotation centre into the image coordinate system.

        Uses the nuScenes ``Box`` API for correct coordinate transformations
        (global → ego → camera → image).

        Args:
            annotation: sample_annotation record.
            sample_data: CAM_FRONT sample_data record.

        Returns:
            Tuple of:
              - ``uv``: (2,) array of pixel coordinates [u, v].
              - ``depth``: Depth in metres from the camera origin.
            or None if the point is behind the camera or outside the image.
        """
        try:
            from nuscenes.utils.geometry_utils import view_points
            from pyquaternion import Quaternion

            img_w = sample_data.get("width", 1600)
            img_h = sample_data.get("height", 900)

            # Ego pose at this sample_data timestamp
            ego_pose = self.get_ego_pose(sample_data)
            # Calibrated sensor for the camera
            cs = self.get_calibrated_sensor(sample_data)

            # Global position of the annotation
            global_xyz = np.array(annotation["translation"])  # (3,)

            # 1. Global → ego frame
            ego_t = np.array(ego_pose["translation"])
            ego_r = Quaternion(ego_pose["rotation"]).inverse
            ego_xyz = ego_r.rotate(global_xyz - ego_t)

            # 2. Ego → sensor frame
            sensor_t = np.array(cs["translation"])
            sensor_r = Quaternion(cs["rotation"]).inverse
            sensor_xyz = sensor_r.rotate(ego_xyz - sensor_t)

            depth = sensor_xyz[2]
            if depth <= 0.0:
                return None  # behind the camera

            # 3. Sensor → image plane
            K = self.get_intrinsic_matrix(sample_data)
            uvd = K @ sensor_xyz
            u = uvd[0] / uvd[2]
            v = uvd[1] / uvd[2]

            if not (0 <= u < img_w and 0 <= v < img_h):
                return None  # outside image boundaries

            return np.array([u, v]), depth

        except Exception as exc:
            logger.debug("Projection failed for annotation %s: %s", annotation.get("token"), exc)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────

    def summarise(self) -> Dict[str, Any]:
        """Return a brief summary dict of the loaded dataset."""
        return {
            "version": self.version,
            "dataroot": self.dataroot,
            "n_scenes": len(self.nusc.scene),
            "n_samples": len(self.nusc.sample),
            "n_sample_data": len(self.nusc.sample_data),
            "n_categories": len(self.nusc.category),
        }
