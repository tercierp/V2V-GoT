"""
osm_dataset.py
==============

Glue layer between V2V-GoT data samples (the QA pairs in merge.jsonl) and
the OSM rasterizer. Given a sample dict from the dataset, returns the OSM
BEV tensor for that scene.

This module is meant to be plugged into the V2V-GoT data loader pipeline
(e.g. inside the LLaVA dataset class that currently loads scene + object
features). The OSM tensor is concatenated channel-wise with the existing
scene BEV features before the mm_scene_projector.

Key entry point:
    >>> osm_provider = OSMFeatureProvider(...)
    >>> tensor = osm_provider.get_features(sample)  # sample is a dict from merge.jsonl

The provider is designed to be called many times (one per sample); it caches
OSM queries on disk so repeated scene/timestamp combinations are free after
the first lookup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coord_transform import LocalToWGS84, pose_translation
from osm_query import OSMQuery, Way, synthetic_grid_roads
from osm_rasterizer import OSMRasterizer


def heading_from_pose(pose_4x4_flat: list[float]) -> float:
    """
    Extract yaw (heading angle in radians, CCW from local +x) from a flattened
    4x4 pose matrix.

    The 3x3 rotation block is at indices [0..2, 4..6, 8..10] in the row-major
    layout. We use the standard atan2(R[1,0], R[0,0]) for yaw.
    """
    if len(pose_4x4_flat) != 16:
        raise ValueError(f"Expected 16-element flattened pose, got {len(pose_4x4_flat)}")
    r00 = pose_4x4_flat[0]
    r10 = pose_4x4_flat[4]
    return math.atan2(r10, r00)


@dataclass
class OSMFeatureProvider:
    """
    Loads/caches OSM features for V2V-GoT samples.

    Args:
        coord_transform: LocalToWGS84 instance for the dataset
        cache_dir: where to cache Overpass query results
        rasterizer: pre-configured OSMRasterizer (resolution, FOV, channels)
        radius_m: how far around ego to query OSM
        synthetic_fallback: if True, use synthetic grid roads when no internet
            (useful for first-pass training infrastructure validation)
    """

    coord_transform: LocalToWGS84
    cache_dir: str | Path
    rasterizer: OSMRasterizer | None = None
    radius_m: float = 200.0
    synthetic_fallback: bool = False

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.rasterizer is None:
            self.rasterizer = OSMRasterizer(
                grid_size=200,
                resolution_m=0.5,
                coord_transform=self.coord_transform,
            )
        self.osm_query = OSMQuery(cache_dir=self.cache_dir)
        # In-memory tensor cache: keyed by (scenario_index, global_timestamp_index, asker_cav_id)
        self._tensor_cache: dict[tuple, np.ndarray] = {}

    @property
    def num_channels(self) -> int:
        return self.rasterizer.num_channels

    def get_features(self, sample: dict) -> np.ndarray:
        """
        Compute the OSM BEV tensor for one V2V-GoT sample.

        Args:
            sample: a dict from merge.jsonl, must contain:
                'cav_ego_lidar_pose' or 'cav_1_lidar_pose' (16-element matrix)
                'asker_cav_id' ('ego' or '1')
                'scenario_index', 'global_timestamp_index' (for caching key)

        Returns:
            np.ndarray of shape (C, H, W) float32
        """
        # 1. Determine which CAV is the asker — use that pose
        asker = sample.get("asker_cav_id", "ego")
        pose_key = "cav_ego_lidar_pose" if asker == "ego" else "cav_1_lidar_pose"
        pose = sample[pose_key]

        # 2. Cache key — same scene + timestamp + asker means same OSM tensor
        cache_key = (
            sample.get("scenario_index", -1),
            sample.get("global_timestamp_index", -1),
            asker,
        )
        if cache_key in self._tensor_cache:
            return self._tensor_cache[cache_key]

        # 3. Extract ego pose in local frame
        ego_x, ego_y, _ = pose_translation(pose)
        heading = heading_from_pose(pose)

        # 4. Convert to WGS84 for OSM query
        lat, lon = self.coord_transform.local_to_wgs84(ego_x, ego_y)

        # 5. Fetch OSM ways
        ways = self._fetch_ways(lat, lon)

        # 6. Rasterize
        tensor = self.rasterizer.rasterize(
            ways=ways,
            ego_x_local=ego_x,
            ego_y_local=ego_y,
            ego_heading_rad=heading,
        )

        self._tensor_cache[cache_key] = tensor
        return tensor

    def _fetch_ways(self, lat: float, lon: float) -> list[Way]:
        if self.synthetic_fallback:
            return synthetic_grid_roads(lat, lon, self.radius_m)
        try:
            return self.osm_query.fetch_roads(lat, lon, self.radius_m)
        except Exception as e:
            # Network / API failure — fall back to empty so training doesn't crash
            print(f"[OSMFeatureProvider] Overpass query failed: {e}; returning empty.")
            return []


# ------------------------------------------------------------------
# Smoke test with a real V2V-GoT sample structure
# ------------------------------------------------------------------
if __name__ == "__main__":
    from coord_transform import V2V4REAL_PROVISIONAL_ORIGIN

    # Mock sample mimicking what's in merge.jsonl
    mock_sample = {
        "scenario_index": 0,
        "global_timestamp_index": 0,
        "asker_cav_id": "ego",
        "cav_ego_lidar_pose": [
            -0.8603860139846802, -0.5096380114555359, 0.0020271099638193846, -747.4990234375,
            0.5096420049667358, -0.8603760004043579, 0.004268390126526356, 246.41400146484375,
            -0.0004312550008762628, 0.004705559927970171, 0.9999889731407166, 4.646309852600098,
            0.0, 0.0, 0.0, 1.0,
        ],
    }

    provider = OSMFeatureProvider(
        coord_transform=V2V4REAL_PROVISIONAL_ORIGIN,
        cache_dir="/tmp/osm_cache",
        synthetic_fallback=True,  # No internet needed for this test
    )

    tensor = provider.get_features(mock_sample)
    print(f"OSM tensor shape: {tensor.shape}")
    print(f"Channels: {provider.num_channels}")
    print(f"Drivable mask coverage: {tensor[0].mean():.2%}")
    print(f"Heading extracted from pose: {math.degrees(heading_from_pose(mock_sample['cav_ego_lidar_pose'])):.1f}°")

    # Verify caching: second call should be instant and identical
    tensor2 = provider.get_features(mock_sample)
    assert np.array_equal(tensor, tensor2), "Cache should return identical tensor"
    print("Caching works ✓")

    # Visualize
    from osm_rasterizer import visualize
    visualize(tensor, save_path="/home/claude/osm/_debug_real_sample.png")
