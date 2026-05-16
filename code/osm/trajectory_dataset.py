"""
trajectory_dataset.py
=====================

Drop-in replacement for OSMFeatureProvider that uses trajectory-derived
roads instead of OSM. No coordinate transforms needed — operates entirely
in V2V4Real's local metric frame.

Usage at training/inference time:
    >>> provider = TrajectoryFeatureProvider(
    ...     db_path="trajectory_roads.json",
    ...     grid_size=200,
    ...     resolution_m=0.5,
    ... )
    >>> tensor = provider.get_features(sample)  # (4, 200, 200) float32

For pre-computation (recommended for training):
    >>> provider.precompute_all(
    ...     merge_jsonl_paths=[...],
    ...     output_dir="/scratch/.../osm_features/",
    ... )
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from coord_transform import pose_translation
from osm_dataset import heading_from_pose
from osm_rasterizer import OSMRasterizer
from trajectory_roads import TrajectoryRoadProvider


@dataclass
class TrajectoryFeatureProvider:
    """
    Per-sample BEV tensor provider using trajectory-derived roads.

    Args:
        db_path: path to the JSON DB built by trajectory_roads.build_trajectory_road_db
        grid_size: BEV grid side length in cells (default 200)
        resolution_m: meters per cell (default 0.5 → 100m FOV)
        max_distance_m: clip for the signed-distance channel
        road_width_m: half-width to inflate trajectory polylines
    """

    db_path: str | Path
    grid_size: int = 200
    resolution_m: float = 0.5
    max_distance_m: float = 25.0
    road_width_m: float = 3.5  # Single lane half-width

    def __post_init__(self):
        self.road_provider = TrajectoryRoadProvider(self.db_path)
        # OSMRasterizer requires a coord_transform but we never call it
        # in nodes_in_local_frame mode. Pass a dummy.
        from coord_transform import LocalToWGS84
        dummy_tx = LocalToWGS84(lat0=0.0, lon0=0.0)
        self.rasterizer = OSMRasterizer(
            grid_size=self.grid_size,
            resolution_m=self.resolution_m,
            max_distance_m=self.max_distance_m,
            road_width_m=self.road_width_m,
            coord_transform=dummy_tx,
        )
        # Cache: (scenario, timestamp, asker) -> tensor
        self._cache: dict[tuple, np.ndarray] = {}

    @property
    def num_channels(self) -> int:
        return self.rasterizer.num_channels

    def get_features(self, sample: dict) -> np.ndarray:
        """
        Compute the BEV tensor for one V2V-GoT sample, using trajectory-derived
        roads from the pre-built database.
        """
        asker = sample.get("asker_cav_id", "ego")
        scenario = sample.get("scenario_index", -1)
        timestamp = sample.get("global_timestamp_index", -1)
        cache_key = (scenario, timestamp, asker)

        if cache_key in self._cache:
            return self._cache[cache_key]

        pose_key = "cav_ego_lidar_pose" if asker == "ego" else "cav_1_lidar_pose"
        pose = sample[pose_key]
        ego_x, ego_y, _ = pose_translation(pose)
        heading = heading_from_pose(pose)

        ways = self.road_provider.get_ways(scenario)
        tensor = self.rasterizer.rasterize(
            ways=ways,
            ego_x_local=ego_x,
            ego_y_local=ego_y,
            ego_heading_rad=heading,
            nodes_in_local_frame=True,  # trajectory mode
        )

        self._cache[cache_key] = tensor
        return tensor

    def precompute_all(
        self,
        merge_jsonl_paths: list[str | Path],
        output_dir: str | Path,
        progress_every: int = 1000,
    ) -> None:
        """
        Pre-compute BEV tensors for every sample in the given files and save
        them as .npy files. Critical for training — you don't want to compute
        these on the fly inside the data loader.

        Supports both:
          - .jsonl (line-by-line, V2V-GoT inference outputs)
          - .json  (single top-level array, V2V-GoT-QA training/test)

        Naming convention:
            output_dir/{scenario_index}_{timestamp_index}_{asker_cav_id}.npy

        Note: the same (scenario, timestamp, asker_cav_id) appears many times
        in V2V-GoT-QA (once per QA sub-type), so the actual number of unique
        tensors is much smaller than the raw sample count. We dedupe by file
        existence — first sample wins, subsequent duplicates are skipped.

        Note: V2V-GoT-QA training/test JSONs typically don't have
        'asker_cav_id'. We default to 'ego' in that case (matching the
        OSMFeatureProvider behavior). If your QA pairs distinguish askers
        explicitly, make sure the field is present.
        """
        from trajectory_roads import _iterate_jsonl

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_written = 0
        n_skipped_existing = 0
        n_skipped_invalid = 0
        n_seen = 0
        for _src, sample in _iterate_jsonl(merge_jsonl_paths):
            n_seen += 1
            if n_seen % progress_every == 0:
                print(f"  [progress] seen={n_seen}  written={n_written}  "
                      f"skipped_existing={n_skipped_existing}  "
                      f"skipped_invalid={n_skipped_invalid}", flush=True)

            scenario = sample.get("scenario_index")
            timestamp = sample.get("global_timestamp_index")
            asker = sample.get("asker_cav_id", "ego")
            if scenario is None or timestamp is None:
                n_skipped_invalid += 1
                continue
            # Need the right pose key to be present
            pose_key = "cav_ego_lidar_pose" if asker == "ego" else "cav_1_lidar_pose"
            if pose_key not in sample:
                n_skipped_invalid += 1
                continue

            out_name = f"{scenario}_{timestamp}_{asker}.npy"
            out_path = output_dir / out_name
            if out_path.exists():
                n_skipped_existing += 1
                continue

            try:
                tensor = self.get_features(sample)
            except Exception as e:
                print(f"[precompute_all] Failed on scenario={scenario} ts={timestamp} "
                      f"asker={asker}: {type(e).__name__}: {e}")
                n_skipped_invalid += 1
                continue

            np.save(out_path, tensor)
            n_written += 1

        print(f"[TrajectoryFeatureProvider] Done.")
        print(f"  Total samples scanned:    {n_seen}")
        print(f"  Wrote new tensors:        {n_written}")
        print(f"  Skipped (already cached): {n_skipped_existing}")
        print(f"  Skipped (invalid/error):  {n_skipped_invalid}")


if __name__ == "__main__":
    # Smoke test using the database we'd build from your real merge.jsonl files
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="trajectory_roads.json")
    ap.add_argument("--merge-jsonl", required=True,
                    help="One merge.jsonl to test on")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--output-dir", default="./_traj_debug")
    args = ap.parse_args()

    provider = TrajectoryFeatureProvider(db_path=args.db)
    print(f"Channels: {provider.num_channels}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from osm_rasterizer import visualize
    n = 0
    with open(args.merge_jsonl) as f:
        for line in f:
            if n >= args.num_samples:
                break
            sample = json.loads(line.strip())
            tensor = provider.get_features(sample)
            print(f"Sample {n}: scenario={sample['scenario_index']} "
                  f"timestamp={sample['global_timestamp_index']} "
                  f"asker={sample.get('asker_cav_id', 'ego')} "
                  f"drivable_mask_coverage={tensor[0].mean():.2%}")
            visualize(tensor, save_path=str(output_dir / f"sample_{n}.png"))
            n += 1
