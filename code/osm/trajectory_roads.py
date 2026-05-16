"""
trajectory_roads.py
===================

Build road-network "Way" objects from V2V4Real CAV trajectories instead of
querying OSM. Bypasses the WGS84 coordinate problem entirely — we operate
in the local metric frame the same way the model does.

Core insight: every CAV's trajectory through a scenario IS a road centerline.
With two CAVs per scenario in V2V4Real, we typically get both lanes of the
road covered. This is exactly the signal needed to fix the "wrong-lane"
failure mode reported in the V2V-GoT paper (Figure 31).

Pipeline:
  1. Walk every sample in merge.jsonl files
  2. For each (scenario, cav_id), collect the time-ordered list of poses
  3. Build a polyline per (scenario, cav_id) and emit it as a Way
  4. Cache to disk so we only do this once per dataset split

Usage:
    >>> # One-time pre-computation:
    >>> from trajectory_roads import build_trajectory_road_db
    >>> db = build_trajectory_road_db(
    ...     merge_jsonl_paths=glob("/path/to/eval/*/merge.jsonl"),
    ...     output_path="trajectory_roads.json",
    ... )
    >>>
    >>> # Then at training/inference time:
    >>> from trajectory_roads import TrajectoryRoadProvider
    >>> provider = TrajectoryRoadProvider("trajectory_roads.json")
    >>> ways = provider.get_ways(scenario_index=0)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from coord_transform import pose_translation
from osm_query import Way


# ----------------------------------------------------------------------------
# Trajectory mirroring (synthetic oncoming-lane augmentation)
# ----------------------------------------------------------------------------

def mirror_trajectory(
    points: list[tuple[int, float, float]],
    lane_offset_m: float = 3.5,
    side: str = "left",
    max_turn_deg: float = 30.0,
) -> list[tuple[int, float, float]]:
    """
    Mirror a CAV trajectory perpendicular to its motion direction to synthesize
    an oncoming-lane trajectory.

    For V2V4Real (US, right-hand traffic), the oncoming lane is to the LEFT
    of the direction of motion. For UK/Japan/Australia datasets, set side='right'.

    The mirror uses per-segment perpendiculars with 3-point smoothing
    (previous, current, next) so it handles curves robustly. Output points
    are returned in REVERSED time order, so the mirror represents a vehicle
    moving in the opposite direction at the same timestamps.

    Trajectories with sharp turns (e.g. parking-lot maneuvers) produce
    geometrically invalid mirrors because the perpendicular flips direction
    discontinuously across the turn. We detect this case (any consecutive
    segment changing direction by more than `max_turn_deg`) and refuse to
    mirror such trajectories — returning an empty list, which the caller
    treats as "no mirror for this trajectory."

    Args:
        points: list of (timestamp_index, x, y) tuples in local meters
        lane_offset_m: perpendicular distance to offset the mirror (default 3.5m,
            roughly one US lane width)
        side: 'left' for right-hand traffic (USA, EU), 'right' for UK/JP/AU
        max_turn_deg: refuse to mirror trajectories with any consecutive
            segment-to-segment turn exceeding this many degrees. Default 30°
            filters out parking-lot maneuvers and sharp U-turns while
            preserving normal road curvature (highway/city turns are <20°
            between consecutive points sampled at ~10 Hz).

    Returns:
        list of (timestamp_index, x_mirrored, y_mirrored) tuples in REVERSED
        time order. EMPTY list if the trajectory has sharp turns.

    Raises:
        ValueError: on unknown `side` argument.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if len(points) < 2:
        return []

    # First pass: detect sharp turns and bail out if found
    max_turn_rad = math.radians(max_turn_deg)
    last_dx = last_dy = None
    for i in range(len(points) - 1):
        dx = points[i + 1][1] - points[i][1]
        dy = points[i + 1][2] - points[i][2]
        norm = (dx * dx + dy * dy) ** 0.5
        if norm < 1e-6:
            continue
        dx /= norm
        dy /= norm
        if last_dx is not None:
            # Angle between consecutive unit vectors via atan2 of cross/dot
            cross = last_dx * dy - last_dy * dx
            dot = last_dx * dx + last_dy * dy
            turn_rad = abs(math.atan2(cross, dot))
            if turn_rad > max_turn_rad:
                # Sharp turn detected — refuse to mirror
                return []
        last_dx, last_dy = dx, dy

    sign = 1.0 if side == "left" else -1.0  # left-of-motion vs right-of-motion

    mirrored = []
    n = len(points)
    for i, (ts, x, y) in enumerate(points):
        # Compute local direction using a 3-point window for smoothing
        if i == 0:
            dx = points[1][1] - x
            dy = points[1][2] - y
        elif i == n - 1:
            dx = x - points[i - 1][1]
            dy = y - points[i - 1][2]
        else:
            # Centered difference, more robust to single-point GPS jitter
            dx = points[i + 1][1] - points[i - 1][1]
            dy = points[i + 1][2] - points[i - 1][2]

        norm = (dx * dx + dy * dy) ** 0.5
        if norm < 1e-6:
            # Stationary — skip this point in the mirror
            continue

        # Perpendicular direction (sign chosen by `side`):
        #   "left of motion" (oncoming in right-hand traffic):  (-dy, +dx) / |v|
        #   "right of motion":                                  (+dy, -dx) / |v|
        # Equivalent unified form:
        perp_x = -sign * dy / norm
        perp_y = +sign * dx / norm

        mx = x + lane_offset_m * perp_x
        my = y + lane_offset_m * perp_y
        mirrored.append((ts, mx, my))

    # Reverse so the mirror represents OPPOSITE direction of travel
    return list(reversed(mirrored))


# ----------------------------------------------------------------------------
# Build phase: aggregate poses across all merge.jsonl files
# ----------------------------------------------------------------------------

def _iterate_jsonl(paths: Iterable[Path | str]):
    """Yield (path, sample_dict) for every entry across input files.

    Supports two file formats:
      - .jsonl: one JSON object per line (V2V-GoT inference outputs)
      - .json: a single top-level JSON array of objects (V2V-GoT-QA training/test)

    Format is auto-detected by file extension.
    """
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"[trajectory_roads] WARNING: file does not exist: {p}")
            continue

        if p.suffix.lower() == ".json":
            # Single-file QA dataset: load as a list
            try:
                with open(p) as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[trajectory_roads] failed to parse {p}: {e}")
                continue
            if not isinstance(data, list):
                print(f"[trajectory_roads] {p} is not a list, skipping")
                continue
            for sample in data:
                if isinstance(sample, dict):
                    yield p, sample
        else:
            # Default: line-by-line JSONL (original behavior)
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield p, json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[trajectory_roads] skipping bad line in {p}: {e}")


def build_trajectory_road_db(
    merge_jsonl_paths: list[Path | str],
    output_path: Path | str,
    min_segment_length_m: float = 5.0,
    deduplicate_threshold_m: float = 0.5,
    add_mirror_lane: bool = False,
    mirror_offset_m: float = 3.5,
    mirror_side: str = "left",
) -> dict:
    """
    Walk every merge.jsonl file once, group poses by (scenario, cav_id, timestamp),
    and write out a JSON database of trajectories suitable for the rasterizer.

    The output JSON is keyed by scenario_index (str) and contains a list of
    polylines (one per CAV in that scenario). Each polyline is a list of
    (timestamp, x, y) triples sorted by timestamp.

    Args:
        merge_jsonl_paths: list of merge.jsonl files to scan. For V2V-GoT,
            you'd pass all 9 inference output files plus optionally the
            training set's merge.jsonl if available.
        output_path: where to write the database JSON.
        min_segment_length_m: drop trajectories shorter than this (artifacts).
        deduplicate_threshold_m: collapse consecutive poses closer than this
            into a single point (poses repeat across QA pairs at the same
            timestamp).
        add_mirror_lane: if True, for each real CAV trajectory also synthesize
            a mirrored polyline representing an oncoming-lane vehicle. Useful
            when the dataset only contains convoy-style trajectories (both
            CAVs going the same direction) so the model never sees opposite-
            lane structure during training.
        mirror_offset_m: perpendicular distance for the mirror (default 3.5m,
            roughly one US lane width).
        mirror_side: 'left' for right-hand traffic (USA), 'right' for UK/JP/AU.

    Returns:
        The database dict (also written to disk).
    """
    # Nested defaultdict: {scenario_index: {cav_id: {timestamp_index: (x, y)}}}
    raw = defaultdict(lambda: defaultdict(dict))

    n_samples = 0
    n_poses_extracted = 0
    for jsonl_path, sample in _iterate_jsonl(merge_jsonl_paths):
        n_samples += 1
        scenario = sample.get("scenario_index")
        timestamp = sample.get("global_timestamp_index")
        if scenario is None or timestamp is None:
            continue

        # Both CAVs' poses are stored on every sample — capture both
        for pose_key, cav_id in [("cav_ego_lidar_pose", "ego"),
                                 ("cav_1_lidar_pose", "1")]:
            pose = sample.get(pose_key)
            if pose is None or len(pose) != 16:
                continue
            x, y, _ = pose_translation(pose)
            # First write wins (deterministic dedup across QA types at same TS)
            raw[scenario][cav_id].setdefault(timestamp, (x, y))
            n_poses_extracted += 1

    print(f"[trajectory_roads] Scanned {n_samples} samples, "
          f"extracted {n_poses_extracted} poses across "
          f"{len(raw)} scenarios.")

    # Now build polylines per (scenario, cav_id)
    db = {}
    n_polylines = 0
    n_dropped_short = 0
    for scenario, cavs in raw.items():
        scenario_polylines = []
        for cav_id, ts_to_xy in cavs.items():
            # Sort by timestamp
            sorted_pts = [(ts, x, y) for ts, (x, y) in sorted(ts_to_xy.items())]
            if len(sorted_pts) < 2:
                continue

            # Deduplicate consecutive close points
            cleaned = [sorted_pts[0]]
            for ts, x, y in sorted_pts[1:]:
                _, lx, ly = cleaned[-1]
                if (x - lx) ** 2 + (y - ly) ** 2 >= deduplicate_threshold_m ** 2:
                    cleaned.append((ts, x, y))

            if len(cleaned) < 2:
                continue

            # Compute total trajectory length and drop short ones
            total_length = 0.0
            for (_, x0, y0), (_, x1, y1) in zip(cleaned[:-1], cleaned[1:]):
                total_length += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            if total_length < min_segment_length_m:
                n_dropped_short += 1
                continue

            scenario_polylines.append({
                "cav_id": cav_id,
                "length_m": total_length,
                "n_points": len(cleaned),
                "points": [[ts, round(x, 3), round(y, 3)] for ts, x, y in cleaned],
                "is_mirror": False,
            })
            n_polylines += 1

            # Synthesize an oncoming-lane mirror, if requested
            if add_mirror_lane:
                mirrored = mirror_trajectory(
                    cleaned,
                    lane_offset_m=mirror_offset_m,
                    side=mirror_side,
                )
                if len(mirrored) >= 2:
                    # Recompute length on the mirrored points
                    mlen = 0.0
                    for (_, x0, y0), (_, x1, y1) in zip(mirrored[:-1], mirrored[1:]):
                        mlen += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                    scenario_polylines.append({
                        "cav_id": f"{cav_id}_mirror",
                        "length_m": mlen,
                        "n_points": len(mirrored),
                        "points": [[ts, round(x, 3), round(y, 3)] for ts, x, y in mirrored],
                        "is_mirror": True,
                    })
                    n_polylines += 1

        if scenario_polylines:
            db[str(scenario)] = scenario_polylines

    print(f"[trajectory_roads] Built {n_polylines} polylines, "
          f"dropped {n_dropped_short} short trajectories.")

    # Write to disk
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(db, f, indent=2)
    print(f"[trajectory_roads] Wrote database to {output_path}")

    return db


# ----------------------------------------------------------------------------
# Provider: load the pre-built DB and serve Way objects
# ----------------------------------------------------------------------------

@dataclass
class TrajectoryRoadProvider:
    """
    Serves Way objects derived from CAV trajectories for the rasterizer.

    Drop-in replacement for OSMQuery.fetch_roads(). Use this when:
      - You don't have the WGS84 origin
      - You want offline/deterministic road geometry
      - You want roads that the actual CAVs drove on (trajectory-grounded)
    """

    db_path: str | Path
    db: dict = field(default_factory=dict)

    def __post_init__(self):
        with open(self.db_path) as f:
            self.db = json.load(f)
        n_scenarios = len(self.db)
        n_polylines = sum(len(v) for v in self.db.values())
        print(f"[TrajectoryRoadProvider] Loaded {n_scenarios} scenarios, "
              f"{n_polylines} polylines from {self.db_path}")

    def get_ways(self, scenario_index: int) -> list[Way]:
        """
        Return a list of Way objects for the given scenario, with coordinates
        in the local metric frame (NOT WGS84) — but the rasterizer treats
        node coordinates as opaque pairs anyway.

        We fake the lat/lon fields by storing local (x, y) there. This works
        because the rasterizer's only use of coord_transform is wgs84_to_local,
        which the trajectory-aware rasterizer (next file) bypasses.
        """
        polylines = self.db.get(str(scenario_index), [])
        ways = []
        for i, poly in enumerate(polylines):
            # Store local (x, y) directly in the nodes field (not lat/lon)
            nodes_local = [(x, y) for _, x, y in poly["points"]]
            ways.append(Way(
                osm_id=hash((scenario_index, poly["cav_id"])) & 0xFFFFFFFF,
                nodes=nodes_local,  # local meters, NOT lat/lon
                highway="primary",  # treat trajectories as primary roads
                oneway=True,        # trajectories are directional
                lanes=1,
                name=f"scenario_{scenario_index}_cav_{poly['cav_id']}",
                raw_tags={"source": "trajectory", "length_m": poly["length_m"]},
            ))
        return ways

    def all_scenarios(self) -> list[int]:
        return sorted(int(k) for k in self.db.keys())

    def stats(self) -> dict:
        return {
            "n_scenarios": len(self.db),
            "n_polylines": sum(len(v) for v in self.db.values()),
            "polylines_per_scenario": {
                k: len(v) for k, v in self.db.items()
            },
            "total_length_m": sum(
                p["length_m"] for v in self.db.values() for p in v
            ),
        }


# ----------------------------------------------------------------------------
# CLI for one-shot DB building
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import glob

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--merge-glob",
        action="append",
        default=None,
        help="Glob pattern matching merge.jsonl or .json files. Can be passed "
             "multiple times to union several globs (e.g. train + test).",
    )
    ap.add_argument(
        "--output",
        default="trajectory_roads.json",
        help="Where to write the trajectory road DB",
    )
    ap.add_argument(
        "--add-mirror-lane",
        action="store_true",
        help="Synthesize a mirrored oncoming-lane polyline for each real CAV "
             "trajectory (useful when the dataset only has convoy-style trajectories)",
    )
    ap.add_argument(
        "--mirror-offset-m",
        type=float,
        default=3.5,
        help="Perpendicular distance for mirror lane (default 3.5m, one US lane)",
    )
    ap.add_argument(
        "--mirror-side",
        choices=["left", "right"],
        default="left",
        help="Side of motion to mirror to ('left' for right-hand traffic / USA, "
             "'right' for UK/JP/AU)",
    )
    args = ap.parse_args()

    # Default to test-split inference outputs if no glob given
    if not args.merge_glob:
        args.merge_glob = [
            "/scratch/izar/tercier/v2v-got/V2V-GoT/LLaVA/playground/data/eval/"
            "v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2_4330_full_nq*/"
            "answers/val/llava-v1.5-7b/merge.jsonl",
        ]

    paths = []
    for g in args.merge_glob:
        paths.extend(sorted(glob.glob(g)))
    paths = sorted(set(paths))  # dedup if globs overlap
    print(f"Found {len(paths)} input files")
    for p in paths:
        print(f"  {p}")

    db = build_trajectory_road_db(
        merge_jsonl_paths=paths,
        output_path=args.output,
        add_mirror_lane=args.add_mirror_lane,
        mirror_offset_m=args.mirror_offset_m,
        mirror_side=args.mirror_side,
    )

    provider = TrajectoryRoadProvider(args.output)
    print("\nStats:", json.dumps(provider.stats(), indent=2))
