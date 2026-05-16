"""
compute_lane_compliance.py
==========================

Lane Compliance Rate (LCR) metric comparing OSM-Q9 vs baseline Q9 trajectories.

Strategy
--------
Road centerlines are reconstructed from the actual CAV poses stored in every
merge.jsonl sample (both ego and CAV-1 poses are on each row).  Since the CAVs
drove on real roads, their observed trajectories ARE the road centerlines.
We collect one global-frame polyline per (scenario, cav_id) and use those as
the ground-truth lane geometry — no external map data needed.

For each Q9 sample we:
  1. Parse the predicted trajectory from the model output (ego-relative coords)
  2. Transform waypoints to the global frame using cav_ego_lidar_pose
  3. Compute minimum lateral distance from each waypoint to the nearest road polyline
  4. Mark as in-lane if distance < HALF_LANE_WIDTH_M (default 1.75 m)

Outputs a summary table + per-scenario breakdown.

Usage
-----
    cd /scratch/izar/tercier/v2v-got/V2V-GoT/code/osm
    python compute_lane_compliance.py
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
EVAL = Path("/scratch/izar/tercier/v2v-got/V2V-GoT/LLaVA/playground/data/eval")

CONFIGS = {
    "baseline": "v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2_4330_full_nq9sm3w6dc",
    "osm_q9":   "v2v4real_3d_grounding_osm_from_q9_2500_full_nq9sm3w6dc",
}
SPLIT = "val"
MODEL = "llava-v1.5-7b"

# Road geometry built from Q1 outputs (dense coverage, all scenarios)
ROAD_SOURCE_TAG = "v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2_4330_full_nq1sm3w0d"

HALF_LANE_WIDTH_M = 1.75   # half of a standard 3.5 m US lane
MIN_ROAD_POINTS   = 4      # minimum poses to consider a polyline a valid road

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def pose_translation(flat: list[float]) -> tuple[float, float]:
    """Extract (x, y) global position from a flat 4×4 row-major pose matrix."""
    return flat[3], flat[7]


def pose_rotation_matrix(flat: list[float]) -> np.ndarray:
    """Extract the 2×2 top-left rotation block from a flat 4×4 pose."""
    return np.array([[flat[0], flat[1]],
                     [flat[4], flat[5]]])


def parse_traj(text: str) -> list[tuple[float, float]]:
    """Extract all (x, y) waypoint pairs from model output text."""
    pairs = re.findall(r'\((-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\)', text)
    return [(float(x), float(y)) for x, y in pairs]


def ego_to_global(
    waypoints: list[tuple[float, float]],
    pose_flat: list[float],
) -> list[tuple[float, float]]:
    """
    Transform ego-relative (forward, lateral) waypoints to global (x, y).

    Ego-relative convention in V2V-GoT:
      - first component  = forward distance (along heading)
      - second component = lateral offset   (left = positive)

    The pose matrix maps local ego frame → global frame.
    """
    R = pose_rotation_matrix(pose_flat)
    tx, ty = pose_translation(pose_flat)
    result = []
    for fwd, lat in waypoints:
        # Ego-frame vector: (fwd along heading, lat perpendicular)
        local = np.array([fwd, lat])
        gx, gy = R @ local
        result.append((tx + gx, ty + gy))
    return result


def point_to_segment_dist(px, py, ax, ay, bx, by) -> float:
    """Minimum distance from point P to segment AB."""
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-10:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def min_dist_to_polyline(px, py, polyline: list[tuple[float, float]]) -> float:
    """Minimum distance from (px, py) to any segment of a polyline."""
    best = math.inf
    for (ax, ay), (bx, by) in zip(polyline[:-1], polyline[1:]):
        d = point_to_segment_dist(px, py, ax, ay, bx, by)
        if d < best:
            best = d
    return best


def min_dist_to_road(px, py, roads: list[list[tuple[float, float]]]) -> float:
    """Minimum distance from (px, py) to the nearest road polyline."""
    best = math.inf
    for poly in roads:
        d = min_dist_to_polyline(px, py, poly)
        if d < best:
            best = d
    return best


# ── Step 1: Build road geometry from CAV poses ────────────────────────────────

def build_road_db(source_tag: str) -> dict[int, list[list[tuple[float, float]]]]:
    """
    Collect global-frame poses for every (scenario, cav_id) from the Q1 merge.jsonl
    and return per-scenario road polylines.
    """
    path = EVAL / source_tag / "answers" / SPLIT / MODEL / "merge.jsonl"
    samples = load_jsonl(path)

    # {scenario: {cav_id: {global_ts: (x, y)}}}
    raw: dict = defaultdict(lambda: defaultdict(dict))
    for s in samples:
        scenario = s.get("scenario_index")
        ts       = s.get("global_timestamp_index")
        if scenario is None or ts is None:
            continue
        for key, cav_id in [("cav_ego_lidar_pose", "ego"), ("cav_1_lidar_pose", "1")]:
            pose = s.get(key)
            if pose and len(pose) == 16:
                x, y = pose_translation(pose)
                raw[scenario][cav_id].setdefault(ts, (x, y))

    roads: dict[int, list[list[tuple[float, float]]]] = {}
    for scenario, cavs in raw.items():
        polys = []
        for cav_id, ts_map in cavs.items():
            pts = [xy for _, xy in sorted(ts_map.items())]
            if len(pts) >= MIN_ROAD_POINTS:
                polys.append(pts)
        if polys:
            roads[scenario] = polys

    print(f"Road DB: {len(roads)} scenarios, "
          f"{sum(len(v) for v in roads.values())} polylines")
    return roads


# ── Step 2: Compute LCR for one config ───────────────────────────────────────

def compute_lcr(
    tag: str,
    roads: dict[int, list[list[tuple[float, float]]]],
    half_lane: float = HALF_LANE_WIDTH_M,
) -> dict:
    path = EVAL / tag / "answers" / SPLIT / MODEL / "merge.jsonl"
    samples = load_jsonl(path)

    total_wpts      = 0
    in_lane_wpts    = 0
    total_lat_err   = 0.0
    off_road_frames = 0
    no_road_frames  = 0
    no_traj_frames  = 0

    per_scenario: dict[int, dict] = defaultdict(lambda: {"in": 0, "total": 0})

    for s in samples:
        scenario = s.get("scenario_index")
        pose     = s.get("cav_ego_lidar_pose")
        traj_raw = parse_traj(s.get("outputs", ""))

        if not traj_raw:
            no_traj_frames += 1
            continue
        if pose is None or len(pose) != 16:
            continue

        road_polys = roads.get(scenario)
        if not road_polys:
            no_road_frames += 1
            continue

        global_wpts = ego_to_global(traj_raw, pose)

        frame_in_lane = True
        for gx, gy in global_wpts:
            d = min_dist_to_road(gx, gy, road_polys)
            total_lat_err += d
            total_wpts    += 1
            if d <= half_lane:
                in_lane_wpts += 1
                per_scenario[scenario]["in"] += 1
            else:
                frame_in_lane = False
            per_scenario[scenario]["total"] += 1

        if not frame_in_lane:
            off_road_frames += 1

    lcr = in_lane_wpts / total_wpts if total_wpts else 0.0
    mean_lat = total_lat_err / total_wpts if total_wpts else 0.0
    off_road_pct = off_road_frames / (len(samples) - no_traj_frames - no_road_frames + 1e-9)

    return {
        "lcr":            lcr,
        "mean_lat_err_m": mean_lat,
        "off_road_pct":   off_road_pct,
        "total_wpts":     total_wpts,
        "in_lane_wpts":   in_lane_wpts,
        "no_traj":        no_traj_frames,
        "no_road":        no_road_frames,
        "per_scenario":   dict(per_scenario),
    }


# ── Step 3: Print results ──────────────────────────────────────────────────────

def main():
    print("Building road geometry from CAV poses …")
    roads = build_road_db(ROAD_SOURCE_TAG)

    results = {}
    for name, tag in CONFIGS.items():
        print(f"\nComputing LCR for [{name}] …")
        results[name] = compute_lcr(tag, roads)

    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Config':<18} {'LCR':>8} {'Mean lat err':>14} {'Off-road frames':>16}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<18} {r['lcr']:>7.1%} {r['mean_lat_err_m']:>13.2f}m {r['off_road_pct']:>15.1%}")
    print("=" * 65)

    # Per-scenario breakdown
    all_scenarios = sorted(set(
        s for r in results.values() for s in r["per_scenario"]
    ))
    if all_scenarios:
        print(f"\nPer-scenario LCR (scenario | {'  |  '.join(results.keys())})")
        print("-" * 55)
        for sc in all_scenarios:
            row = f"  scenario {sc:>3} |"
            for name, r in results.items():
                ps = r["per_scenario"].get(sc, {"in": 0, "total": 1})
                lcr_sc = ps["in"] / ps["total"] if ps["total"] else 0
                row += f"  {lcr_sc:>6.1%}  |"
            print(row)

    # Delta
    if "baseline" in results and "osm_q9" in results:
        delta_lcr = results["osm_q9"]["lcr"] - results["baseline"]["lcr"]
        delta_lat = results["osm_q9"]["mean_lat_err_m"] - results["baseline"]["mean_lat_err_m"]
        print(f"\nOSM-Q9 vs baseline:")
        print(f"  ΔLCR            = {delta_lcr:+.1%}")
        print(f"  Δmean lat err   = {delta_lat:+.2f} m  ({'better' if delta_lat < 0 else 'worse'})")


if __name__ == "__main__":
    main()
