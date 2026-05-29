#!/usr/bin/env python3
"""
Phase 3 evaluation: apply the conflict resolver to Phase 0 per-frame data.

Reads the per_frame.json produced by role_swap_eval.py, applies the resolver to
every frame, and reports before/after conflict statistics.

Usage:
    python scripts/run_resolver_eval.py \
        --per_frame outputs/role_swap/per_frame.json \
        --output_dir outputs/resolver_eval
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from v2vgotd.resolver import (
    EgoDecision, OSMContext, FinalDecision,
    resolve, _trajectory_conflict, DEFAULT_SAFETY_M, SPEED_CLASSES, STEERING_CLASSES,
)


def load_per_frame(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def decision_from_frame(frame: dict, cav_id: str) -> EgoDecision:
    """Reconstruct an EgoDecision from a per_frame.json record."""
    gt = frame['gt']
    if cav_id == 'ego':
        speed_str    = gt['ego_speed']
        steering_str = gt['ego_steer']
        traj         = gt['ego_traj']
    else:
        speed_str    = gt['cav1_speed']
        steering_str = gt['cav1_steer']
        traj         = gt['cav1_traj']
    speed_idx    = SPEED_CLASSES.index(speed_str)    if speed_str    in SPEED_CLASSES    else 0
    steering_idx = STEERING_CLASSES.index(steering_str) if steering_str in STEERING_CLASSES else 2
    return EgoDecision(
        cav_id=cav_id,
        speed_idx=speed_idx,
        steering_idx=steering_idx,
        trajectory=traj,
    )


def eval_frame(frame: dict, safety_m: float) -> dict:
    """Apply resolver to one frame; return metrics dict."""
    gt = frame['gt']

    ego_dec  = decision_from_frame(frame, 'ego')
    cav1_dec = decision_from_frame(frame, '1')

    # Before: raw conflict from Phase 0
    conflict_before  = gt.get('conflict', False)
    min_dist_before  = gt.get('traj_min_dist_m')

    # After: apply resolver (no OSM context for this baseline run)
    result = resolve(ego_dec, cav1_dec, osm=None, safety_threshold_m=safety_m)

    # Recompute conflict with the resolved speeds.
    # We scale the trajectory by speed ratio as a simple proxy:
    # if the yielding CAV slows from speed_idx i to i+1, its trajectory
    # endpoints shrink by the ratio of distances per step.
    _SPEED_MPS = [8.0, 5.5, 3.5, 1.5, 0.0]   # rough m/0.5s per speed class

    def _scale_traj(traj, old_idx, new_idx):
        if not traj or old_idx == new_idx:
            return traj
        old_spd = _SPEED_MPS[old_idx]
        new_spd = _SPEED_MPS[new_idx]
        if old_spd == 0:
            return traj
        ratio = new_spd / old_spd
        return [(x * ratio, z * ratio) for (x, z) in traj]

    orig_ego_idx  = SPEED_CLASSES.index(gt['ego_speed'])
    orig_cav1_idx = SPEED_CLASSES.index(gt['cav1_speed'])

    resolved_ego_traj  = _scale_traj(ego_dec.trajectory,
                                     orig_ego_idx, result.ego_speed_idx)
    resolved_cav1_traj = _scale_traj(cav1_dec.trajectory,
                                     orig_cav1_idx, result.cav1_speed_idx)

    conflict_after, min_dist_after, _ = _trajectory_conflict(
        resolved_ego_traj, resolved_cav1_traj, threshold_m=safety_m)

    return {
        'global_timestamp_index': frame['global_timestamp_index'],
        'scenario_index':         frame['scenario_index'],
        'conflict_before':        conflict_before,
        'conflict_after':         conflict_after,
        'min_dist_before_m':      min_dist_before,
        'min_dist_after_m':       round(min_dist_after, 3)
                                  if min_dist_after != float('inf') else None,
        'rule_applied':           result.rule_applied,
        'yielding_cav':           result.yielding_cav,
        'action_agree_before':    gt.get('action_agreement'),
        'ego_speed_before':       gt['ego_speed'],
        'cav1_speed_before':      gt['cav1_speed'],
        'ego_speed_after':        result.ego_speed_str,
        'cav1_speed_after':       result.cav1_speed_str,
    }


def aggregate(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}

    def pct(lst, val):
        return round(100 * sum(1 for x in lst if x == val) / len(lst), 1) if lst else None

    cb = [r['conflict_before'] for r in records]
    ca = [r['conflict_after']  for r in records]
    rules = [r['rule_applied'] for r in records]
    yields = [r['yielding_cav'] for r in records if r['yielding_cav']]

    db = [r['min_dist_before_m'] for r in records if r['min_dist_before_m'] is not None]
    da = [r['min_dist_after_m']  for r in records if r['min_dist_after_m']  is not None]

    n_conflict_before = sum(cb)
    n_conflict_after  = sum(ca)
    resolved = sum(1 for b, a in zip(cb, ca) if b and not a)

    return {
        'n_frames':               n,
        'conflict_before_pct':    pct(cb, True),
        'conflict_after_pct':     pct(ca, True),
        'conflicts_before':       n_conflict_before,
        'conflicts_after':        n_conflict_after,
        'conflicts_resolved':     resolved,
        'resolution_rate_pct':    round(100 * resolved / n_conflict_before, 1)
                                  if n_conflict_before > 0 else None,
        'mean_dist_before_m':     round(sum(db) / len(db), 3) if db else None,
        'mean_dist_after_m':      round(sum(da) / len(da), 3) if da else None,
        'rule_agreement_pct':     pct(rules, 'agreement'),
        'rule_no_conflict_pct':   pct(rules, 'no_conflict'),
        'rule_ttc_pct':           pct(rules, 'ttc'),
        'rule_osm_pct':           pct(rules, 'osm'),
        'rule_tiebreak_pct':      pct(rules, 'tiebreak'),
        'yielding_ego_pct':       pct(yields, 'ego'),
        'yielding_cav1_pct':      pct(yields, '1'),
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 3: resolver evaluation')
    parser.add_argument('--per_frame', required=True,
                        help='Path to per_frame.json from role_swap_eval.py')
    parser.add_argument('--safety_m', type=float, default=DEFAULT_SAFETY_M,
                        help=f'Safety threshold in metres (default {DEFAULT_SAFETY_M})')
    parser.add_argument('--output_dir', default='outputs/resolver_eval')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    frames = load_per_frame(args.per_frame)
    print(f'Loaded {len(frames)} frames from {args.per_frame}')

    per_frame_results = [eval_frame(f, args.safety_m) for f in frames]

    # per-scenario aggregation
    by_scene = defaultdict(list)
    for r in per_frame_results:
        by_scene[r['scenario_index']].append(r)

    per_scene = []
    for sc_idx in sorted(by_scene.keys()):
        per_scene.append({
            'scenario_index': sc_idx,
            **aggregate(by_scene[sc_idx]),
        })
    overall = {'scenario_index': 'all', **aggregate(per_frame_results)}
    per_scene.append(overall)

    # write outputs
    frame_path  = os.path.join(args.output_dir, 'per_frame.json')
    scene_path  = os.path.join(args.output_dir, 'per_scene.json')
    csv_path    = os.path.join(args.output_dir, 'summary.csv')

    with open(frame_path, 'w') as f:
        json.dump(per_frame_results, f, indent=2)
    with open(scene_path, 'w') as f:
        json.dump(per_scene, f, indent=2)

    if per_scene:
        fieldnames = [k for k in per_scene[0].keys()]
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_scene)

    print(f'Outputs written to {args.output_dir}/')

    # ── Print summary ──────────────────────────────────────────────────────────
    ov = overall
    print(f"""
── Resolver evaluation summary ─────────────────────────────────────────────
  Frames analysed:         {ov['n_frames']}
  Safety threshold:        {args.safety_m} m

  Trajectory conflicts BEFORE resolver:  {ov['conflicts_before']:4d}  ({ov['conflict_before_pct']}%)
  Trajectory conflicts AFTER  resolver:  {ov['conflicts_after']:4d}  ({ov['conflict_after_pct']}%)
  Conflicts resolved:                    {ov['conflicts_resolved']:4d}
  Resolution rate:                       {ov['resolution_rate_pct']}%

  Mean min-distance BEFORE:  {ov['mean_dist_before_m']} m
  Mean min-distance AFTER:   {ov['mean_dist_after_m']} m

  Rule distribution (% of all frames):
    Agreement (no change):   {ov['rule_agreement_pct']}%
    No physical conflict:    {ov['rule_no_conflict_pct']}%
    TTC (faster yields):     {ov['rule_ttc_pct']}%
    OSM right-of-way:        {ov['rule_osm_pct']}%
    Deterministic tiebreak:  {ov['rule_tiebreak_pct']}%

  When resolver acted (conflict detected):
    Ego yielded:   {ov['yielding_ego_pct']}%  of yield decisions
    CAV 1 yielded: {ov['yielding_cav1_pct']}%  of yield decisions
─────────────────────────────────────────────────────────────────────────────""")


if __name__ == '__main__':
    main()
