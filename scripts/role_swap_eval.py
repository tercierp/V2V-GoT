#!/usr/bin/env python3
"""
Phase 0 pre-flight: measure decision divergence between ego and CAV 1.

Loads existing GT QA data (nq8/nq9 JSON files) — no LLM or GPU required.
Optionally also loads LLM inference outputs (merge.jsonl) when available
to measure model-level divergence in addition to GT-level divergence.

Usage (GT only, runs in ~seconds):
    python scripts/role_swap_eval.py \
        --nq8_gt  <path>/nq8sm3w6dc.json \
        --nq9_gt  <path>/nq9sm3w6dc.json \
        --output_dir outputs/role_swap

Usage with LLM outputs (requires inference to have been run first):
    python scripts/role_swap_eval.py \
        --nq8_gt  <path>/nq8sm3w6dc.json \
        --nq9_gt  <path>/nq9sm3w6dc.json \
        --nq8_llm <path>/nq8_merge.jsonl \
        --nq9_llm <path>/nq9_merge.jsonl \
        --output_dir outputs/role_swap

Outputs:
    <output_dir>/per_frame.json    — per-frame metrics
    <output_dir>/per_scene.json    — per-scenario aggregates
    <output_dir>/summary.csv       — printable summary table
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict

# ── Speed / steering class definitions (must match inference.py) ───────────────
SPEED_CLASSES    = ['fast', 'moderate', 'slow', 'very slow', 'stop']
STEERING_CLASSES = ['left', 'slightly left', 'straight', 'slightly right', 'right']

SAFETY_THRESHOLD_M = 3.0   # metres — CAVs closer than this are "in conflict"
NUM_SCENARIOS      = 9     # V2V4Real val set has 9 sequences


# ── Parsing helpers ────────────────────────────────────────────────────────────

def parse_trajectory(traj_str: str) -> list[tuple[float, float]]:
    """Parse '[(x0,z0),(x1,z1),...]' → list of (x, z) float tuples."""
    pairs = re.findall(r'\(([^)]+)\)', traj_str)
    result = []
    for p in pairs:
        parts = p.split(',')
        if len(parts) == 2:
            try:
                result.append((float(parts[0].strip()), float(parts[1].strip())))
            except ValueError:
                pass
    return result


def parse_nq8_llm_output(text: str) -> tuple[str | None, str | None]:
    """
    Extract (speed_str, steering_str) from nq8 LLM output.
    Expected format: 'The suggested speed setting is: fast. The suggested steering setting is: straight.'
    Returns (None, None) if parsing fails.
    """
    speed = re.search(r'speed setting is:\s*([a-z ]+)\.', text, re.IGNORECASE)
    steer = re.search(r'steering setting is:\s*([a-z ]+)\.', text, re.IGNORECASE)
    speed_str = speed.group(1).strip().lower() if speed else None
    steer_str = steer.group(1).strip().lower() if steer else None
    return speed_str, steer_str


def parse_nq9_llm_output(text: str) -> list[tuple[float, float]]:
    """
    Extract trajectory from nq9 LLM output.
    Expected format: 'The suggested future trajectory is [(x0,z0),(x1,z1),...].'
    Handles truncated outputs (missing closing bracket).
    Returns [] if parsing fails.
    """
    # greedy: grab everything from first '[' to closing ']' or end-of-string
    m = re.search(r'\[(.+?)(?:\]|$)', text, re.DOTALL)
    if not m:
        return []
    return parse_trajectory('[' + m.group(1) + ']')


# ── Geometric metrics ──────────────────────────────────────────────────────────

def min_trajectory_distance(traj_a: list, traj_b: list) -> float:
    """
    Min L2 distance between two trajectory sequences, aligned by timestep.
    If lengths differ, compares up to the shorter length.
    Returns inf if either trajectory is empty.
    """
    if not traj_a or not traj_b:
        return float('inf')
    n = min(len(traj_a), len(traj_b))
    return min(
        math.sqrt((traj_a[i][0] - traj_b[i][0])**2 + (traj_a[i][1] - traj_b[i][1])**2)
        for i in range(n)
    )


def time_to_collision(
    traj_ego: list, traj_cav1: list, threshold_m: float = SAFETY_THRESHOLD_M
) -> int | None:
    """
    First timestep index where |ego[t] - cav1[t]| < threshold_m.
    Returns None if the trajectories never come that close.
    """
    n = min(len(traj_ego), len(traj_cav1))
    for i in range(n):
        d = math.sqrt(
            (traj_ego[i][0] - traj_cav1[i][0])**2
            + (traj_ego[i][1] - traj_cav1[i][1])**2
        )
        if d < threshold_m:
            return i
    return None


def action_agreement(speed_a: int, steer_a: int, speed_b: int, steer_b: int) -> str:
    """
    Classify pair of actions into agreement categories.
    Returns one of: 'full_agree' | 'speed_agree' | 'steer_agree' | 'disagree'
    """
    sp = speed_a == speed_b
    st = steer_a == steer_b
    if sp and st:
        return 'full_agree'
    if sp:
        return 'speed_agree'
    if st:
        return 'steer_agree'
    return 'disagree'


def reconcilability(speed_ego: int, speed_cav1: int) -> str:
    """
    Simple rule: can the conflict be resolved by one CAV yielding?
    Returns 'both_yield' | 'one_yields' | 'both_go' | 'both_stop'
    """
    both_going = speed_ego <= 1 and speed_cav1 <= 1   # fast or moderate
    both_slow  = speed_ego >= 3 and speed_cav1 >= 3   # very slow or stop
    if both_going:
        return 'both_go'
    if both_slow:
        return 'both_stop'
    # one is faster than the other → one yields
    return 'one_yields'


# ── Data loading ───────────────────────────────────────────────────────────────

def load_gt_json(path: str) -> dict:
    """
    Load a GT QA JSON file.
    Returns dict: (global_timestamp_index, asker_cav_id) → record.
    """
    with open(path) as f:
        data = json.load(f)
    index = {}
    for rec in data:
        key = (rec['global_timestamp_index'], rec['asker_cav_id'])
        index[key] = rec
    return index


def load_llm_jsonl(path: str) -> dict:
    """
    Load an LLM inference merge.jsonl.
    Returns dict: id → record.
    V2V-GoT merge.jsonl stores the model prediction in the 'outputs' field.
    """
    if path is None or not os.path.exists(path):
        return {}
    index = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get('question_id', rec.get('id'))
            # normalise: expose prediction under 'text' for downstream consumers
            if 'text' not in rec and 'outputs' in rec:
                rec['text'] = rec['outputs']
            index[qid] = rec
    return index


# ── Per-frame analysis ─────────────────────────────────────────────────────────

def analyse_frame(
    ts: int, scenario_idx: int,
    nq8_gt: dict, nq9_gt: dict,
    nq8_llm: dict, nq9_llm: dict,
    gt_id_map: dict,   # (ts, cav_id) → record['id'] for LLM lookup
) -> dict:
    """
    Compute all divergence metrics for a single frame (global_timestamp_index).
    Returns a result dict, or None if the frame is missing either CAV's record.
    """
    result = {
        'global_timestamp_index': ts,
        'scenario_index': scenario_idx,
    }

    # ── GT-level metrics ──────────────────────────────────────────────────────
    rec8_ego = nq8_gt.get((ts, 'ego'))
    rec8_cav1 = nq8_gt.get((ts, '1'))
    rec9_ego = nq9_gt.get((ts, 'ego'))
    rec9_cav1 = nq9_gt.get((ts, '1'))

    if not (rec8_ego and rec8_cav1 and rec9_ego and rec9_cav1):
        return None   # frame not present for both CAVs — skip

    # actions
    sp_ego   = rec8_ego['suggested_speed_idx']
    st_ego   = rec8_ego['suggested_steering_idx']
    sp_cav1  = rec8_cav1['suggested_speed_idx']
    st_cav1  = rec8_cav1['suggested_steering_idx']

    result['gt'] = {
        'ego_speed':   SPEED_CLASSES[sp_ego],
        'ego_steer':   STEERING_CLASSES[st_ego],
        'cav1_speed':  SPEED_CLASSES[sp_cav1],
        'cav1_steer':  STEERING_CLASSES[st_cav1],
        'action_agreement': action_agreement(sp_ego, st_ego, sp_cav1, st_cav1),
        'reconcilability':  reconcilability(sp_ego, sp_cav1),
    }

    # trajectories (already in ego's reference frame)
    traj_ego  = parse_trajectory(rec9_ego['future_trajectory_str_in_ego'])
    traj_cav1 = parse_trajectory(rec9_cav1['future_trajectory_str_in_ego'])

    min_dist = min_trajectory_distance(traj_ego, traj_cav1)
    ttc      = time_to_collision(traj_ego, traj_cav1)

    result['gt']['traj_min_dist_m']    = round(min_dist, 3) if min_dist != float('inf') else None
    result['gt']['time_to_collision']  = ttc       # timestep index or None
    result['gt']['conflict']           = ttc is not None
    result['gt']['ego_traj']           = traj_ego
    result['gt']['cav1_traj']          = traj_cav1

    # ── LLM-level metrics (optional) ─────────────────────────────────────────
    if nq8_llm or nq9_llm:
        id_ego_8  = gt_id_map.get((ts, 'ego',  'nq8'))
        id_cav1_8 = gt_id_map.get((ts, '1',    'nq8'))
        id_ego_9  = gt_id_map.get((ts, 'ego',  'nq9'))
        id_cav1_9 = gt_id_map.get((ts, '1',    'nq9'))

        llm8_ego  = nq8_llm.get(id_ego_8,  {})
        llm8_cav1 = nq8_llm.get(id_cav1_8, {})
        llm9_ego  = nq9_llm.get(id_ego_9,  {})
        llm9_cav1 = nq9_llm.get(id_cav1_9, {})

        llm_result = {}

        # nq8 LLM actions
        if llm8_ego and llm8_cav1:
            sp_ego_s,  st_ego_s  = parse_nq8_llm_output(llm8_ego.get('text', ''))
            sp_cav1_s, st_cav1_s = parse_nq8_llm_output(llm8_cav1.get('text', ''))

            sp_ego_i  = SPEED_CLASSES.index(sp_ego_s)   if sp_ego_s  in SPEED_CLASSES    else -1
            sp_cav1_i = SPEED_CLASSES.index(sp_cav1_s)  if sp_cav1_s in SPEED_CLASSES    else -1
            st_ego_i  = STEERING_CLASSES.index(st_ego_s)  if st_ego_s  in STEERING_CLASSES else -1
            st_cav1_i = STEERING_CLASSES.index(st_cav1_s) if st_cav1_s in STEERING_CLASSES else -1

            llm_result['ego_speed']  = sp_ego_s
            llm_result['ego_steer']  = st_ego_s
            llm_result['cav1_speed'] = sp_cav1_s
            llm_result['cav1_steer'] = st_cav1_s

            if sp_ego_i >= 0 and sp_cav1_i >= 0 and st_ego_i >= 0 and st_cav1_i >= 0:
                llm_result['action_agreement'] = action_agreement(sp_ego_i, st_ego_i, sp_cav1_i, st_cav1_i)
                llm_result['reconcilability']  = reconcilability(sp_ego_i, sp_cav1_i)

        # nq9 LLM trajectories
        if llm9_ego and llm9_cav1:
            traj_ego_llm  = parse_nq9_llm_output(llm9_ego.get('text', ''))
            traj_cav1_llm = parse_nq9_llm_output(llm9_cav1.get('text', ''))
            md = min_trajectory_distance(traj_ego_llm, traj_cav1_llm)
            tc = time_to_collision(traj_ego_llm, traj_cav1_llm)
            llm_result['traj_min_dist_m']   = round(md, 3) if md != float('inf') else None
            llm_result['time_to_collision'] = tc
            llm_result['conflict']          = tc is not None

        result['llm'] = llm_result

    return result


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(frames: list[dict], source: str = 'gt') -> dict:
    """Compute aggregate stats over a list of frame result dicts."""
    records = [f[source] for f in frames if source in f]
    if not records:
        return {}

    n = len(records)
    agreements = [r['action_agreement'] for r in records if 'action_agreement' in r]
    conflicts  = [r['conflict'] for r in records if 'conflict' in r]
    dists      = [r['traj_min_dist_m'] for r in records
                  if r.get('traj_min_dist_m') is not None]
    recon      = [r['reconcilability'] for r in records if 'reconcilability' in r]

    def pct(lst, val): return round(100 * lst.count(val) / len(lst), 1) if lst else None

    return {
        'n_frames':              n,
        'full_agree_pct':        pct(agreements, 'full_agree'),
        'speed_agree_only_pct':  pct(agreements, 'speed_agree'),
        'steer_agree_only_pct':  pct(agreements, 'steer_agree'),
        'full_disagree_pct':     pct(agreements, 'disagree'),
        'conflict_pct':          pct(conflicts,  True),
        'no_conflict_pct':       pct(conflicts,  False),
        'mean_min_dist_m':       round(sum(dists) / len(dists), 3) if dists else None,
        'min_min_dist_m':        round(min(dists), 3) if dists else None,
        'both_go_pct':           pct(recon, 'both_go'),
        'one_yields_pct':        pct(recon, 'one_yields'),
        'both_stop_pct':         pct(recon, 'both_stop'),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def build_id_map(nq8_gt_raw: list, nq9_gt_raw: list) -> dict:
    """Build (ts, cav_id, 'nq8'/'nq9') → record id mapping for LLM lookup."""
    m = {}
    for rec in nq8_gt_raw:
        m[(rec['global_timestamp_index'], rec['asker_cav_id'], 'nq8')] = rec['id']
    for rec in nq9_gt_raw:
        m[(rec['global_timestamp_index'], rec['asker_cav_id'], 'nq9')] = rec['id']
    return m


def main():
    parser = argparse.ArgumentParser(description='Phase 0 pre-flight divergence measurement')
    parser.add_argument('--nq8_gt',  required=True, help='Path to nq8 GT QA JSON')
    parser.add_argument('--nq9_gt',  required=True, help='Path to nq9 GT QA JSON')
    parser.add_argument('--nq8_llm', default=None,  help='Path to nq8 LLM merge.jsonl (optional)')
    parser.add_argument('--nq9_llm', default=None,  help='Path to nq9 LLM merge.jsonl (optional)')
    parser.add_argument('--num_scenes', type=int, default=NUM_SCENARIOS,
                        help='Number of scenarios to evaluate (default: all 9)')
    parser.add_argument('--output_dir', default='outputs/role_swap',
                        help='Directory for output JSON and CSV files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load raw lists for id_map, then index by (ts, cav_id)
    print(f'Loading GT data from:\n  {args.nq8_gt}\n  {args.nq9_gt}')
    with open(args.nq8_gt) as f:
        nq8_gt_raw = json.load(f)
    with open(args.nq9_gt) as f:
        nq9_gt_raw = json.load(f)

    nq8_gt = {(r['global_timestamp_index'], r['asker_cav_id']): r for r in nq8_gt_raw}
    nq9_gt = {(r['global_timestamp_index'], r['asker_cav_id']): r for r in nq9_gt_raw}
    gt_id_map = build_id_map(nq8_gt_raw, nq9_gt_raw)

    nq8_llm = load_llm_jsonl(args.nq8_llm)
    nq9_llm = load_llm_jsonl(args.nq9_llm)
    have_llm = bool(nq8_llm or nq9_llm)
    if have_llm:
        print(f'Loaded LLM outputs: {len(nq8_llm)} nq8, {len(nq9_llm)} nq9 records')
    else:
        print('No LLM outputs provided — GT-only analysis')

    # collect all timestamps from nq9 GT (has the fullest coverage)
    # group timestamps by scenario_index using the scenario_index field in records
    ts_to_scenario = {r['global_timestamp_index']: r['scenario_index'] for r in nq9_gt_raw}

    # per-frame analysis
    per_frame_results = []
    skipped = 0
    timestamps = sorted(ts_to_scenario.keys())
    for ts in timestamps:
        scenario_idx = ts_to_scenario[ts]
        if scenario_idx >= args.num_scenes:
            continue
        frame_result = analyse_frame(
            ts, scenario_idx,
            nq8_gt, nq9_gt,
            nq8_llm, nq9_llm,
            gt_id_map,
        )
        if frame_result is None:
            skipped += 1
            continue
        per_frame_results.append(frame_result)

    print(f'Analysed {len(per_frame_results)} frames, skipped {skipped}')

    # per-scenario aggregation
    frames_by_scene = defaultdict(list)
    for f in per_frame_results:
        frames_by_scene[f['scenario_index']].append(f)

    per_scene_results = []
    for sc_idx in sorted(frames_by_scene.keys()):
        frames = frames_by_scene[sc_idx]
        entry = {
            'scenario_index': sc_idx,
            'gt': aggregate(frames, 'gt'),
        }
        if have_llm:
            entry['llm'] = aggregate(frames, 'llm')
        per_scene_results.append(entry)

    # overall aggregate
    overall = {
        'scenario_index': 'all',
        'gt': aggregate(per_frame_results, 'gt'),
    }
    if have_llm:
        overall['llm'] = aggregate(per_frame_results, 'llm')
    per_scene_results.append(overall)

    # write outputs
    per_frame_path = os.path.join(args.output_dir, 'per_frame.json')
    per_scene_path = os.path.join(args.output_dir, 'per_scene.json')
    summary_path   = os.path.join(args.output_dir, 'summary.csv')

    with open(per_frame_path, 'w') as f:
        json.dump(per_frame_results, f, indent=2)

    with open(per_scene_path, 'w') as f:
        json.dump(per_scene_results, f, indent=2)

    # summary CSV — GT metrics only (LLM row added if available)
    csv_rows = []
    for entry in per_scene_results:
        row = {'scenario': entry['scenario_index']}
        for k, v in entry.get('gt', {}).items():
            row[f'gt_{k}'] = v
        if have_llm:
            for k, v in entry.get('llm', {}).items():
                row[f'llm_{k}'] = v
        csv_rows.append(row)

    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f'\nOutputs written to {args.output_dir}/')
    print(f'  {per_frame_path}')
    print(f'  {per_scene_path}')
    print(f'  {summary_path}')

    # print summary table to stdout
    ov = per_scene_results[-1]['gt']
    print('\n── GT-level divergence summary (all scenarios) ──')
    print(f"  Frames analysed:       {ov.get('n_frames')}")
    print(f"  Full agreement:        {ov.get('full_agree_pct')}%")
    print(f"  Speed agree only:      {ov.get('speed_agree_only_pct')}%")
    print(f"  Steering agree only:   {ov.get('steer_agree_only_pct')}%")
    print(f"  Full disagreement:     {ov.get('full_disagree_pct')}%")
    print(f"  Trajectory conflict:   {ov.get('conflict_pct')}%  "
          f"(threshold {SAFETY_THRESHOLD_M} m)")
    print(f"  Mean min-distance:     {ov.get('mean_min_dist_m')} m")
    print(f"  Min min-distance:      {ov.get('min_min_dist_m')} m")
    print(f"  Both go (hard to resolve):   {ov.get('both_go_pct')}%")
    print(f"  One yields (easy to resolve):{ov.get('one_yields_pct')}%")
    print(f"  Both stop:             {ov.get('both_stop_pct')}%")

    if have_llm and 'llm' in per_scene_results[-1]:
        lv = per_scene_results[-1]['llm']
        print('\n── LLM-level divergence summary (all scenarios) ──')
        print(f"  Full agreement:        {lv.get('full_agree_pct')}%")
        print(f"  Trajectory conflict:   {lv.get('conflict_pct')}%")
        print(f"  Mean min-distance:     {lv.get('mean_min_dist_m')} m")

    # Gate decision hint
    conflict_pct = ov.get('conflict_pct', 100)
    disagree_pct = ov.get('full_disagree_pct', 100)
    print('\n── Phase 0 gate recommendation ──')
    if conflict_pct is not None and conflict_pct < 10 and disagree_pct is not None and disagree_pct < 20:
        print('  → LOW divergence. Resolver alone is likely sufficient.')
        print('    Proceed to Phase 3 (resolver) before committing GPU time to Phase 2 (LoRA).')
    elif conflict_pct is not None and conflict_pct < 30:
        print('  → MODERATE divergence. Run resolver first; assess residual conflict.')
        print('    LoRA (Phase 2) may still improve results but is not guaranteed.')
    else:
        print('  → HIGH divergence. LoRA (Phase 2) is recommended.')
        print('    Report numbers to confirm before starting the 24h training run.')


if __name__ == '__main__':
    main()
