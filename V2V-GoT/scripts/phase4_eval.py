#!/usr/bin/env python3
"""
Phase 4: full evaluation — four-variant comparison table.

Variants:
  (A)       Centralized baseline — ego LLM trajectory vs cav1 GT trajectory.
            Models the current V2V-GoT setup: ego plans with the LLM, cav1
            follows its actual GT path. Conflict rate is the baseline risk.
  (B)       Peer-symmetric + resolver — both CAVs run ego-inference; resolver
            reconciles their plans using the TTC / OSM / tiebreak cascade.
  (C)       Ablation: peer-symmetric, no resolver — ego always yields on conflict
            (random tiebreak), no TTC or OSM logic.

Usage:
    python scripts/phase4_eval.py \
        --nq8_gt   <path>/nq8sm3w6dc.json \
        --nq9_gt   <path>/nq9sm3w6dc.json \
        --nq8_llm  <path>/nq8sm3w6dc_merge.jsonl \
        --nq9_llm  <path>/nq9sm3w6dc_merge.jsonl \
        --output_dir outputs/phase4
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from v2vgotd.resolver import (
    EgoDecision, resolve, _trajectory_conflict,
    DEFAULT_SAFETY_M, SPEED_CLASSES, STEERING_CLASSES, _SPEED_FALLBACK,
)

SAFETY_M = DEFAULT_SAFETY_M
_SPEED_MPS = [8.0, 5.5, 3.5, 1.5, 0.0]   # rough m per 0.5 s step per speed class


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_trajectory(traj_str: str) -> list:
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


def parse_nq8(text: str) -> tuple[Optional[int], Optional[int]]:
    sp = re.search(r'speed setting is:\s*([a-z ]+?)(?:\.|$)', text, re.IGNORECASE)
    st = re.search(r'steering setting is:\s*([a-z ]+?)(?:\.|$)', text, re.IGNORECASE)
    sp_str = sp.group(1).strip().lower() if sp else None
    st_str = st.group(1).strip().lower() if st else None
    sp_idx = SPEED_CLASSES.index(sp_str)    if sp_str in SPEED_CLASSES    else None
    st_idx = STEERING_CLASSES.index(st_str) if st_str in STEERING_CLASSES else None
    return sp_idx, st_idx


def parse_nq9(text: str) -> list:
    m = re.search(r'\[(.+?)(?:\]|$)', text, re.DOTALL)
    return parse_trajectory('[' + m.group(1) + ']') if m else []


def traj_endpoint_error(pred: list, gt: list) -> Optional[float]:
    if not pred or not gt:
        return None
    px, pz = pred[-1]; gx, gz = gt[-1]
    return math.sqrt((px - gx)**2 + (pz - gz)**2)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_gt(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {(r['global_timestamp_index'], r['asker_cav_id']): r for r in data}


def load_llm(path: str) -> dict:
    index = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get('question_id', rec.get('id'))
            rec.setdefault('text', rec.get('outputs', ''))
            index[qid] = rec
    return index


# ── Trajectory scaling helper ─────────────────────────────────────────────────

def _scale(traj, old_idx, new_idx):
    if not traj or old_idx == new_idx or _SPEED_MPS[old_idx] == 0:
        return traj
    r = _SPEED_MPS[new_idx] / _SPEED_MPS[old_idx]
    return [(x * r, z * r) for x, z in traj]


# ── Per-frame analysis ─────────────────────────────────────────────────────────

def analyse(ts: int, sc: int,
            nq8_gt: dict, nq9_gt: dict,
            nq8_llm: dict, nq9_llm: dict) -> Optional[dict]:

    r8e = nq8_gt.get((ts, 'ego')); r8c = nq8_gt.get((ts, '1'))
    r9e = nq9_gt.get((ts, 'ego')); r9c = nq9_gt.get((ts, '1'))
    if not (r8e and r8c and r9e and r9c):
        return None

    l8e = nq8_llm.get(r8e['id'], {}); l8c = nq8_llm.get(r8c['id'], {})
    l9e = nq9_llm.get(r9e['id'], {}); l9c = nq9_llm.get(r9c['id'], {})
    if not (l8e and l8c and l9e and l9c):
        return None

    # ── Trajectories ──────────────────────────────────────────────────────────
    gt_traj_ego  = parse_trajectory(r9e['future_trajectory_str_in_ego'])
    gt_traj_cav1 = parse_trajectory(r9c['future_trajectory_str_in_ego'])
    llm_traj_ego  = parse_nq9(l9e.get('text', ''))
    llm_traj_cav1 = parse_nq9(l9c.get('text', ''))

    llm_sp_ego,  llm_st_ego  = parse_nq8(l8e.get('text', ''))
    llm_sp_cav1, llm_st_cav1 = parse_nq8(l8c.get('text', ''))
    sp_valid = llm_sp_ego is not None and llm_sp_cav1 is not None
    st_valid = llm_st_ego is not None and llm_st_cav1 is not None

    # ── Action accuracy ───────────────────────────────────────────────────────
    gt_sp_ego, gt_st_ego = r8e['suggested_speed_idx'], r8e['suggested_steering_idx']

    # ── (A) Centralized baseline: ego LLM traj vs cav1 GT traj ───────────────
    # The standard V2V-GoT setup: only ego is planning; cav1 follows GT.
    conflict_A, min_dist_A, _ = _trajectory_conflict(
        llm_traj_ego, gt_traj_cav1, SAFETY_M)

    # ── (B) Peer-symmetric + resolver ────────────────────────────────────────
    conflict_B_before, min_dist_B_before, _ = _trajectory_conflict(
        llm_traj_ego, llm_traj_cav1, SAFETY_M)

    resolver_rule = resolver_yield = None
    conflict_B_after = conflict_B_before
    min_dist_B_after = min_dist_B_before

    if sp_valid and st_valid and llm_traj_ego and llm_traj_cav1:
        dec_ego  = EgoDecision('ego', llm_sp_ego,  llm_st_ego,  llm_traj_ego)
        dec_cav1 = EgoDecision('1',   llm_sp_cav1, llm_st_cav1, llm_traj_cav1)
        resolved = resolve(dec_ego, dec_cav1, osm=None, safety_threshold_m=SAFETY_M)
        resolver_rule  = resolved.rule_applied
        resolver_yield = resolved.yielding_cav
        res_traj_ego  = _scale(llm_traj_ego,  llm_sp_ego,  resolved.ego_speed_idx)
        res_traj_cav1 = _scale(llm_traj_cav1, llm_sp_cav1, resolved.cav1_speed_idx)
        conflict_B_after, min_dist_B_after, _ = _trajectory_conflict(
            res_traj_ego, res_traj_cav1, SAFETY_M)
    else:
        res_traj_ego  = llm_traj_ego
        res_traj_cav1 = llm_traj_cav1

    # ── (C) Ablation: peer-symmetric, ego always yields ───────────────────────
    if conflict_B_before and llm_sp_ego is not None and llm_traj_ego:
        ab_sp = _SPEED_FALLBACK[llm_sp_ego]
        ab_traj_ego = _scale(llm_traj_ego, llm_sp_ego, ab_sp)
        conflict_C, _, _ = _trajectory_conflict(ab_traj_ego, llm_traj_cav1, SAFETY_M)
    else:
        conflict_C = conflict_B_before

    natural_agree = (sp_valid and st_valid
                     and llm_sp_ego == llm_sp_cav1
                     and llm_st_ego == llm_st_cav1)

    # ablation trajectory for visualizer: ego slows one step if there was a conflict
    if conflict_B_before and llm_sp_ego is not None and llm_traj_ego:
        ab_traj_ego = _scale(llm_traj_ego, llm_sp_ego, _SPEED_FALLBACK[llm_sp_ego])
    else:
        ab_traj_ego = llm_traj_ego

    return {
        'ts': ts, 'sc': sc,
        # A
        'conflict_A': conflict_A,
        'min_dist_A': round(min_dist_A, 3) if min_dist_A != float('inf') else None,
        # B
        'natural_agree':       natural_agree,
        'conflict_B_before':   conflict_B_before,
        'min_dist_B_before':   round(min_dist_B_before, 3) if min_dist_B_before != float('inf') else None,
        'conflict_B_after':    conflict_B_after,
        'min_dist_B_after':    round(min_dist_B_after, 3)  if min_dist_B_after  != float('inf') else None,
        'resolver_rule':       resolver_rule,
        'resolver_yield':      resolver_yield,
        # C
        'conflict_C':          conflict_C,
        # accuracy
        'sp_acc_ego': (llm_sp_ego == gt_sp_ego) if llm_sp_ego is not None else None,
        'st_acc_ego': (llm_st_ego == gt_st_ego) if llm_st_ego is not None else None,
        'ep_err_ego':  traj_endpoint_error(llm_traj_ego,  gt_traj_ego),
        'ep_err_cav1': traj_endpoint_error(llm_traj_cav1, gt_traj_cav1),
        # trajectories — stored for visualizer
        'traj': {
            'gt_ego':   gt_traj_ego,
            'gt_cav1':  gt_traj_cav1,
            'llm_ego':  llm_traj_ego,
            'llm_cav1': llm_traj_cav1,
            'res_ego':  res_traj_ego,
            'res_cav1': res_traj_cav1,
            'ab_ego':   ab_traj_ego,   # ablation: ego scaled down one step
        },
    }


# ── Aggregation ────────────────────────────────────────────────────────────────

def pct(lst, val=True):
    valid = [x for x in lst if x is not None]
    return round(100 * sum(1 for x in valid if x == val) / len(valid), 1) if valid else None


def mean_f(lst):
    valid = [x for x in lst if x is not None]
    return round(sum(valid) / len(valid), 3) if valid else None


def agg(records: list) -> dict:
    n = len(records)
    rules  = [r['resolver_rule'] for r in records if r['resolver_rule']]
    cb     = [r['conflict_B_before'] for r in records]
    ca_B   = [r['conflict_B_after']  for r in records]
    n_cb   = sum(1 for x in cb if x)
    n_ca_B = sum(1 for x in ca_B if x)
    resolved = sum(1 for b, a in zip(cb, ca_B) if b and not a)

    # per-scenario conflict breakdown (only used at 'all' level)
    by_sc: dict[int, int] = defaultdict(int)
    for r in records:
        if r['conflict_B_before']:
            by_sc[r['sc']] += 1

    return {
        'n_frames': n,
        # (A) centralized
        'conflict_A_pct':        pct([r['conflict_A'] for r in records]),
        'n_conflicts_A':         sum(1 for r in records if r['conflict_A']),
        # action accuracy (ego perspective)
        'speed_acc_pct':         pct([r['sp_acc_ego']  for r in records]),
        'steer_acc_pct':         pct([r['st_acc_ego']  for r in records]),
        'mean_ep_err_m':         mean_f([r['ep_err_ego'] for r in records]),
        # (B) peer-symmetric + resolver
        'natural_agree_pct':     pct([r['natural_agree'] for r in records]),
        'conflict_B_before_pct': pct(cb),
        'conflict_B_after_pct':  pct(ca_B),
        'n_conflicts_B_before':  n_cb,
        'n_conflicts_B_after':   n_ca_B,
        'n_conflicts_resolved':  resolved,
        'resolution_rate_pct':   round(100 * resolved / n_cb, 1) if n_cb else None,
        # (C) ablation
        'conflict_C_pct':        pct([r['conflict_C'] for r in records]),
        'n_conflicts_C':         sum(1 for r in records if r['conflict_C']),
        # resolver rule breakdown
        'rule_agreement_pct':    pct(rules, 'agreement'),
        'rule_no_conflict_pct':  pct(rules, 'no_conflict'),
        'rule_ttc_pct':          pct(rules, 'ttc'),
        'rule_tiebreak_pct':     pct(rules, 'tiebreak'),
        # per-scenario conflict breakdown (for 'all' row)
        'conflicts_B_by_scenario': dict(by_sc),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    BACKUP   = '/home/tercier/v2v-got-backup-20260514'
    EVAL_DIR = f'{BACKUP}/eval_results/osm_from_q9_2500'
    PLAY     = (f'{BACKUP}/LLaVA/playground/data/eval/'
                'v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2_4330_full')

    parser = argparse.ArgumentParser()
    parser.add_argument('--nq8_gt',    default=f'{PLAY}_nq8sm3w6dc/answers/val/llava-v1.5-7b/nq8sm3w6dc.json')
    parser.add_argument('--nq9_gt',    default=f'{PLAY}_nq9sm3w6dc/answers/val/llava-v1.5-7b/nq9sm3w6dc.json')
    parser.add_argument('--nq8_llm',   default=f'{EVAL_DIR}/nq8sm3w6dc_merge.jsonl')
    parser.add_argument('--nq9_llm',   default=f'{EVAL_DIR}/nq9sm3w6dc_merge.jsonl')
    parser.add_argument('--output_dir', default='outputs/phase4')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading data...')
    nq8_gt  = load_gt(args.nq8_gt);   nq9_gt  = load_gt(args.nq9_gt)
    nq8_llm = load_llm(args.nq8_llm); nq9_llm = load_llm(args.nq9_llm)

    ts_to_sc = {ts: r['scenario_index']
                for (ts, cav), r in nq9_gt.items() if cav == 'ego'}

    print(f'Analysing {len(ts_to_sc)} frames...')
    records, skipped = [], 0
    for ts in sorted(ts_to_sc):
        r = analyse(ts, ts_to_sc[ts], nq8_gt, nq9_gt, nq8_llm, nq9_llm)
        if r is None:
            skipped += 1
        else:
            records.append(r)
    print(f'  Processed {len(records)}, skipped {skipped}')

    # aggregate per scenario
    by_sc = defaultdict(list)
    for r in records:
        by_sc[r['sc']].append(r)

    per_scene = [{'scenario': sc, **agg(recs)} for sc, recs in sorted(by_sc.items())]
    overall   = {'scenario': 'all', **agg(records)}
    per_scene.append(overall)

    # write outputs (strip traj from per_scene to keep CSV clean)
    records_no_traj = [{k: v for k, v in r.items() if k != 'traj'} for r in records]
    with open(os.path.join(args.output_dir, 'per_frame.json'), 'w') as f:
        json.dump(records, f)          # keeps traj for visualiser
    with open(os.path.join(args.output_dir, 'per_frame_notraj.json'), 'w') as f:
        json.dump(records_no_traj, f, indent=2)
    with open(os.path.join(args.output_dir, 'per_scene.json'), 'w') as f:
        json.dump(per_scene, f, indent=2)

    # CSV — flatten conflicts_B_by_scenario
    csv_rows = []
    for row in per_scene:
        flat = {k: v for k, v in row.items() if k != 'conflicts_B_by_scenario'}
        csv_rows.append(flat)
    if csv_rows:
        with open(os.path.join(args.output_dir, 'summary.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader(); w.writerows(csv_rows)

    ov = overall
    sc_breakdown = ov.get('conflicts_B_by_scenario', {})
    sc_str = '  '.join(f'sc{k}:{v}' for k, v in sorted(sc_breakdown.items()))

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             Phase 4 evaluation — {ov['n_frames']} frames, threshold {SAFETY_M} m              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  (A) Centralized baseline (ego LLM plan vs cav1 GT trajectory)             ║
║    Speed accuracy vs GT        {str(ov['speed_acc_pct']):>6}%                                ║
║    Steering accuracy vs GT     {str(ov['steer_acc_pct']):>6}%                                ║
║    Traj endpoint error vs GT   {str(ov['mean_ep_err_m']):>6} m                               ║
║    Conflict rate (ego vs cav1-GT)  {str(ov['conflict_A_pct']):>5}%  ({ov['n_conflicts_A']} frames)          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  (B) Peer-symmetric + resolver (ours)                                       ║
║    Natural agreement (before resolver)  {str(ov['natural_agree_pct']):>5}%                   ║
║    Conflict rate BEFORE resolver        {str(ov['conflict_B_before_pct']):>5}%  ({ov['n_conflicts_B_before']} frames)   ║
║    Conflict rate AFTER  resolver        {str(ov['conflict_B_after_pct']):>5}%  ({ov['n_conflicts_B_after']} frames)   ║
║    Conflicts resolved                   {ov['n_conflicts_resolved']:>3}    ({ov['resolution_rate_pct']}%)           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  (C) Ablation: peer-symmetric, no resolver (ego always yields)              ║
║    Conflict rate AFTER ablation         {str(ov['conflict_C_pct']):>5}%  ({ov['n_conflicts_C']} frames)   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Conflict B breakdown by scenario (before resolver):                        ║
║    {sc_str:<72} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Resolver rule distribution (% of all frames):                              ║
║    Agreement (no intervention)  {str(ov['rule_agreement_pct']):>5}%                          ║
║    No physical conflict         {str(ov['rule_no_conflict_pct']):>5}%                          ║
║    TTC (faster CAV yields)      {str(ov['rule_ttc_pct']):>5}%                          ║
║    Deterministic tiebreak       {str(ov['rule_tiebreak_pct']):>5}%                          ║
╚══════════════════════════════════════════════════════════════════════════════╝""")


if __name__ == '__main__':
    main()
