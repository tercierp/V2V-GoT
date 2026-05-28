#!/usr/bin/env python3
"""
Transform nq8 and nq9 QA datasets to use cav1's own reference frame.

Current state: all coordinates in the QA text are in EGO's frame, even for
records where asker_cav_id='1'. This means:
  - CAV_1 appears at (-75.7, 5.2) instead of (0.0, 0.0)
  - CAV_1's planned trajectory goes in negative-x direction (toward ego)
  - Notable objects are at ego-relative positions

After this script:
  - CAV_1 asker records have it at (0.0, 0.0)
  - Planned trajectory uses future_trajectory_str_in_self (cav1's own frame)
  - All (x, z) coordinates in the question are transformed to cav1's frame
  - GT answer trajectory (nq9) also uses the in_self version

Ego asker records are left unchanged.

Limitation: the visual feature maps fed to the LLM are always stacked as
[ego_features, cav1_features]; that ordering asymmetry is not addressed here
and would require re-running the perception encoder.

Usage:
    python scripts/gen_cav1_qa.py \\
        --nq8_gt  <path>/nq8sm3w6dc.json \\
        --nq9_gt  <path>/nq9sm3w6dc.json \\
        --output_dir outputs/cav1_perspective
"""

import argparse
import json
import numpy as np
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                '..', 'DMSTrack', 'V2V4Real'))
from opencood.utils.transformation_utils import x1_to_x2


# ── Coordinate helpers ─────────────────────────────────────────────────────────

def _build_transform(record: dict) -> np.ndarray:
    """
    Return 4×4 matrix that transforms a point from EGO's local frame to
    CAV1's local frame, built from the lidar poses stored in the QA record.
    """
    pose_ego  = np.array(record['cav_ego_lidar_pose']).reshape(4, 4)
    pose_cav1 = np.array(record['cav_1_lidar_pose']).reshape(4, 4)
    # x1_to_x2(ego, cav1) = inv(pose_cav1) @ pose_ego
    # applied to [x, y, 0, 1] in ego's frame → [x', y', *, 1] in cav1's frame
    return x1_to_x2(pose_ego, pose_cav1)


def _transform_point(T: np.ndarray, x: float, z: float) -> tuple[float, float]:
    """
    Transform a 2-D ground-plane point (x=forward, z=lateral) from ego's
    frame to cav1's frame.  The QA convention: (row0, row1) of the translation.
    """
    p = T @ np.array([x, z, 0.0, 1.0])
    return round(float(p[0]), 1), round(float(p[1]), 1)


def _transform_coord_str(T: np.ndarray, coord_str: str) -> str:
    """
    Find every (x,z) pair in a string and replace with cav1-frame coordinates.
    Handles both single points "(x,z)" and trajectory lists "[(x,z),...]".
    """
    def _replace(m: re.Match) -> str:
        x = float(m.group(1))
        z = float(m.group(2))
        nx, nz = _transform_point(T, x, z)
        return f'({nx},{nz})'

    return re.sub(r'\((-?[\d.]+),(-?[\d.]+)\)', _replace, coord_str)


def _parse_traj_str(s: str) -> list[tuple[float, float]]:
    pairs = re.findall(r'\((-?[\d.]+),(-?[\d.]+)\)', s)
    return [(float(a), float(b)) for a, b in pairs]


def _fmt_traj(pts: list[tuple[float, float]]) -> str:
    return '[' + ','.join(f'({x},{z})' for x, z in pts) + ']'


# ── Per-record transformation ──────────────────────────────────────────────────

def transform_cav1_record_nq8(record: dict) -> dict:
    """
    For a cav1-asker nq8 record: rewrite the question so that cav1 is at
    (0,0) and all coordinates are in cav1's own frame.

    Order of operations matters:
    1. Transform ALL coordinates in the question to cav1's frame first
       (this includes asker position, ego-frame trajectory, object positions).
    2. Explicitly overwrite the asker position with (0.0,0.0) — it should
       be ~zero after the transform, but floating-point noise makes it cleaner
       to hard-code.
    3. Replace the planned trajectory with the exact pre-computed in_self string
       (the transform produces the same result numerically, but in_self avoids
       any rounding divergence).
    """
    rec = {k: v for k, v in record.items()}
    T   = _build_transform(rec)
    in_self = rec['future_trajectory_str_in_self']

    question = rec['conversations'][0]['value']

    # 1. Transform ALL (x,z) coordinates → cav1's frame
    question = _transform_coord_str(T, question)

    # 2. Hard-set asker position to (0.0,0.0)
    question = re.sub(
        r'I am CAV_1 at \((-?[\d.]+),(-?[\d.]+)\)\.',
        'I am CAV_1 at (0.0,0.0).',
        question
    )

    # 3. Overwrite the planned trajectory with the exact in_self string
    question = re.sub(
        r'my planned future trajectory is \[[^\]]+\]',
        f'my planned future trajectory is {in_self}',
        question
    )

    rec['conversations'] = [
        {'from': 'human', 'value': question},
        rec['conversations'][1],   # GT answer is speed/steering text — no coords
    ]
    rec['reference_frame'] = 'cav1'
    return rec


def transform_cav1_record_nq9(record: dict) -> dict:
    """
    For a cav1-asker nq9 record: rewrite question and GT answer so that cav1
    is at (0,0).  nq9's context is speed/steering text only (no coordinates),
    so only the asker position header and GT trajectory need changing.
    """
    rec = {k: v for k, v in record.items()}
    in_self = rec['future_trajectory_str_in_self']

    question = rec['conversations'][0]['value']

    # Replace asker position — nq9 context has no other coordinates
    question = re.sub(
        r'I am CAV_1 at \((-?[\d.]+),(-?[\d.]+)\)\.',
        'I am CAV_1 at (0.0,0.0).',
        question
    )

    # GT answer: use in_self trajectory
    gt_answer = f'The suggested future trajectory is {in_self}.'

    rec['conversations'] = [
        {'from': 'human', 'value': question},
        {'from': 'gpt',   'value': gt_answer},
    ]
    rec['reference_frame'] = 'cav1'
    return rec


# ── Main ───────────────────────────────────────────────────────────────────────

def transform_dataset(records: list, nq: str) -> tuple[list, dict]:
    """
    Return (transformed_records, stats).
    Ego records are copied unchanged; cav1 records are rewritten.
    """
    out, stats = [], {'ego_kept': 0, 'cav1_transformed': 0, 'errors': 0}

    for rec in records:
        if rec['asker_cav_id'] == 'ego':
            out.append(rec)
            stats['ego_kept'] += 1
        else:
            try:
                if nq == 'nq8':
                    out.append(transform_cav1_record_nq8(rec))
                else:
                    out.append(transform_cav1_record_nq9(rec))
                stats['cav1_transformed'] += 1
            except Exception as e:
                out.append(rec)   # fall back to original on error
                stats['errors'] += 1
                print(f'  WARNING: ts={rec["global_timestamp_index"]} error: {e}')

    return out, stats


def main():
    PLAY = ('/home/tercier/v2v-got-backup-20260514/LLaVA/playground/data/eval/'
            'v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2_4330_full')

    parser = argparse.ArgumentParser()
    parser.add_argument('--nq8_gt',
        default=f'{PLAY}_nq8sm3w6dc/answers/val/llava-v1.5-7b/nq8sm3w6dc.json')
    parser.add_argument('--nq9_gt',
        default=f'{PLAY}_nq9sm3w6dc/answers/val/llava-v1.5-7b/nq9sm3w6dc.json')
    parser.add_argument('--output_dir', default='outputs/cav1_perspective')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for path, nq, out_name in [
        (args.nq8_gt, 'nq8', 'nq8sm3w6dc_cav1ref.json'),
        (args.nq9_gt, 'nq9', 'nq9sm3w6dc_cav1ref.json'),
    ]:
        print(f'Processing {nq}...')
        with open(path) as f:
            records = json.load(f)

        out, stats = transform_dataset(records, nq)

        out_path = os.path.join(args.output_dir, out_name)
        with open(out_path, 'w') as f:
            json.dump(out, f)

        print(f'  ego kept:          {stats["ego_kept"]}')
        print(f'  cav1 transformed:  {stats["cav1_transformed"]}')
        print(f'  errors (fallback): {stats["errors"]}')
        print(f'  saved → {out_path}')

        # sanity-check one cav1 record
        sample = next(r for r in out if r.get('asker_cav_id') == '1')
        q = sample['conversations'][0]['value']
        print(f'  sample question start: {q[:120]}')
        print()


if __name__ == '__main__':
    main()
