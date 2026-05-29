#!/usr/bin/env python3
"""
Build an injected Q8 or Q9 eval JSON file for re-inference experiments.

Supports multiple source signals (q6_compact, q7_compact, q9_full, q9_endpoint)
and multiple target layers (q8, q9), with selective per-frame injection using
scope modes (all, ids, conflict_before, conflict_after).

For subset scopes, this script outputs ONLY the matched records. After inference,
use merge_llm_outputs.py to merge the subset LLM outputs back into the full
original LLM output before running phase4_eval.py.

Usage examples:

  # Dry-run: q9_full → q8, 5 records
  python scripts/build_injected_experiment.py \\
    --target_json $NQ8_GT --target_q q8 \\
    --source_signal q9_full --source_json $NQ9_GT --source_llm $ORIG_Q9_LLM \\
    --source_matching_mode other_asker \\
    --max_records 5 --dry_run

  # q9_full → q8, conflict frames only (full run)
  python scripts/build_injected_experiment.py \\
    --target_json $NQ8_GT --target_q q8 \\
    --source_signal q9_full --source_json $NQ9_GT --source_llm $ORIG_Q9_LLM \\
    --source_matching_mode other_asker \\
    --scope conflict_before \\
    --phase4_per_frame outputs/phase4/per_frame_notraj.json \\
    --out outputs/experiments/e1_q9full_to_q8_conflict/nq8_injected.json

  # q7_compact → q8, all frames
  python scripts/build_injected_experiment.py \\
    --target_json $NQ8_GT --target_q q8 \\
    --source_signal q7_compact --source_json $NQ7_GT --source_llm $ORIG_Q7_LLM \\
    --source_matching_mode same_asker \\
    --scope all \\
    --out outputs/experiments/e3_q7compact_to_q8_all/nq8_injected.json
"""

import argparse
import csv
import io
import json
import math
import os
import re
import sys
from collections import defaultdict

# ── Constants ─────────────────────────────────────────────────────────────────

_NEW_MARKER = "Peer coordination signal:"
_ALL_MARKERS = (
    "Neighbor intent information from Q6:",  # old raw
    "Peer intent signal:",                   # old compact q6
    "Peer coordination signal:",             # new unified
)

_INJECT_Q8_TEMPLATE = (
    "\n\nPeer coordination signal:\n"
    "{signal_block}\n"
    "Use this only to choose safe speed and steering.\n"
    "If the peer plan conflicts with my nominal path, prefer slowing/yielding "
    "over maintaining speed.\n"
    "Answer with exactly one sentence containing:\n"
    "The suggested speed setting is:\n"
    "The suggested steering setting is:"
)

_INJECT_Q9_TEMPLATE = (
    "\n\nPeer coordination signal:\n"
    "{signal_block}\n"
    "Use this only for collision avoidance.\n"
    "Predict only {self_id}'s own future trajectory.\n"
    "Do not describe the peer vehicle.\n"
    "Do not copy the peer trajectory.\n"
    "Answer with exactly one sentence starting with:\n"
    "The suggested future trajectory is"
)

_OPPOSITE = {'ego': '1', '1': 'ego'}


# ── Helpers copied from build_q9_injected.py ──────────────────────────────────

def _deep_copy(rec):
    return json.loads(json.dumps(rec))


def _normalize(text, max_chars):
    text = ' '.join(text.split())
    return text[:max_chars].rstrip() if len(text) > max_chars else text


def parse_q6_intent(q6_text):
    """Parse Q6-style natural language into compact intent fields.

    Returns (intent_dict, missing_set) where intent_dict has keys:
        neighbor_id, neighbor_position, neighbor_next_waypoint, neighbor_motion
    """
    _UNKNOWN = "unknown"
    fields = {
        'neighbor_id':            _UNKNOWN,
        'neighbor_position':      _UNKNOWN,
        'neighbor_next_waypoint': _UNKNOWN,
        'neighbor_motion':        _UNKNOWN,
    }
    try:
        text = ' '.join(q6_text.split())

        m = re.search(r'\b(CAV_(?:EGO|\d+))\b', text)
        if m:
            fields['neighbor_id'] = m.group(1)

        m = re.search(r'is at \(([^)]+)\)', text)
        if m:
            fields['neighbor_position'] = '(' + m.group(1).replace(' ', '') + ')'

        if re.search(r'stay(?:ing|s) at the same location|stationary|stopped',
                     text, re.IGNORECASE):
            fields['neighbor_motion'] = 'stationary'
        else:
            m = re.search(r'moving ([^.]+)', text)
            if m:
                fields['neighbor_motion'] = m.group(1).split()[0]

        m = re.search(r'planned future trajectory is \[\(([^)]+)\)', text)
        if m:
            fields['neighbor_next_waypoint'] = '(' + m.group(1).replace(' ', '') + ')'

    except Exception:
        pass

    missing = {k for k, v in fields.items() if v == _UNKNOWN}
    return fields, missing


# ── New parsers ───────────────────────────────────────────────────────────────

def _parse_last_waypoint(bracket_content):
    """Parse last (x,z) pair from a trajectory bracket-content string."""
    pairs = re.findall(r'\(([^)]+)\)', bracket_content)
    if not pairs:
        return None
    last = pairs[-1]
    parts = last.split(',')
    if len(parts) == 2:
        try:
            x = float(parts[0].strip())
            z = float(parts[1].strip())
            return f'({x},{z})'
        except ValueError:
            pass
    return None


def parse_q7_compact(gt_prompt, llm_text):
    """Parse Q7 GT prompt + Q7 LLM text into compact signal fields.

    From gt_prompt: extract peer CAV id, position, and planned endpoint.
      - "CAV_1 is at (x,z). Its planned future trajectory is [...]"
      - Do NOT use "if my planned future trajectory is [...]" — that is the asker's own.
    From llm_text: extract n_objects, first object position/motion/predicted_endpoint.

    Returns (fields_dict, missing_set).
    """
    _UNKNOWN = "unknown"
    fields = {
        'peer_id':                _UNKNOWN,
        'peer_position':          _UNKNOWN,
        'peer_planned_endpoint':  _UNKNOWN,
        'n_objects':              _UNKNOWN,
        'obj_position':           _UNKNOWN,
        'obj_motion':             _UNKNOWN,
        'obj_predicted_endpoint': _UNKNOWN,
    }

    # --- Parse GT prompt for peer CAV ---
    try:
        prompt_text = ' '.join(gt_prompt.split())
        # Find "CAV_X is at (...)." — this is the peer CAV context embedded in Q7 prompt
        m = re.search(r'\b(CAV_(?:EGO|\d+))\s+is at \(([^)]+)\)', prompt_text)
        if m:
            fields['peer_id'] = m.group(1)
            fields['peer_position'] = '(' + m.group(2).replace(' ', '') + ')'

        # "Its planned future trajectory is [...]" — peer's plan
        m = re.search(r'Its planned future trajectory is \[([^\]]+)', prompt_text)
        if m:
            wp = _parse_last_waypoint(m.group(1))
            if wp:
                fields['peer_planned_endpoint'] = wp
    except Exception:
        pass

    # --- Parse LLM answer for dynamic objects ---
    try:
        llm = ' '.join(llm_text.split())

        if re.search(r'no notable object', llm, re.IGNORECASE):
            fields['n_objects'] = '0'
        else:
            count = len(re.findall(r'There is a car at', llm, re.IGNORECASE))
            fields['n_objects'] = str(count) if count > 0 else _UNKNOWN

            m = re.search(r'There is a car at \(([^)]+)\)', llm, re.IGNORECASE)
            if m:
                fields['obj_position'] = '(' + m.group(1).replace(' ', '') + ')'

            if re.search(r'stay(?:ing|s) at the same location|stationary|stopped',
                         llm, re.IGNORECASE):
                fields['obj_motion'] = 'stationary'
            else:
                m2 = re.search(r'(moving|turning)\s+([a-z]+)', llm, re.IGNORECASE)
                if m2:
                    fields['obj_motion'] = m2.group(2).lower()

            m = re.search(r'predicted future trajectory is \[([^\]]+)', llm, re.IGNORECASE)
            if m:
                wp = _parse_last_waypoint(m.group(1))
                if wp:
                    fields['obj_predicted_endpoint'] = wp
    except Exception:
        pass

    missing = {k for k, v in fields.items() if v == _UNKNOWN}
    return fields, missing


def parse_q9_trajectory(text):
    """Parse trajectory waypoints from Q9-style LLM text.

    Uses the same regex pattern as phase4_eval.parse_trajectory.
    Returns list of (x, z) float tuples.
    """
    try:
        pairs = re.findall(r'\(([^)]+)\)', text)
        result = []
        for p in pairs:
            parts = p.split(',')
            if len(parts) == 2:
                try:
                    result.append((float(parts[0].strip()), float(parts[1].strip())))
                except ValueError:
                    pass
        return result
    except Exception:
        return []


# ── Data loading ──────────────────────────────────────────────────────────────

def load_source_gt(path):
    """Load source GT JSON → {(global_timestamp_index, asker_cav_id): record}."""
    with open(path) as f:
        data = json.load(f)
    index = {}
    for r in data:
        key = (r['global_timestamp_index'], r['asker_cav_id'])
        index[key] = r
    return index


def load_source_llm(path):
    """Load source LLM merge.jsonl → {question_id: text_string}."""
    index = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get('question_id', rec.get('id'))
            text = rec.get('text', rec.get('outputs', ''))
            index[qid] = text
    return index


# ── Scope logic ───────────────────────────────────────────────────────────────

def _load_ids_file(path):
    """Load a set of record IDs from a file.

    Supports:
    - Plain text: one ID per line
    - CSV: auto-detected if first non-comment line contains a comma;
           uses column named 'id' or 'question_id'
    """
    with open(path) as f:
        raw = f.read()

    lines = [l.strip() for l in raw.splitlines()
             if l.strip() and not l.strip().startswith('#')]
    if not lines:
        return set()

    # Detect CSV
    if ',' in lines[0]:
        reader = csv.DictReader(io.StringIO(raw))
        ids = set()
        for row in reader:
            val = row.get('id') or row.get('question_id')
            if val and val.strip():
                ids.add(val.strip())
        return ids

    return {l for l in lines}


def get_scope_ids(target_records, scope, ids_file, phase4_per_frame):
    """Compute the set of target record IDs to include.

    Returns None for scope='all' (include everything).
    For subset scopes, returns a set of record IDs.
    Also returns a dict of scope_stats for printing.
    """
    scope_stats = {}

    if scope == 'all':
        return None, scope_stats

    if scope == 'ids':
        if not ids_file:
            print("ERROR: --ids_file is required for --scope ids", file=sys.stderr)
            sys.exit(1)
        ids = _load_ids_file(ids_file)
        scope_stats['ids_file_count'] = len(ids)
        return ids, scope_stats

    # conflict_before or conflict_after
    if not phase4_per_frame:
        print(
            "ERROR: --phase4_per_frame is required for "
            f"--scope {scope}.\n"
            "       Provide the path to outputs/phase4/per_frame_notraj.json "
            "(or similar).\n"
            "       Alternatively, extract conflict timestamps manually and use "
            "--scope ids --ids_file.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(phase4_per_frame) as f:
        per_frame = json.load(f)

    conflict_field = 'conflict_B_before' if scope == 'conflict_before' else 'conflict_B_after'
    conflict_ts = {r['ts'] for r in per_frame if r.get(conflict_field)}

    # Build ts → list of record IDs from target
    ts_to_ids = defaultdict(list)
    for rec in target_records:
        ts = rec.get('global_timestamp_index')
        rec_id = rec.get('id')
        if ts is not None and rec_id is not None:
            ts_to_ids[ts].append(rec_id)

    scope_ids = set()
    counts_per_ts = {}
    for ts in conflict_ts:
        ids_for_ts = ts_to_ids.get(ts, [])
        scope_ids.update(ids_for_ts)
        counts_per_ts[ts] = len(ids_for_ts)

    scope_stats['conflict_timestamps'] = len(conflict_ts)
    scope_stats['selected_target_records'] = len(scope_ids)
    scope_stats['counts_per_ts'] = counts_per_ts

    return scope_ids, scope_stats


# ── Signal block builders ─────────────────────────────────────────────────────

def _format_traj(waypoints):
    """Format a list of (x,z) tuples as compact string."""
    return '[' + ','.join(f'({x},{z})' for x, z in waypoints) + ']'


def make_signal_block(source_signal, source_rec, source_text, self_id,
                      injection_format, q9_parse_stats):
    """Build (signal_block_str, preview_str, skipped_reason).

    Returns (None, None, reason) when the signal cannot be built.
    q9_parse_stats is a mutable dict updated in-place for q9_full/q9_endpoint.
    """
    if source_signal == 'none':
        return None, None, None

    if source_signal == 'q6_compact':
        if injection_format == 'raw':
            block = source_text
            preview = source_text[:160]
        else:
            intent, missing = parse_q6_intent(source_text)
            if len(missing) == 4:
                return None, None, 'parse_failed'
            block = (
                f"neighbor_id={intent['neighbor_id']}; "
                f"neighbor_position={intent['neighbor_position']}; "
                f"neighbor_next_waypoint_in_ego_frame={intent['neighbor_next_waypoint']}; "
                f"neighbor_motion={intent['neighbor_motion']}"
            )
            preview = (
                f"neighbor_id={intent['neighbor_id']}; "
                f"pos={intent['neighbor_position']}; "
                f"next={intent['neighbor_next_waypoint']}; "
                f"motion={intent['neighbor_motion']}"
            )
        return block, preview, None

    if source_signal == 'q7_compact':
        gt_prompt = ''
        try:
            gt_prompt = source_rec['conversations'][0]['value']
        except (KeyError, IndexError, TypeError):
            pass

        fields, missing = parse_q7_compact(gt_prompt, source_text)

        # q7 is still useful even with partial info; only skip if all GT fields missing
        gt_fields = {'peer_id', 'peer_position', 'peer_planned_endpoint'}
        if gt_fields.issubset(missing):
            return None, None, 'parse_failed'

        parts = []
        for key in ('peer_id', 'peer_position', 'peer_planned_endpoint',
                    'n_objects', 'obj_position', 'obj_motion', 'obj_predicted_endpoint'):
            val = fields[key]
            if val != 'unknown':
                parts.append(f"{key}={val}")
        block = '; '.join(parts)
        preview = block[:160]
        return block, preview, None

    if source_signal in ('q9_full', 'q9_endpoint'):
        waypoints = parse_q9_trajectory(source_text)
        if not waypoints:
            q9_parse_stats['failed'] = q9_parse_stats.get('failed', 0) + 1
            return None, None, 'parse_failed'

        q9_parse_stats['ok'] = q9_parse_stats.get('ok', 0) + 1
        q9_parse_stats['total_pts'] = q9_parse_stats.get('total_pts', 0) + len(waypoints)

        # neighbor_id is derived from the source record's asker_cav_id
        # (opposite of the current target's asker, selected upstream)
        src_cav = source_rec.get('asker_cav_id', 'unknown')
        neighbor_id = 'CAV_EGO' if src_cav == 'ego' else 'CAV_1'

        if source_signal == 'q9_full':
            traj_str = _format_traj(waypoints)
            block = (
                f"neighbor_id={neighbor_id}; "
                f"neighbor_future_trajectory={traj_str}; "
                f"source=q9_full"
            )
            preview = f"neighbor_id={neighbor_id}; traj={traj_str[:80]}"
        else:  # q9_endpoint
            last = waypoints[-1]
            wp_str = f'({last[0]},{last[1]})'
            block = (
                f"neighbor_id={neighbor_id}; "
                f"neighbor_final_waypoint={wp_str}; "
                f"source=q9_endpoint"
            )
            preview = f"neighbor_id={neighbor_id}; final_wp={wp_str}"

        return block, preview, None

    return None, None, f'unknown_signal_{source_signal}'


# ── Injection text builder ────────────────────────────────────────────────────

def make_injection(signal_block, target_q, self_id):
    """Wrap signal_block in the appropriate template for target_q."""
    if target_q == 'q8':
        return _INJECT_Q8_TEMPLATE.format(signal_block=signal_block)
    else:  # q9
        return _INJECT_Q9_TEMPLATE.format(signal_block=signal_block, self_id=self_id)


# ── Prompt validation ─────────────────────────────────────────────────────────

def _validate_target_prompt(prompt, target_q):
    """Return None if prompt looks valid for target_q, else a reason string."""
    lower = prompt.lower()
    if target_q == 'q8':
        # Accept "speed and steering settings", "speed setting", "speed/steering", etc.
        if 'speed' not in lower or 'steering' not in lower:
            return 'missing_speed_or_steering_in_prompt'
    else:  # q9
        if 'future trajectory' not in lower and 'trajectory' not in lower:
            return 'missing_future_trajectory_in_prompt'
    return None


# ── Main build loop ───────────────────────────────────────────────────────────

def build_records(target_records, source_gt, source_llm, scope_ids, args):
    """Iterate over target_records and inject where applicable.

    For scope_ids=None (all): every input record appears in result.
    For subset scopes: only scope-included records appear in result.

    Returns (results, stats, examples).
    """
    stats = {
        'total': len(target_records),
        'injected': 0,
        'skipped_scope': 0,
        'skipped_already_injected': 0,
        'missing_source_gt': 0,
        'missing_source_llm': 0,
        'parse_failed': 0,
        'invalid_target': 0,
        'output_records': 0,
    }
    q9_parse_stats = {}  # ok, failed, total_pts

    result = []
    examples = {'ego': None, 'cav1': None}

    for raw in target_records:
        rec = _deep_copy(raw)

        # --- validate basic structure ---
        try:
            ts = rec['global_timestamp_index']
            cid = rec['asker_cav_id']
            prompt = rec['conversations'][0]['value']
            assert isinstance(prompt, str)
            rec_id = rec['id']
        except (KeyError, IndexError, TypeError, AssertionError):
            stats['invalid_target'] += 1
            rec = _stamp(rec, False, args, skipped_reason='invalid_record',
                         source_qid=None, preview=None)
            if scope_ids is None:
                result.append(rec)
            continue

        # --- scope filter ---
        if scope_ids is not None and rec_id not in scope_ids:
            stats['skipped_scope'] += 1
            continue  # subset mode: don't include this record

        # --- prompt validity ---
        bad = _validate_target_prompt(prompt, args.target_q)
        if bad:
            stats['invalid_target'] += 1
            rec = _stamp(rec, False, args, skipped_reason='invalid_record',
                         source_qid=None, preview=None)
            result.append(rec)
            continue

        # --- reinject guard ---
        if not args.allow_reinject and any(m in prompt for m in _ALL_MARKERS):
            stats['skipped_already_injected'] += 1
            rec = _stamp(rec, False, args, skipped_reason='already_injected',
                         source_qid=None, preview=None)
            result.append(rec)
            continue

        # --- source lookup ---
        source_qid = None
        source_rec = None
        source_text = None

        if args.source_signal != 'none':
            if args.source_matching_mode == 'same_asker':
                src_key = (ts, cid)
            else:
                other = _OPPOSITE.get(cid)
                if other is None:
                    stats['invalid_target'] += 1
                    rec = _stamp(rec, False, args, skipped_reason='invalid_record',
                                 source_qid=None, preview=None)
                    result.append(rec)
                    continue
                src_key = (ts, other)

            source_rec = source_gt.get(src_key) if source_gt else None
            if source_rec is None:
                stats['missing_source_gt'] += 1
                rec = _stamp(rec, False, args, skipped_reason='missing_source_gt',
                             source_qid=None, preview=None)
                result.append(rec)
                continue

            source_qid = source_rec.get('id')
            source_text_raw = source_llm.get(source_qid) if source_llm else None
            if source_text_raw is None:
                stats['missing_source_llm'] += 1
                rec = _stamp(rec, False, args, skipped_reason='missing_source_llm',
                             source_qid=source_qid, preview=None)
                result.append(rec)
                continue

            source_text = _normalize(source_text_raw, 800)

        # --- build signal ---
        self_id = 'CAV_EGO' if cid == 'ego' else 'CAV_1'
        signal_block, preview, skip_reason = make_signal_block(
            args.source_signal, source_rec, source_text, self_id,
            args.injection_format, q9_parse_stats,
        )
        if skip_reason:
            stats['parse_failed'] += 1
            rec = _stamp(rec, False, args, skipped_reason=skip_reason,
                         source_qid=source_qid, preview=None)
            result.append(rec)
            continue

        # --- inject ---
        if args.source_signal == 'none':
            # passthrough — no injection
            rec = _stamp(rec, False, args, skipped_reason=None,
                         source_qid=None, preview=None)
        else:
            injection = make_injection(signal_block, args.target_q, self_id)
            rec['conversations'][0]['value'] += injection
            rec = _stamp(rec, True, args, skipped_reason=None,
                         source_qid=source_qid, preview=preview)
            stats['injected'] += 1

        result.append(rec)
        stats['output_records'] += 1

        if cid == 'ego' and examples['ego'] is None:
            examples['ego'] = (rec_id, source_qid, rec['conversations'][0]['value'])
        if cid == '1' and examples['cav1'] is None:
            examples['cav1'] = (rec_id, source_qid, rec['conversations'][0]['value'])

    # Update output count for scope=all (includes non-injected records)
    if scope_ids is None:
        stats['output_records'] = len(result)

    stats['q9_parse_stats'] = q9_parse_stats
    return result, stats, examples


def _stamp(rec, enabled, args, skipped_reason, source_qid, preview):
    rec['injection_enabled'] = enabled
    rec['injection_target_q'] = args.target_q
    rec['injection_source_signal'] = args.source_signal
    rec['injection_source_qid'] = source_qid
    rec['injection_source_matching_mode'] = args.source_matching_mode
    rec['injection_scope'] = args.scope
    rec['injection_preview'] = preview
    rec['injection_skipped_reason'] = skipped_reason
    return rec


# ── Validation ────────────────────────────────────────────────────────────────

def validate_output(original_records, result_records, scope):
    original_ids = [r.get('id') for r in original_records]
    result_ids = [r.get('id') for r in result_records]

    if scope == 'all':
        if len(result_records) != len(original_records):
            raise AssertionError(
                f"Record count changed: input={len(original_records)}, "
                f"output={len(result_records)}"
            )
        for i, (oid, rid) in enumerate(zip(original_ids, result_ids)):
            if oid != rid:
                raise AssertionError(
                    f"Record {i}: id changed from {oid!r} to {rid!r}"
                )
    else:
        original_id_set = set(original_ids)
        for rid in result_ids:
            if rid not in original_id_set:
                raise AssertionError(
                    f"Result record id {rid!r} not found in original records"
                )
        if len(result_ids) != len(set(result_ids)):
            raise AssertionError("Duplicate IDs in result records")


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_scope_stats(scope, scope_stats):
    if scope in ('conflict_before', 'conflict_after'):
        print(f"  conflict_timestamps      : {scope_stats.get('conflict_timestamps', 'n/a')}")
        print(f"  selected_target_records  : {scope_stats.get('selected_target_records', 'n/a')}")
        counts = scope_stats.get('counts_per_ts', {})
        bad = {ts: n for ts, n in counts.items() if n != 2}
        if bad:
            print(f"  WARNING: {len(bad)} timestamps with != 2 records (expected ego+cav1): "
                  f"{dict(list(bad.items())[:5])}")
        else:
            print(f"  Records-per-timestamp    : all == 2 (ego + cav1 as expected)")
    elif scope == 'ids':
        print(f"  ids_file_count           : {scope_stats.get('ids_file_count', 'n/a')}")


def print_stats(stats, args, scope_stats):
    is_subset = args.scope != 'all'
    mode_str = (f"subset output (scope={args.scope})"
                if is_subset else "full output (scope=all)")
    print()
    print(f"Output mode: {mode_str}")
    if is_subset:
        print("  -> Use merge_llm_outputs.py to merge subset LLM outputs back before phase4_eval.")
    print()
    print_scope_stats(args.scope, scope_stats)
    print()
    print("=== Build statistics ===")
    print(f"  total_target_records     : {stats['total']}")
    print(f"  injected                 : {stats['injected']}")
    print(f"  skipped_scope            : {stats['skipped_scope']}")
    print(f"  skipped_already_injected : {stats['skipped_already_injected']}")
    print(f"  missing_source_gt        : {stats['missing_source_gt']}")
    print(f"  missing_source_llm       : {stats['missing_source_llm']}")
    print(f"  parse_failed             : {stats['parse_failed']}")
    print(f"  invalid_target           : {stats['invalid_target']}")
    print(f"  output_records           : {stats['output_records']}")

    q9s = stats.get('q9_parse_stats', {})
    if q9s:
        ok = q9s.get('ok', 0)
        failed = q9s.get('failed', 0)
        total_pts = q9s.get('total_pts', 0)
        mean_pts = round(total_pts / ok, 2) if ok else 'n/a'
        print()
        print("=== Q9 parse statistics ===")
        print(f"  q9_parse_ok              : {ok}")
        print(f"  q9_parse_failed          : {failed}")
        print(f"  mean_trajectory_points   : {mean_pts}")


def print_examples(examples, args):
    labels = [
        ('ego',  'ego CAV   (asker_cav_id="ego")'),
        ('cav1', 'cav1 CAV  (asker_cav_id="1")'),
    ]
    print()
    print("=" * 72)
    print(f"Examples  [target_q={args.target_q}  source_signal={args.source_signal}  "
          f"matching_mode={args.source_matching_mode}]")
    print("=" * 72)
    for key, label in labels:
        ex = examples[key]
        if ex is None:
            print(f"\n[{label}]  — no injected record found")
            continue
        rec_id, src_qid, full_prompt = ex
        print(f"\n[{label}]")
        print(f"  Target record id  : {rec_id}")
        print(f"  Source record id  : {src_qid}")
        tail = full_prompt[-600:]
        print(f"  Modified prompt (last 600 chars):\n"
              f"  ···{tail!r}")
    print("=" * 72)


def print_next_steps(out_path, dry_run):
    if dry_run:
        print("\nDry run complete — output NOT written.")
    else:
        print(f"\nWrote output → {out_path}")

    out_dir = os.path.dirname(os.path.abspath(out_path)) if out_path else '<out_dir>'
    print()
    print("=== Next step: run inference ===")
    print("python -m llava.eval.model_vqa_loader \\")
    print(f"  --model-path <checkpoint> \\")
    print(f"  --question-file {out_path} \\")
    print(f"  --image-folder <image_dir> \\")
    print(f"  --answers-file {out_dir}/merge.jsonl \\")
    print(f"  --temperature 0 --conv-mode vicuna_v1")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Build an injected Q8 or Q9 eval JSON for re-inference experiments. '
            'Supports multiple source signals (q6_compact, q7_compact, q9_full, '
            'q9_endpoint) and scope modes (all, ids, conflict_before, conflict_after). '
            'For subset scopes, output contains only matched records — use '
            'merge_llm_outputs.py to merge back after inference.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--target_json', required=True,
                        help='Original Q8 or Q9 GT eval JSON to inject into')
    parser.add_argument('--target_q', required=True, choices=['q8', 'q9'],
                        help='Target layer: q8 (speed/steering) or q9 (trajectory)')
    parser.add_argument('--out', required=True,
                        help='Output path for the injected JSON')
    parser.add_argument('--source_signal',
                        choices=['q6_compact', 'q7_compact', 'q9_full', 'q9_endpoint', 'none'],
                        default='none',
                        help='Source signal type to inject')
    parser.add_argument('--source_json',
                        help='Source layer GT JSON (required unless source_signal=none)')
    parser.add_argument('--source_llm',
                        help='Source layer LLM merge.jsonl (required unless source_signal=none)')
    parser.add_argument('--source_matching_mode',
                        choices=['same_asker', 'other_asker'], default=None,
                        help=(
                            'How to match source to target. '
                            'same_asker (default for q6/q7): same asker_cav_id. '
                            'other_asker (default for q9_full/q9_endpoint): opposite asker.'
                        ))
    parser.add_argument('--scope',
                        choices=['all', 'ids', 'conflict_before', 'conflict_after'],
                        default='all',
                        help='Which records to inject into')
    parser.add_argument('--ids_file',
                        help='File with one record id per line (for --scope ids)')
    parser.add_argument('--phase4_per_frame',
                        help='per_frame_notraj.json path (for --scope conflict_before/after)')
    parser.add_argument('--max_records', type=int, default=None,
                        help='Debug limit: process at most N target records')
    parser.add_argument('--injection_format', choices=['compact', 'raw'], default='compact',
                        help='compact (default) or raw (debugging only)')
    parser.add_argument('--allow_reinject', action='store_true',
                        help='Allow injection even when an injection marker is already present')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite --out if it already exists')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print examples and stats; do not write output file')
    args = parser.parse_args()

    # --- validate arg combinations ---
    if args.source_signal != 'none':
        if not args.source_json:
            parser.error('--source_json is required unless --source_signal none')
        if not args.source_llm:
            parser.error('--source_llm is required unless --source_signal none')

    if args.injection_format == 'raw' and args.source_signal not in ('q6_compact', 'q7_compact'):
        parser.error('--injection_format raw is only supported for q6_compact and q7_compact')

    # --- apply matching_mode defaults ---
    if args.source_matching_mode is None:
        if args.source_signal in ('q6_compact', 'q7_compact', 'none'):
            args.source_matching_mode = 'same_asker'
        else:  # q9_full, q9_endpoint
            args.source_matching_mode = 'other_asker'
        print(f"source_matching_mode defaulted to: {args.source_matching_mode}")

    # --- refuse overwrite ---
    if not args.dry_run and os.path.exists(args.out) and not args.force:
        print(f"ERROR: output file already exists: {args.out}", file=sys.stderr)
        print("       Pass --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    # --- load target ---
    print(f"Loading target JSON  : {args.target_json}")
    with open(args.target_json) as f:
        target_records = json.load(f)
    print(f"  {len(target_records)} records")

    if args.max_records:
        target_records = target_records[:args.max_records]
        print(f"  Limiting to {len(target_records)} records (--max_records)")

    # --- load source ---
    source_gt = None
    source_llm = None
    if args.source_signal != 'none':
        print(f"Loading source GT    : {args.source_json}")
        source_gt = load_source_gt(args.source_json)
        print(f"  {len(source_gt)} records indexed")

        print(f"Loading source LLM   : {args.source_llm}")
        source_llm = load_source_llm(args.source_llm)
        print(f"  {len(source_llm)} records indexed")

    # --- compute scope IDs ---
    print(f"\nScope: {args.scope}")
    scope_ids, scope_stats = get_scope_ids(
        target_records, args.scope, args.ids_file, args.phase4_per_frame
    )

    print(f"source_signal={args.source_signal}  "
          f"target_q={args.target_q}  "
          f"matching_mode={args.source_matching_mode}  "
          f"injection_format={args.injection_format}\n")

    # --- build ---
    result, stats, examples = build_records(
        target_records, source_gt, source_llm, scope_ids, args,
    )

    # --- print examples and stats ---
    print_examples(examples, args)
    print_stats(stats, args, scope_stats)

    if not args.dry_run:
        # --- validate ---
        try:
            validate_output(target_records, result, args.scope)
            print("\n  id_preservation_ok       : True")
        except AssertionError as e:
            print(f"\nVALIDATION FAILED: {e}", file=sys.stderr)
            sys.exit(1)

        # --- write ---
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(args.out, 'w') as f:
            json.dump(result, f)

    print_next_steps(args.out, args.dry_run)


if __name__ == '__main__':
    main()
