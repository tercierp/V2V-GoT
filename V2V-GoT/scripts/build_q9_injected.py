#!/usr/bin/env python3
"""
Build a Q9 eval JSON file where each Q9 human prompt is augmented with the
neighbor's Q6 intent, ready for re-inference without retraining.

This script produces the input JSON for the B2/B3 experiment:
  B2  peer-symmetric + Q9 injected, no resolver  → conflict_B_before
  B3  peer-symmetric + Q9 injected, with resolver → conflict_B_after

The output JSON preserves every original Q9 record id so that
phase4_eval.py can still match nq9_gt records to the new nq9_llm outputs.

Injection formats
-----------------
compact (default):
    Parses Q6 LLM text into a structured signal and injects only the key
    fields.  Avoids contaminating the Q9 prompt with raw Q6 phrases such as
    "CAV_EGO is at ...", "Its planned future trajectory is ...", or
    "is a notable object", which in the raw experiment caused the model to
    imitate Q6 output style instead of answering in Q9 format.

raw_q6:
    Injects the raw Q6 LLM text verbatim (the original experiment).  Use this
    only to reproduce the failed baseline.

Usage (--matching_mode same_asker is the default; inspect the dry-run
examples first to confirm the correct mode for your Q6 prompt design):

    # 1. Dry run — inspect examples, write nothing
    python scripts/build_q9_injected.py \\
        --q9_json  <path>/nq9sm3w6dc.json \\
        --q6_json  <path>/nq6sm3w1dc.json \\
        --q6_llm   <path>/nq6sm3w1dc_merge.jsonl \\
        --out      outputs/q9_injected/nq9_injected_compact_same_asker.json \\
        --injection_format compact \\
        --dry_run

    # 2. Generate injected JSON (compact, recommended)
    python scripts/build_q9_injected.py \\
        --q9_json  <path>/nq9sm3w6dc.json \\
        --q6_json  <path>/nq6sm3w1dc.json \\
        --q6_llm   <path>/nq6sm3w1dc_merge.jsonl \\
        --out      outputs/q9_injected/nq9_injected_compact_same_asker.json \\
        --injection_format compact

    # 3. Reproduce original raw experiment
    python scripts/build_q9_injected.py \\
        --q9_json  <path>/nq9sm3w6dc.json \\
        --q6_json  <path>/nq6sm3w1dc.json \\
        --q6_llm   <path>/nq6sm3w1dc_merge.jsonl \\
        --out      outputs/q9_injected/nq9_injected_same_asker.json \\
        --injection_format raw_q6

    # 4. Run Q9 inference on the injected JSON (use your normal nq9 script
    #    with --question-file pointing to the output above).

    # 5. Evaluate with phase4_eval.py
    python scripts/phase4_eval.py \\
        --nq8_gt   <path>/nq8sm3w6dc.json \\
        --nq9_gt   <path>/nq9sm3w6dc.json \\
        --nq8_llm  <path>/nq8sm3w6dc_merge.jsonl \\
        --nq9_llm  outputs/q9_injected/nq9_injected_compact_same_asker_merge.jsonl \\
        --output_dir outputs/phase4_q9_injected
    # conflict_B_before → B2,  conflict_B_after → B3
"""

import argparse
import json
import os
import re
import sys

# ── Injection templates ───────────────────────────────────────────────────────

_INJECT_TEMPLATE = (
    "\n\nNeighbor intent information from Q6:\n"
    "{q6_text}\n"
    "Use this neighbor intent when predicting the future trajectory. "
    "Keep the original output format: The suggested future trajectory is [(x,z),...]. "
    "Avoid future trajectory conflicts when possible while preserving a reasonable plan."
)

_COMPACT_INJECT_TEMPLATE = (
    "\n\nPeer intent signal:\n"
    "neighbor_id={neighbor_id}; "
    "neighbor_position={neighbor_position}; "
    "neighbor_next_waypoint_in_ego_frame={neighbor_next_waypoint}; "
    "neighbor_motion={neighbor_motion}.\n"
    "All peer coordinates are expressed in the current Q9 coordinate frame.\n"
    "Use this peer intent only as context for collision avoidance. "
    "Predict only {self_id}'s own future trajectory. "
    "Do not describe the peer vehicle. "
    "Do not copy the peer waypoint. "
    "Answer with exactly one sentence starting with: The suggested future trajectory is"
)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_q6_gt(path):
    """Load Q6 GT JSON → {(global_timestamp_index, asker_cav_id): record}."""
    with open(path) as f:
        data = json.load(f)
    index = {}
    for r in data:
        key = (r['global_timestamp_index'], r['asker_cav_id'])
        index[key] = r
    return index


def load_q6_llm(path):
    """Load Q6 LLM merge.jsonl → {question_id: text_string}."""
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

# ── Helpers ───────────────────────────────────────────────────────────────────

_OPPOSITE = {'ego': '1', '1': 'ego'}


def _normalize(text, max_chars):
    text = ' '.join(text.split())
    return text[:max_chars].rstrip() if len(text) > max_chars else text


def _deep_copy(rec):
    return json.loads(json.dumps(rec))


def parse_q6_intent(q6_text):
    """
    Parse Q6-style natural language into compact intent fields.

    Expected Q6 examples:
    - "CAV_1 is at (-75.7,5.2) moving forward. Its planned future trajectory
      is [(-26.8,0.3)]. CAV_1 is a not notable object."
    - "CAV_EGO is at (0.0,0.0) moving forward. Its planned future trajectory
      is [(52.6,0.6)]. CAV_EGO is a notable object."

    Returns (intent_dict, missing_set) where intent_dict has keys:
        neighbor_id, neighbor_position, neighbor_next_waypoint, neighbor_motion
    and missing_set contains the names of fields that defaulted to "unknown".
    """
    _UNKNOWN = "unknown"
    fields = {
        'neighbor_id':           _UNKNOWN,
        'neighbor_position':     _UNKNOWN,
        'neighbor_next_waypoint': _UNKNOWN,
        'neighbor_motion':       _UNKNOWN,
    }
    try:
        text = ' '.join(q6_text.split())

        m = re.search(r'\b(CAV_(?:EGO|\d+))\b', text)
        if m:
            fields['neighbor_id'] = m.group(1)

        m = re.search(r'is at \(([^)]+)\)', text)
        if m:
            fields['neighbor_position'] = '(' + m.group(1).replace(' ', '') + ')'

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

# ── Core build logic ──────────────────────────────────────────────────────────

def build_injected(q9_records, q6_gt, q6_llm, matching_mode, max_chars,
                   injection_format='compact', allow_reinject=False,
                   allow_unknown_compact=False):
    """
    Iterate over q9_records; inject Q6 text where possible.
    Returns (result_records, stats_dict, examples_dict).
    Every input record appears exactly once in result_records.
    """
    total = len(q9_records)
    n_injected = 0
    n_missing_gt = 0
    n_missing_llm = 0
    n_invalid = 0
    already_injected = 0
    compact_parse_all_ok = 0
    compact_parse_partial = 0
    compact_parse_failed = 0

    result = []
    examples = {'ego': None, 'cav1': None, 'any': None}

    _REINJECT_MARKERS = ('Neighbor intent information from Q6:', 'Peer intent signal:')

    for raw in q9_records:
        rec = _deep_copy(raw)

        # --- validate required structure ---------------------------------
        try:
            ts  = rec['global_timestamp_index']
            cid = rec['asker_cav_id']
            _   = rec['conversations'][0]['value']
            assert isinstance(rec['conversations'][0]['value'], str)
            rec_id = rec['id']
        except (KeyError, IndexError, TypeError, AssertionError):
            n_invalid += 1
            rec['q9_neighbor_intent_injected']        = False
            rec['q9_neighbor_intent_source_q6_id']    = None
            rec['q9_neighbor_intent_matching_mode']   = matching_mode
            rec['q9_neighbor_intent_injection_format'] = injection_format
            rec['q9_neighbor_intent_preview']         = None
            result.append(rec)
            continue

        # --- duplicate-injection guard -----------------------------------
        current_prompt = rec['conversations'][0]['value']
        if any(marker in current_prompt for marker in _REINJECT_MARKERS):
            if not allow_reinject:
                already_injected += 1
                rec['q9_neighbor_intent_injected']        = False
                rec['q9_neighbor_intent_source_q6_id']    = None
                rec['q9_neighbor_intent_matching_mode']   = matching_mode
                rec['q9_neighbor_intent_injection_format'] = injection_format
                rec['q9_neighbor_intent_preview']         = None
                result.append(rec)
                continue

        # --- determine Q6 lookup key ------------------------------------
        if matching_mode == 'same_asker':
            q6_key = (ts, cid)
        else:  # other_asker
            other = _OPPOSITE.get(cid)
            if other is None:
                n_invalid += 1
                rec['q9_neighbor_intent_injected']        = False
                rec['q9_neighbor_intent_source_q6_id']    = None
                rec['q9_neighbor_intent_matching_mode']   = matching_mode
                rec['q9_neighbor_intent_injection_format'] = injection_format
                rec['q9_neighbor_intent_preview']         = None
                result.append(rec)
                continue
            q6_key = (ts, other)

        # --- look up Q6 GT record ----------------------------------------
        q6_rec = q6_gt.get(q6_key)
        if q6_rec is None:
            n_missing_gt += 1
            rec['q9_neighbor_intent_injected']        = False
            rec['q9_neighbor_intent_source_q6_id']    = None
            rec['q9_neighbor_intent_matching_mode']   = matching_mode
            rec['q9_neighbor_intent_injection_format'] = injection_format
            rec['q9_neighbor_intent_preview']         = None
            result.append(rec)
            continue

        q6_id = q6_rec['id']

        # --- look up Q6 LLM output ---------------------------------------
        q6_text_raw = q6_llm.get(q6_id)
        if q6_text_raw is None:
            n_missing_llm += 1
            rec['q9_neighbor_intent_injected']        = False
            rec['q9_neighbor_intent_source_q6_id']    = q6_id
            rec['q9_neighbor_intent_matching_mode']   = matching_mode
            rec['q9_neighbor_intent_injection_format'] = injection_format
            rec['q9_neighbor_intent_preview']         = None
            result.append(rec)
            continue

        # --- inject -------------------------------------------------------
        q6_text = _normalize(q6_text_raw, max_chars)

        if injection_format == 'raw_q6':
            injection = _INJECT_TEMPLATE.format(q6_text=q6_text)
            preview = q6_text[:160]

        else:  # compact
            intent, missing = parse_q6_intent(q6_text)
            if len(missing) == 4 and not allow_unknown_compact:
                compact_parse_failed += 1
                rec['q9_neighbor_intent_injected']        = False
                rec['q9_neighbor_intent_source_q6_id']    = q6_id
                rec['q9_neighbor_intent_matching_mode']   = matching_mode
                rec['q9_neighbor_intent_injection_format'] = injection_format
                rec['q9_neighbor_intent_preview']         = None
                result.append(rec)
                continue
            elif len(missing) == 0:
                compact_parse_all_ok += 1
            else:
                compact_parse_partial += 1

            self_id = 'CAV_EGO' if cid == 'ego' else 'CAV_1'
            injection = _COMPACT_INJECT_TEMPLATE.format(**intent, self_id=self_id)
            preview = (
                "neighbor_id={neighbor_id}; pos={neighbor_position}; "
                "next={neighbor_next_waypoint}; motion={neighbor_motion}"
            ).format(**intent)

        rec['conversations'][0]['value'] += injection

        rec['q9_neighbor_intent_injected']        = True
        rec['q9_neighbor_intent_source_q6_id']    = q6_id
        rec['q9_neighbor_intent_matching_mode']   = matching_mode
        rec['q9_neighbor_intent_injection_format'] = injection_format
        rec['q9_neighbor_intent_preview']         = preview
        n_injected += 1

        if cid == 'ego'  and examples['ego']  is None:
            examples['ego']  = (rec_id, q6_id, rec['conversations'][0]['value'])
        if cid == '1'    and examples['cav1'] is None:
            examples['cav1'] = (rec_id, q6_id, rec['conversations'][0]['value'])
        if examples['any'] is None:
            examples['any']  = (rec_id, q6_id, rec['conversations'][0]['value'])

        result.append(rec)

    stats = {
        'total':               total,
        'injected':            n_injected,
        'missing_gt':          n_missing_gt,
        'missing_llm':         n_missing_llm,
        'invalid':             n_invalid,
        'already_injected':    already_injected,
        'compact_parse_all_ok': compact_parse_all_ok,
        'compact_parse_partial': compact_parse_partial,
        'compact_parse_failed': compact_parse_failed,
    }
    return result, stats, examples

# ── Validation ────────────────────────────────────────────────────────────────

def validate(original, result):
    if len(result) != len(original):
        raise AssertionError(
            f"Record count changed: input={len(original)}, output={len(result)}"
        )
    for i, (orig, out) in enumerate(zip(original, result)):
        if orig['id'] != out['id']:
            raise AssertionError(
                f"Record {i}: id changed from {orig['id']!r} to {out['id']!r}"
            )
        if out.get('q9_neighbor_intent_injected'):
            val = out['conversations'][0]['value']
            if not isinstance(val, str):
                raise AssertionError(
                    f"Record id={out['id']}: conversations[0]['value'] is not a string"
                )
            if ('Neighbor intent information from Q6:' not in val
                    and 'Peer intent signal:' not in val):
                raise AssertionError(
                    f"Record id={out['id']}: injected=True but no injection marker found"
                )

# ── Pretty printing ───────────────────────────────────────────────────────────

def print_examples(examples, matching_mode, injection_format):
    labels = [
        ('ego',  'ego CAV   (asker_cav_id="ego")'),
        ('cav1', 'cav1 CAV  (asker_cav_id="1")'),
        ('any',  'first injected record (any)'),
    ]
    print("=" * 72)
    print(f"Examples  [matching_mode={matching_mode}  injection_format={injection_format}]")
    print("=" * 72)
    for key, label in labels:
        ex = examples[key]
        if ex is None:
            print(f"\n[{label}]  — no injected record found")
            continue
        rec_id, q6_id, full_prompt = ex
        print(f"\n[{label}]")
        print(f"  Q9 record id : {rec_id}")
        print(f"  Q6 source id : {q6_id}")
        tail = full_prompt[-500:]
        print(f"  Modified prompt (last 500 chars):\n"
              f"  ···{tail!r}")
    print("=" * 72)


def print_stats(stats, out_path, dry_run, injection_format):
    print()
    print("=== Statistics ===")
    print(f"  Total Q9 records              : {stats['total']}")
    print(f"  Injected                      : {stats['injected']}")
    print(f"  Missing Q6 GT match           : {stats['missing_gt']}")
    print(f"  Missing Q6 LLM output         : {stats['missing_llm']}")
    print(f"  Invalid structure             : {stats['invalid']}")
    print(f"  Already injected (skipped)    : {stats['already_injected']}")
    if injection_format == 'compact':
        print(f"  Compact parse — all fields ok : {stats['compact_parse_all_ok']}")
        print(f"  Compact parse — partial       : {stats['compact_parse_partial']}")
        print(f"  Compact parse — all unknown (skipped) : {stats['compact_parse_failed']}")
    if dry_run:
        print(f"\nDry run complete — output NOT written.")
    else:
        print(f"\nOutput written to: {out_path}")


def print_next_steps(out_path):
    print()
    print("=== Next steps ===")
    print("1. Run Q9 inference on the injected JSON:")
    print(f"   Use your nq9 inference script with:")
    print(f"     --question-file {out_path}")
    print()
    print("2. Run phase4 evaluation with the resulting injected merge.jsonl:")
    print("   python scripts/phase4_eval.py \\")
    print("     --nq8_gt  <original_nq8_gt.json> \\")
    print("     --nq9_gt  <original_nq9_gt.json> \\")
    print("     --nq8_llm <original_nq8_merge.jsonl> \\")
    print("     --nq9_llm <injected_nq9_merge.jsonl> \\")
    print("     --output_dir outputs/phase4_q9_injected")
    print("   # conflict_B_before → B2,  conflict_B_after → B3")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Augment a Q9 eval JSON with the neighbor Q6 LLM intent '
            'to enable pre-LLM intent injection experiments (B2/B3). '
            'Use --injection_format compact (default) to avoid Q6-style '
            'output contamination, or --injection_format raw_q6 to '
            'reproduce the original experiment.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--q9_json', required=True,
                        help='Path to original Q9 GT eval JSON (must not be modified)')
    parser.add_argument('--q6_json', required=True,
                        help='Path to Q6 GT eval JSON (for record-ID lookup)')
    parser.add_argument('--q6_llm',  required=True,
                        help='Path to Q6 LLM merge.jsonl (model outputs to inject)')
    parser.add_argument('--out',     required=True,
                        help='Output path for the augmented Q9 JSON')
    parser.add_argument('--matching_mode', default='same_asker',
                        choices=['same_asker', 'other_asker'],
                        help=(
                            'same_asker (default): inject Q6 from the SAME asker_cav_id '
                            '(useful when Q6 asks "what does my neighbor plan?"). '
                            'other_asker: inject Q6 from the OPPOSITE asker_cav_id '
                            '(useful when Q6 is a self-broadcast from the neighbor). '
                            'Run --dry_run first to inspect which mode is semantically correct.'
                        ))
    parser.add_argument('--injection_format', default='compact',
                        choices=['compact', 'raw_q6'],
                        help=(
                            'compact (default): parse Q6 output into a structured signal '
                            'to avoid Q6-style output contamination. '
                            'raw_q6: inject the raw Q6 LLM text verbatim '
                            '(reproduces the original failed experiment).'
                        ))
    parser.add_argument('--max_chars', type=int, default=600,
                        help='Max characters of Q6 text to inject (default: 600)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print examples and statistics; do not write output file')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite --out if it already exists')
    parser.add_argument('--allow_reinject', action='store_true',
                        help=(
                            'If set, allow injection even when the prompt already '
                            'contains an injection marker. Default: skip such records.'
                        ))
    parser.add_argument('--allow_unknown_compact', action='store_true',
                        help=(
                            'In compact mode, inject even when all parsed fields are '
                            'unknown. Default: skip such records for signal quality.'
                        ))
    args = parser.parse_args()

    # --- safety: refuse to overwrite without --force ----------------------
    if not args.dry_run and os.path.exists(args.out) and not args.force:
        print(f"ERROR: output file already exists: {args.out}", file=sys.stderr)
        print("       Pass --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    # --- load data --------------------------------------------------------
    print(f"Loading Q9 GT JSON  : {args.q9_json}")
    with open(args.q9_json) as f:
        q9_records = json.load(f)

    print(f"Loading Q6 GT JSON  : {args.q6_json}")
    q6_gt = load_q6_gt(args.q6_json)

    print(f"Loading Q6 LLM JSONL: {args.q6_llm}")
    q6_llm = load_q6_llm(args.q6_llm)

    print(f"matching_mode={args.matching_mode}  injection_format={args.injection_format}  max_chars={args.max_chars}")
    print()

    # --- build ------------------------------------------------------------
    result, stats, examples = build_injected(
        q9_records, q6_gt, q6_llm,
        matching_mode=args.matching_mode,
        max_chars=args.max_chars,
        injection_format=args.injection_format,
        allow_reinject=args.allow_reinject,
        allow_unknown_compact=args.allow_unknown_compact,
    )

    # --- print examples BEFORE writing (so dry_run is useful) -------------
    print_examples(examples, args.matching_mode, args.injection_format)
    print_stats(stats, args.out, args.dry_run, args.injection_format)

    if args.dry_run:
        return

    # --- validate ---------------------------------------------------------
    try:
        validate(q9_records, result)
    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    # --- write ------------------------------------------------------------
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'w') as f:
        json.dump(result, f)

    print(f"Wrote {len(result)} records → {args.out}")
    print_next_steps(args.out)


if __name__ == '__main__':
    main()
