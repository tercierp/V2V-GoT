#!/usr/bin/env python3
"""
Build a Q9 eval JSON whose speed/steering context sentence is replaced with the
Q8 LLM output (rather than the original Q8 GT answer).

This enables the staged pipeline:
  1. Run injected Q8 inference
  2. Merge subset Q8 outputs (merge_llm_outputs.py)
  3. THIS SCRIPT: rebuild Q9 prompts with hybrid Q8 context
  4. Run Q9 inference
  5. Evaluate with phase4_eval.py

The script looks for the Q8 GT answer sentence in each Q9 prompt:
    "The suggested speed setting is: X. The suggested steering setting is: Y."
and replaces it with the Q8 LLM output for the same (global_timestamp_index,
asker_cav_id). Records where this sentence is not found are preserved unchanged
and flagged with q9_from_q8_replaced=False.

Usage:
    # Dry run
    python scripts/build_q9_from_q8_context.py \\
        --q9_json  <path>/nq9sm3w6dc.json \\
        --q8_gt    <path>/nq8sm3w6dc.json \\
        --q8_llm   <hybrid_q8_merge.jsonl> \\
        --out      outputs/experiments/e1/nq9_from_hybrid_q8.json \\
        --dry_run

    # Full run
    python scripts/build_q9_from_q8_context.py \\
        --q9_json  <path>/nq9sm3w6dc.json \\
        --q8_gt    <path>/nq8sm3w6dc.json \\
        --q8_llm   <hybrid_q8_merge.jsonl> \\
        --out      outputs/experiments/e1/nq9_from_hybrid_q8.json \\
        --force
"""

import argparse
import csv
import io
import json
import os
import re
import sys

# ── Constants ─────────────────────────────────────────────────────────────────

Q8_CONTEXT_RE = re.compile(
    r'The suggested speed setting is:\s*\w[\w ]*\.\s*'
    r'The suggested steering setting is:\s*\w[\w ]*\.',
    re.IGNORECASE,
)

_OPPOSITE = {'ego': '1', '1': 'ego'}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _deep_copy(rec):
    return json.loads(json.dumps(rec))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_q8_gt_ids(path):
    """Load Q8 GT JSON → {(global_timestamp_index, asker_cav_id): q8_record_id}."""
    with open(path) as f:
        data = json.load(f)
    index = {}
    for r in data:
        key = (r['global_timestamp_index'], r['asker_cav_id'])
        index[key] = r['id']
    return index


def load_q8_llm(path):
    """Load Q8 LLM merge.jsonl → {question_id: text_string}."""
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


# ── Scope helpers ─────────────────────────────────────────────────────────────

def _load_ids_file(path):
    """Load record IDs from plain text or CSV file."""
    with open(path) as f:
        raw = f.read()
    lines = [l.strip() for l in raw.splitlines()
             if l.strip() and not l.strip().startswith('#')]
    if not lines:
        return set()
    if ',' in lines[0]:
        reader = csv.DictReader(io.StringIO(raw))
        ids = set()
        for row in reader:
            val = row.get('id') or row.get('question_id')
            if val and val.strip():
                ids.add(val.strip())
        return ids
    return {l for l in lines}


def get_scope_ids_c(target_records, scope, ids_file):
    """Return set of target record IDs for scope, or None for all."""
    if scope == 'all':
        return None
    if scope == 'ids':
        if not ids_file:
            print("ERROR: --ids_file is required for --scope ids", file=sys.stderr)
            sys.exit(1)
        return _load_ids_file(ids_file)
    print(f"ERROR: unsupported scope {scope!r}", file=sys.stderr)
    sys.exit(1)


# ── Q8 parsing and replacement ────────────────────────────────────────────────

def parse_q8_output(text):
    """Extract (speed_str, steering_str) from Q8 LLM output text.

    Returns None if parsing fails.
    Uses the same regex as phase4_eval.parse_nq8.
    """
    sp = re.search(r'speed setting is:\s*([a-z ]+?)(?:\.|$)', text, re.IGNORECASE)
    st = re.search(r'steering setting is:\s*([a-z ]+?)(?:\.|$)', text, re.IGNORECASE)
    if not sp or not st:
        return None
    speed_str = sp.group(1).strip().lower()
    steer_str = st.group(1).strip().lower()
    if not speed_str or not steer_str:
        return None
    return speed_str, steer_str


def replace_q8_context(prompt, new_speed, new_steering):
    """Find and replace the Q8 context sentence in a Q9 prompt.

    Returns (new_prompt, was_replaced).
    Uses re.sub with count=1 to replace only the first occurrence.
    """
    replacement = (
        f"The suggested speed setting is: {new_speed}. "
        f"The suggested steering setting is: {new_steering}."
    )
    new_prompt, n_subs = Q8_CONTEXT_RE.subn(replacement, prompt, count=1)
    return new_prompt, n_subs > 0


# ── Main build loop ───────────────────────────────────────────────────────────

def build_records(q9_records, q8_gt_ids, q8_llm, scope_ids, args):
    """Process Q9 records and replace Q8 context where possible.

    Returns (results, stats, examples).
    Every input record appears in results (records are never dropped).
    """
    stats = {
        'total': len(q9_records),
        'replaced': 0,
        'skipped_no_context': 0,
        'skipped_bad_q8_parse': 0,
        'skipped_scope': 0,
        'missing_q8_gt': 0,
        'missing_q8_llm': 0,
        'output_count': 0,
    }

    result = []
    examples = {'ego': None, 'cav1': None}

    for raw in q9_records:
        rec = _deep_copy(raw)

        try:
            ts = rec['global_timestamp_index']
            cid = rec['asker_cav_id']
            prompt = rec['conversations'][0]['value']
            assert isinstance(prompt, str)
            rec_id = rec['id']
        except (KeyError, IndexError, TypeError, AssertionError):
            rec['q9_from_q8_replaced'] = False
            rec['q9_from_q8_skipped_reason'] = 'invalid_record'
            result.append(rec)
            continue

        # --- scope filter (subset: exclude, but still add to result for scope=all) ---
        if scope_ids is not None and rec_id not in scope_ids:
            stats['skipped_scope'] += 1
            rec['q9_from_q8_replaced'] = False
            rec['q9_from_q8_skipped_reason'] = 'scope_excluded'
            result.append(rec)
            continue

        # --- look up Q8 GT record id ---
        q8_key = (ts, cid)
        q8_id = q8_gt_ids.get(q8_key)
        if q8_id is None:
            stats['missing_q8_gt'] += 1
            rec['q9_from_q8_replaced'] = False
            rec['q9_from_q8_skipped_reason'] = 'missing_q8_gt'
            result.append(rec)
            continue

        # --- look up Q8 LLM output ---
        q8_text = q8_llm.get(q8_id)
        if q8_text is None:
            stats['missing_q8_llm'] += 1
            rec['q9_from_q8_replaced'] = False
            rec['q9_from_q8_skipped_reason'] = 'missing_q8_llm'
            result.append(rec)
            continue

        # --- parse Q8 output ---
        parsed = parse_q8_output(q8_text)
        if parsed is None:
            stats['skipped_bad_q8_parse'] += 1
            rec['q9_from_q8_replaced'] = False
            rec['q9_from_q8_skipped_reason'] = 'bad_q8_parse'
            result.append(rec)
            continue

        new_speed, new_steering = parsed

        # --- replace context ---
        new_prompt, was_replaced = replace_q8_context(prompt, new_speed, new_steering)
        if not was_replaced:
            stats['skipped_no_context'] += 1
            rec['q9_from_q8_replaced'] = False
            rec['q9_from_q8_skipped_reason'] = 'no_q8_context_in_prompt'
            result.append(rec)
            continue

        rec['conversations'][0]['value'] = new_prompt
        rec['q9_from_q8_replaced'] = True
        rec['q9_from_q8_skipped_reason'] = None
        rec['q9_from_q8_source_q8_id'] = q8_id
        rec['q9_from_q8_new_speed'] = new_speed
        rec['q9_from_q8_new_steering'] = new_steering
        stats['replaced'] += 1

        result.append(rec)

        if cid == 'ego' and examples['ego'] is None:
            examples['ego'] = (rec_id, q8_id, new_prompt)
        if cid == '1' and examples['cav1'] is None:
            examples['cav1'] = (rec_id, q8_id, new_prompt)

    stats['output_count'] = len(result)
    return result, stats, examples


# ── Printing ──────────────────────────────────────────────────────────────────

def print_stats(stats, dry_run, out_path):
    print()
    print("=== Statistics ===")
    print(f"  total                : {stats['total']}")
    print(f"  replaced             : {stats['replaced']}")
    print(f"  skipped_no_context   : {stats['skipped_no_context']}")
    print(f"  skipped_bad_q8_parse : {stats['skipped_bad_q8_parse']}")
    print(f"  skipped_scope        : {stats['skipped_scope']}")
    print(f"  missing_q8_gt        : {stats['missing_q8_gt']}")
    print(f"  missing_q8_llm       : {stats['missing_q8_llm']}")
    print(f"  output_count         : {stats['output_count']}")
    if dry_run:
        print("\nDry run complete — output NOT written.")
    else:
        print(f"\nOutput written to: {out_path}")


def print_examples(examples):
    labels = [
        ('ego',  'ego CAV   (asker_cav_id="ego")'),
        ('cav1', 'cav1 CAV  (asker_cav_id="1")'),
    ]
    print()
    print("=" * 72)
    print("Examples [build_q9_from_q8_context]")
    print("=" * 72)
    for key, label in labels:
        ex = examples[key]
        if ex is None:
            print(f"\n[{label}]  — no replaced record found")
            continue
        rec_id, q8_id, new_prompt = ex
        print(f"\n[{label}]")
        print(f"  Q9 record id : {rec_id}")
        print(f"  Q8 source id : {q8_id}")
        tail = new_prompt[-600:]
        print(f"  Modified prompt (last 600 chars):\n  ···{tail!r}")
    print("=" * 72)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Build a Q9 eval JSON whose Q8 speed/steering context sentence is '
            'replaced with Q8 LLM output. Enables staged pipeline: '
            'injected Q8 inference → rebuild Q9 context → Q9 inference → evaluate.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--q9_json', required=True,
                        help='Original Q9 GT eval JSON')
    parser.add_argument('--q8_gt', required=True,
                        help='Q8 GT JSON (to map (ts, asker_cav_id) → q8_id)')
    parser.add_argument('--q8_llm', required=True,
                        help='Q8 LLM merge.jsonl (source of new Q8 context text)')
    parser.add_argument('--out', required=True,
                        help='Output path for the rebuilt Q9 JSON')
    parser.add_argument('--scope', choices=['all', 'ids'], default='all',
                        help='all: process all records; ids: only records in --ids_file')
    parser.add_argument('--ids_file',
                        help='File with one record id per line (for --scope ids)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite --out if it already exists')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print examples and stats; do not write output file')
    args = parser.parse_args()

    if not args.dry_run and os.path.exists(args.out) and not args.force:
        print(f"ERROR: output file already exists: {args.out}", file=sys.stderr)
        print("       Pass --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Q9 GT JSON  : {args.q9_json}")
    with open(args.q9_json) as f:
        q9_records = json.load(f)
    print(f"  {len(q9_records)} records")

    print(f"Loading Q8 GT JSON  : {args.q8_gt}")
    q8_gt_ids = load_q8_gt_ids(args.q8_gt)
    print(f"  {len(q8_gt_ids)} records indexed")

    print(f"Loading Q8 LLM JSONL: {args.q8_llm}")
    q8_llm = load_q8_llm(args.q8_llm)
    print(f"  {len(q8_llm)} records indexed")

    scope_ids = get_scope_ids_c(q9_records, args.scope, args.ids_file)

    result, stats, examples = build_records(q9_records, q8_gt_ids, q8_llm, scope_ids, args)

    print_examples(examples)
    print_stats(stats, args.dry_run, args.out)

    if args.dry_run:
        return

    # Validate output length for scope=all
    if args.scope == 'all' and len(result) != len(q9_records):
        print(f"VALIDATION FAILED: output count {len(result)} != input count {len(q9_records)}",
              file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'w') as f:
        json.dump(result, f)

    print(f"Wrote {len(result)} records → {args.out}")
    out_dir_abs = os.path.dirname(os.path.abspath(args.out))
    print()
    print("=== Next step: run Q9 inference ===")
    print("python -m llava.eval.model_vqa_loader \\")
    print(f"  --model-path <checkpoint> \\")
    print(f"  --question-file {args.out} \\")
    print(f"  --image-folder <image_dir> \\")
    print(f"  --answers-file {out_dir_abs}/merge.jsonl \\")
    print(f"  --temperature 0 --conv-mode vicuna_v1")


if __name__ == '__main__':
    main()
