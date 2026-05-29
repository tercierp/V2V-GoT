#!/usr/bin/env python3
"""
Merge a subset (or full) injected LLM merge.jsonl (patch) into a full
original LLM merge.jsonl (base), preserving base order.

Use this after running inference on a subset of injected records to combine
the new answers back into the full output before phase4_eval.py.

Usage:
    python scripts/merge_llm_outputs.py \\
        --base_llm  <original_full_merge.jsonl> \\
        --patch_llm <injected_subset_merge.jsonl> \\
        --out       <output_merged.jsonl>

    # dry run — validate but do not write
    python scripts/merge_llm_outputs.py \\
        --base_llm  <original_full_merge.jsonl> \\
        --patch_llm <injected_subset_merge.jsonl> \\
        --out       <output_merged.jsonl> \\
        --dry_run
"""

import argparse
import json
import os
import sys


# ── Data loading ──────────────────────────────────────────────────────────────

def _get_qid(rec):
    qid = rec.get('question_id') or rec.get('id')
    if qid is None:
        raise ValueError(f"Record has neither 'question_id' nor 'id': {list(rec.keys())}")
    return qid


def load_jsonl_keyed(path, allow_duplicate_patch=False):
    """Load a merge.jsonl file.

    Returns (ordered_list, {qid: record}).
    Raises ValueError on duplicate qids unless allow_duplicate_patch is True
    (in which case the last-seen value wins and duplicates are tracked).
    """
    ordered = []
    keyed = {}
    duplicates = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: JSON parse error: {e}") from e
            qid = _get_qid(rec)
            if qid in keyed:
                if not allow_duplicate_patch:
                    raise ValueError(
                        f"{path}: duplicate question_id {qid!r} at line {lineno}"
                    )
                duplicates.append(qid)
            keyed[qid] = rec
            ordered.append(rec)
    return ordered, keyed, duplicates


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Merge a subset injected LLM merge.jsonl (patch) into a full '
            'original merge.jsonl (base), preserving base order. '
            'Use after inference on a subset of injected records.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--base_llm', required=True,
                        help='Full original merge.jsonl (e.g. 3446 lines)')
    parser.add_argument('--patch_llm', required=True,
                        help='Subset or full injected merge.jsonl to merge in')
    parser.add_argument('--out', required=True,
                        help='Output path for merged merge.jsonl')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite --out if it already exists')
    parser.add_argument('--dry_run', action='store_true',
                        help='Validate and print stats; do not write output')
    args = parser.parse_args()

    if not args.dry_run and os.path.exists(args.out) and not args.force:
        print(f"ERROR: output already exists: {args.out}", file=sys.stderr)
        print("       Pass --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading base  : {args.base_llm}")
    base_list, base_keyed, base_dups = load_jsonl_keyed(args.base_llm, allow_duplicate_patch=False)
    print(f"  {len(base_list)} records")

    print(f"Loading patch : {args.patch_llm}")
    _, patch_keyed, patch_dups = load_jsonl_keyed(args.patch_llm, allow_duplicate_patch=True)
    print(f"  {len(patch_keyed)} unique records")

    if patch_dups:
        print(f"  WARNING: {len(patch_dups)} duplicate question_ids in patch (last-seen used): "
              f"{patch_dups[:5]}{'...' if len(patch_dups) > 5 else ''}")

    # Merge: iterate base in order, replace records whose qid is in patch
    merged = []
    replaced = 0
    for rec in base_list:
        qid = _get_qid(rec)
        if qid in patch_keyed:
            merged.append(patch_keyed[qid])
            replaced += 1
        else:
            merged.append(rec)

    # Identify patch qids not found in base
    base_ids = {_get_qid(r) for r in base_list}
    missing_patch_ids = [qid for qid in patch_keyed if qid not in base_ids]

    if missing_patch_ids:
        print(f"  WARNING: {len(missing_patch_ids)} patch question_ids not found in base "
              f"(ignored): {missing_patch_ids[:5]}{'...' if len(missing_patch_ids) > 5 else ''}")

    # Validation
    out_ids = [_get_qid(r) for r in merged]
    base_ids_ordered = [_get_qid(r) for r in base_list]

    errors = []
    if len(merged) != len(base_list):
        errors.append(
            f"Output count {len(merged)} != base count {len(base_list)}"
        )
    if out_ids != base_ids_ordered:
        errors.append("Output ID order does not match base order")

    if errors:
        for e in errors:
            print(f"VALIDATION ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Stats
    print()
    print("=== Merge statistics ===")
    print(f"  base_count           : {len(base_list)}")
    print(f"  patch_count          : {len(patch_keyed)}")
    print(f"  replaced             : {replaced}")
    print(f"  output_count         : {len(merged)}")
    print(f"  duplicate_patch_ids  : {len(patch_dups)}")
    print(f"  missing_patch_ids    : {len(missing_patch_ids)}")
    print(f"  id_order_preserved   : True")

    if args.dry_run:
        print("\nDry run complete — output NOT written.")
        return

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'w') as f:
        for rec in merged:
            f.write(json.dumps(rec) + '\n')

    print(f"\nWrote {len(merged)} records → {args.out}")


if __name__ == '__main__':
    main()
