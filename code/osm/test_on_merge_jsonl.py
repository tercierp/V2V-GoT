"""
test_on_merge_jsonl.py
======================

Run the OSM pipeline against a real V2V-GoT merge.jsonl file (from your Izar
inference outputs) to verify everything plumbs together end-to-end.

Usage on Izar:
    cd /scratch/izar/$USER/v2v-got/V2V-GoT/code/osm
    python test_on_merge_jsonl.py \\
        --merge-jsonl /scratch/izar/$USER/v2v-got/V2V-GoT/LLaVA/playground/data/eval/v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2_4330_full_nq1sm3w0d/answers/val/llava-v1.5-7b/merge.jsonl \\
        --num-samples 5 \\
        --output-dir ./osm_debug

This will sample 5 entries, fetch synthetic OSM (no internet required), and
save a visualization for each so you can sanity-check the pipeline.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from coord_transform import V2V4REAL_PROVISIONAL_ORIGIN
from osm_dataset import OSMFeatureProvider
from osm_rasterizer import visualize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-jsonl", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--output-dir", type=Path, default=Path("./osm_debug"))
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/osm_cache"))
    ap.add_argument("--use-real-osm", action="store_true",
                    help="Hit the real Overpass API (default: synthetic)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load samples
    samples = []
    with open(args.merge_jsonl) as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {args.merge_jsonl}")

    # Set up provider
    provider = OSMFeatureProvider(
        coord_transform=V2V4REAL_PROVISIONAL_ORIGIN,
        cache_dir=args.cache_dir,
        synthetic_fallback=not args.use_real_osm,
    )

    # Process each sample
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i} ---")
        print(f"  scenario={sample.get('scenario_index')}  "
              f"timestamp={sample.get('global_timestamp_index')}  "
              f"asker={sample.get('asker_cav_id')}")
        print(f"  question: {sample['conversations'][0]['value'][:80]}...")

        tensor = provider.get_features(sample)
        print(f"  tensor: shape={tensor.shape}, drivable_coverage={tensor[0].mean():.2%}")

        out_path = args.output_dir / f"sample_{i}.png"
        visualize(tensor, save_path=str(out_path))

    print(f"\nDone. Debug images in {args.output_dir}")
    print(f"\nNext step: scp these PNGs to your laptop and inspect:")
    print(f"  scp -r tercier@izar.hpc.epfl.ch:{args.output_dir.absolute()} ~/Downloads/")


if __name__ == "__main__":
    main()
