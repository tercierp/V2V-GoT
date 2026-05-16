# OSM Integration for V2V-GoT

This module adds OpenStreetMap road context as an additional input modality to V2V-GoT, with the goal of improving Q8/Q9 (planning) performance.

## Why OSM

V2V-GoT predicts wrong-lane trajectories (Figure 31 of the paper). The model has no map context — it only sees LiDAR perception features. OSM provides road geometry and lane direction for free, which is exactly the missing signal.

## Module structure

```
code/osm/
├── coord_transform.py     Local frame ↔ WGS84 conversion (parameterised on origin)
├── osm_query.py           Overpass API client with on-disk caching + synthetic fallback
├── osm_rasterizer.py      OSM ways → BEV tensor (4 channels, 200x200, 0.5m/cell)
├── osm_dataset.py         Sample dict → OSM tensor (with rotation to body frame)
├── test_on_merge_jsonl.py Smoke test on real V2V-GoT outputs
└── README.md              this file
```

Each module is independently runnable as a script (`python <module>.py`).

## What's working today

- ✅ Coordinate conversion (local meters ↔ WGS84) parameterised on `(lat₀, lon₀)`
- ✅ Pose extraction from V2V-GoT's flattened 4×4 pose matrices
- ✅ Body-frame rotation (ego always faces +x in the BEV)
- ✅ 4-channel BEV: drivable mask, road class, oneway direction, signed distance
- ✅ On-disk + in-memory caching (deterministic per scenario+timestamp+CAV)
- ✅ Synthetic-grid fallback for offline development
- ✅ Overpass API client (untested without internet, but standard library code)

## What's not working yet

### 1. The coordinate origin is unknown
V2V4Real publishes positions in a custom local metric frame. We have placeholder coordinates near Columbus, OH, but the actual origin is not in the dataset. **This blocks real OSM queries.**

Two ways to resolve:
- **Ask the V2V4Real authors** for the WGS84 origin of the local frame. One email.
- **Recover empirically**: pick a distinctive intersection visible in the dataset's LiDAR map, find its (x, y) in the local frame, locate the same intersection on OSM, solve for `(lat₀, lon₀)` algebraically.

Until this is resolved, use `synthetic_fallback=True` to develop everything else.

### 2. Model integration not yet written
The next file to add is `model_integration.py` which:
- Wraps the existing V2V-GoT scene feature loader
- Concatenates the OSM tensor channel-wise with PointPillars features
- Updates `mm_scene_projector_input_size` from 3072 to `3072 + 4 * (something)`
- Patches the LLaVA dataset class to call `OSMFeatureProvider.get_features()`

### 3. Training script not patched
For 2× V100 (vs. paper's 8× A100):
- `--nproc_per_node=2` (not 8)
- `--per_device_train_batch_size 2 --gradient_accumulation_steps 8` (effective batch 32)
- `MODEL="osm_v2vgot_3ep"` (avoid overwriting pretrained checkpoint)
- `--num_train_epochs 3` (not 10 — fast iteration)

## Quick test on Izar

```bash
# After scp'ing the osm/ folder to /scratch/izar/$USER/v2v-got/V2V-GoT/code/osm
cd /scratch/izar/$USER/v2v-got/V2V-GoT/code/osm
python test_on_merge_jsonl.py \
    --merge-jsonl /scratch/izar/$USER/v2v-got/V2V-GoT/LLaVA/playground/data/eval/v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2_4330_full_nq1sm3w0d/answers/val/llava-v1.5-7b/merge.jsonl \
    --num-samples 5
```

This produces 5 PNGs of OSM BEV tensors. scp them to your laptop and verify they look like roads (not garbage). Without a real origin, the geometry will be a synthetic grid — that's expected for now.

## Roadmap

| Step | Status | Notes |
|------|--------|-------|
| Coord transform module | ✅ done | parametric on origin |
| Overpass client + cache | ✅ done | tested with synthetic data |
| BEV rasterizer | ✅ done | 4 channels, body-frame rotation |
| Dataset provider | ✅ done | per-sample caching |
| Smoke test on merge.jsonl | ✅ done | runs synthetically |
| Find V2V4Real origin | ⏳ blocked | email authors / empirical recovery |
| Wire into LLaVA dataset | ⏳ next | concat with scene features |
| Patch training script for 2x V100 | ⏳ next | reduce nproc, increase grad accum |
| First OSM training run | ⏳ next | 3 epochs, ~2 days on 2x V100 |
| Inference + eval on OSM model | ⏳ next | reuse existing inference slurm |
| Comparison table for report | ⏳ next | baseline vs OSM on Q8/Q9 |

## Pragmatic decision: synthetic OSM as a baseline

Even if we can't immediately resolve the V2V4Real origin, training with the **synthetic grid OSM** gives us a meaningful ablation. If synthetic-OSM training improves Q8/Q9 over baseline, that proves the model can use map-style context. If real OSM does even better, that's the final win.

This is actually a publishable framing: "we test whether map structure (synthetic) and real-world map alignment (OSM) each contribute to planning quality."

## Citations

When writing this up, the relevant references are:
- V2V4Real (CVPR'23, arXiv:2303.07601) — base dataset
- V2V-LLM (arXiv'24) — preceding model
- V2V-GoT (ICRA'26) — current model
- OpenStreetMap — © OpenStreetMap contributors, ODbL
- Overpass API — interfaces to OSM, MIT-licensed
