"""
test_pipeline_e2e.py
====================

End-to-end smoke test of the trajectory-derived road pipeline using a
synthetic merge.jsonl. Validates that:
  - DB builder correctly aggregates poses by (scenario, cav, timestamp)
  - Provider correctly serves Way objects with local-frame coords
  - Rasterizer correctly handles nodes_in_local_frame=True
  - Pre-computation writes one .npy per (scenario, timestamp, asker)
  - Round-trip from synthetic data back to BEV is sane
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from trajectory_roads import build_trajectory_road_db, TrajectoryRoadProvider
from trajectory_dataset import TrajectoryFeatureProvider
from osm_rasterizer import visualize


def make_pose_matrix(x: float, y: float, yaw_deg: float) -> list[float]:
    """Build a flattened 4x4 pose with the given translation and yaw."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return [
        c, -s, 0.0, x,
        s,  c, 0.0, y,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def synthesize_merge_jsonl(out_path: Path, num_scenarios: int = 2, num_timestamps: int = 30):
    """Generate a fake merge.jsonl with two CAVs driving down parallel lanes."""
    samples = []
    for scen in range(num_scenarios):
        # Scenario rotation so each scenario looks different
        scen_yaw = 30 * scen  # degrees

        for ts in range(num_timestamps):
            # CAV_EGO drives forward in lane 1: y=0, x increases
            # CAV_1 drives the opposite way in lane 2 (oncoming): y=3, x decreases
            ego_x_raw = ts * 5.0   # 5 m/timestep
            ego_y_raw = 0.0
            cav1_x_raw = 100.0 - ts * 5.0
            cav1_y_raw = 3.5

            # Apply scenario rotation so different scenarios occupy different
            # parts of the local frame (mimics V2V4Real's varied scenes)
            cy = math.cos(math.radians(scen_yaw))
            sy = math.sin(math.radians(scen_yaw))
            ego_x = cy * ego_x_raw - sy * ego_y_raw + 200 * scen
            ego_y = sy * ego_x_raw + cy * ego_y_raw - 100 * scen
            cav1_x = cy * cav1_x_raw - sy * cav1_y_raw + 200 * scen
            cav1_y = sy * cav1_x_raw + cy * cav1_y_raw - 100 * scen

            ego_pose = make_pose_matrix(ego_x, ego_y, scen_yaw)
            cav1_pose = make_pose_matrix(cav1_x, cav1_y, scen_yaw + 180)

            # Mimic V2V-GoT QA structure — two askers per timestamp
            for asker in ["ego", "1"]:
                samples.append({
                    "scenario_index": scen,
                    "global_timestamp_index": ts,
                    "asker_cav_id": asker,
                    "cav_ego_lidar_pose": ego_pose,
                    "cav_1_lidar_pose": cav1_pose,
                    "conversations": [
                        {"from": "human", "value": f"fake question {scen}/{ts}/{asker}"},
                        {"from": "gpt", "value": "fake answer"},
                    ],
                })

    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"[test] Wrote {len(samples)} synthetic samples to {out_path}")
    return samples


def main():
    workdir = Path(tempfile.mkdtemp(prefix="osm_e2e_"))
    print(f"Test working directory: {workdir}")

    # 1. Synthesize a fake merge.jsonl
    fake_merge = workdir / "merge.jsonl"
    samples = synthesize_merge_jsonl(fake_merge, num_scenarios=2, num_timestamps=30)

    # 2. Build the trajectory DB
    print("\n--- Step 2: Build trajectory DB ---")
    db_path = workdir / "trajectory_roads.json"
    db = build_trajectory_road_db(
        merge_jsonl_paths=[fake_merge],
        output_path=db_path,
    )
    assert "0" in db, "Scenario 0 missing"
    assert "1" in db, "Scenario 1 missing"
    assert len(db["0"]) == 2, f"Expected 2 polylines for scenario 0, got {len(db['0'])}"

    # 3. Load via the provider
    print("\n--- Step 3: TrajectoryRoadProvider ---")
    rp = TrajectoryRoadProvider(db_path)
    print("Stats:", json.dumps(rp.stats(), indent=2))
    ways = rp.get_ways(0)
    print(f"Scenario 0 has {len(ways)} ways")
    assert len(ways) == 2

    # 4. Run the full feature provider on a sample
    print("\n--- Step 4: TrajectoryFeatureProvider ---")
    fp = TrajectoryFeatureProvider(db_path=db_path)
    test_sample = samples[20]  # Mid-scenario, ego asker
    tensor = fp.get_features(test_sample)
    print(f"Tensor: shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"  drivable mask coverage: {tensor[0].mean():.2%}")
    print(f"  signed_dist: min={tensor[3].min():.3f}, max={tensor[3].max():.3f}")
    assert tensor.shape == (4, 200, 200)
    assert tensor[0].mean() > 0.01, "Drivable mask should be non-trivial"

    # 5. Visualize a few different samples to check body-frame rotation
    print("\n--- Step 5: Visualizing samples ---")
    for i, idx in enumerate([0, 30, 60, 90]):
        if idx >= len(samples):
            continue
        sample = samples[idx]
        tensor = fp.get_features(sample)
        out = workdir / f"e2e_sample_{i}_scen{sample['scenario_index']}_ts{sample['global_timestamp_index']}_{sample['asker_cav_id']}.png"
        visualize(tensor, save_path=str(out))

    # 6. Test pre-computation
    print("\n--- Step 6: Pre-compute all ---")
    feature_dir = workdir / "osm_features"
    fp2 = TrajectoryFeatureProvider(db_path=db_path)  # fresh, no in-mem cache
    fp2.precompute_all(
        merge_jsonl_paths=[fake_merge],
        output_dir=feature_dir,
    )

    n_files = len(list(feature_dir.glob("*.npy")))
    expected_unique = 2 * 30 * 2  # 2 scenarios * 30 timestamps * 2 askers
    print(f"Wrote {n_files} feature files (expected {expected_unique})")
    assert n_files == expected_unique, f"Expected {expected_unique} files, got {n_files}"

    # 7. Verify a saved tensor matches the live computation
    sample = samples[40]
    expected = fp.get_features(sample)
    saved_path = feature_dir / f"{sample['scenario_index']}_{sample['global_timestamp_index']}_{sample['asker_cav_id']}.npy"
    loaded = np.load(saved_path)
    assert np.allclose(expected, loaded), "Saved tensor differs from live!"
    print("Saved tensor matches live computation ✓")

    print(f"\nAll tests passed. Debug images saved to {workdir}/e2e_*.png")
    return workdir


if __name__ == "__main__":
    main()
