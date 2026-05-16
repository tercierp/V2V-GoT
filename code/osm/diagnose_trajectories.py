"""
diagnose_trajectories.py
========================

Plot raw CAV trajectories per scenario in the local world frame to see
whether the two CAVs drove in opposite lanes (good for wrong-lane signal)
or in convoy (less useful).

Output: one PNG per scenario showing both polylines, color-coded by CAV.

Usage:
    python diagnose_trajectories.py \\
        --db /scratch/izar/tercier/v2v-got/trajectory_roads.json \\
        --output-dir /scratch/izar/tercier/v2v-got/_traj_diag
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.db) as f:
        db = json.load(f)

    print(f"Loaded {len(db)} scenarios from {args.db}")

    # Per-scenario plots
    for scenario_id, polylines in sorted(db.items(), key=lambda x: int(x[0])):
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

        all_x = []
        all_y = []
        for i, poly in enumerate(polylines):
            xs = [p[1] for p in poly["points"]]
            ys = [p[2] for p in poly["points"]]
            all_x.extend(xs)
            all_y.extend(ys)

            color = colors[i % len(colors)]
            ax.plot(xs, ys, "-", color=color, linewidth=2,
                    label=f"CAV {poly['cav_id']} ({poly['length_m']:.0f}m, "
                          f"{poly['n_points']} pts)")
            # Mark direction with arrows
            if len(xs) >= 4:
                quarters = [len(xs) // 4, len(xs) // 2, 3 * len(xs) // 4]
                for q in quarters:
                    if q + 1 < len(xs):
                        dx = xs[q + 1] - xs[q]
                        dy = ys[q + 1] - ys[q]
                        norm = math.sqrt(dx * dx + dy * dy)
                        if norm > 0:
                            ax.annotate("",
                                xy=(xs[q] + 5 * dx / norm, ys[q] + 5 * dy / norm),
                                xytext=(xs[q], ys[q]),
                                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                            )
            # Start marker
            ax.plot(xs[0], ys[0], "o", color=color, markersize=8,
                    markeredgecolor="black")

        # Compute "convoy or oncoming" diagnostic
        diag_msg = ""
        if len(polylines) == 2:
            # Average direction of each polyline
            dirs = []
            for poly in polylines:
                pts = poly["points"]
                dx = pts[-1][1] - pts[0][1]
                dy = pts[-1][2] - pts[0][2]
                norm = math.sqrt(dx * dx + dy * dy)
                if norm > 0:
                    dirs.append((dx / norm, dy / norm))
            if len(dirs) == 2:
                dot = dirs[0][0] * dirs[1][0] + dirs[0][1] * dirs[1][1]
                if dot > 0.7:
                    diag_msg = "CONVOY (same direction)"
                elif dot < -0.7:
                    diag_msg = "ONCOMING (opposite directions)"
                else:
                    diag_msg = "CROSSING / TURNING"

        ax.set_title(f"Scenario {scenario_id}: {len(polylines)} polylines"
                     + (f" — {diag_msg}" if diag_msg else ""))
        ax.set_xlabel("Local X (m)")
        ax.set_ylabel("Local Y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        out_path = out_dir / f"scenario_{scenario_id}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=80)
        plt.close(fig)
        print(f"  Wrote {out_path}  [{diag_msg or 'N/A'}]")


if __name__ == "__main__":
    main()
