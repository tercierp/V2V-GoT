"""
test_mirror.py
==============

Verify that mirror_trajectory produces geometrically correct results:
  1. Mirror is offset perpendicular to motion direction
  2. Mirror is on the LEFT for side='left' (oncoming lane in right-hand traffic)
  3. Mirror is REVERSED in time order (represents oncoming vehicle)
  4. Mirror handles curves smoothly (no spikes)
"""

import math
import matplotlib.pyplot as plt

from trajectory_roads import mirror_trajectory


def test_straight_eastbound_mirrors_to_north():
    """A trajectory moving east (+x) should mirror to the NORTH (+y)
    when side='left' — that's the oncoming lane in right-hand traffic."""
    pts = [(t, float(t * 5), 0.0) for t in range(10)]  # east, y=0
    mirrored = mirror_trajectory(pts, lane_offset_m=3.5, side="left")

    # All mirrored y-coordinates should be approximately +3.5
    ys = [m[2] for m in mirrored]
    assert all(abs(y - 3.5) < 0.01 for y in ys), f"Expected y≈3.5, got {ys}"

    # Mirrored x-coordinates should match input x-coordinates (just reversed order)
    expected_xs = list(reversed([p[1] for p in pts]))
    actual_xs = [m[1] for m in mirrored]
    assert all(abs(a - e) < 0.01 for a, e in zip(actual_xs, expected_xs)), \
        f"Expected xs={expected_xs}, got {actual_xs}"

    # Time order should be reversed
    timestamps = [m[0] for m in mirrored]
    assert timestamps == list(reversed(range(10))), f"got {timestamps}"
    print("  ✓ Straight eastbound: mirror is to the north, time-reversed")


def test_straight_eastbound_right_side():
    """side='right' should put the mirror to the SOUTH."""
    pts = [(t, float(t * 5), 0.0) for t in range(10)]
    mirrored = mirror_trajectory(pts, lane_offset_m=3.5, side="right")
    ys = [m[2] for m in mirrored]
    assert all(abs(y + 3.5) < 0.01 for y in ys), f"Expected y≈-3.5, got {ys}"
    print("  ✓ Straight eastbound, side='right': mirror is to the south")


def test_curved_trajectory():
    """Curved trajectory should still produce a smooth mirror.

    Motion is counterclockwise (east toward north). 'Left of motion' points
    INWARD to the origin in this case. So the mirror radius should be
    SMALLER than the original by lane_offset_m.
    """
    # Quarter-circle from (10, 0) to (0, 10), radius 10, going CCW
    pts = [(t, 10 * math.cos(math.radians(t * 9)),
                10 * math.sin(math.radians(t * 9)))
           for t in range(11)]
    mirrored = mirror_trajectory(pts, lane_offset_m=2.0, side="left")
    assert len(mirrored) > 5, f"Mirror has only {len(mirrored)} points"

    # Mirror radius should be ~8m (10 - 2) since left-of-CCW-motion is inward
    for orig, mirr in zip(reversed(pts), mirrored):
        r_orig = math.sqrt(orig[1] ** 2 + orig[2] ** 2)
        r_mirr = math.sqrt(mirr[1] ** 2 + mirr[2] ** 2)
        # left of CCW motion = INWARD, so mirror is smaller radius
        assert abs(r_mirr - (r_orig - 2.0)) < 0.5, \
            f"At ts {orig[0]}: orig r={r_orig:.2f}, mirror r={r_mirr:.2f} " \
            f"(expected ~{r_orig - 2:.2f})"
    print("  ✓ Curve: mirror radius is reduced by lane_offset_m on concave side")


def test_sharp_turn_rejected():
    """A trajectory with a sharp 90° turn should produce an empty mirror."""
    # Hard right-angle turn at point 5
    pts = []
    for i in range(5):
        pts.append((i, float(i * 5), 0.0))   # going east
    for j in range(1, 6):
        pts.append((4 + j, 20.0, float(j * 5)))  # then going north
    mirrored = mirror_trajectory(pts, lane_offset_m=3.5, side="left")
    assert mirrored == [], f"Sharp turn should produce empty mirror, got {len(mirrored)} points"
    print("  ✓ Sharp 90° turn: mirror is correctly empty")


def test_visual():
    """Generate a side-by-side plot for visual inspection."""
    # Synthetic CAV trajectory: drives north, then turns east
    pts = []
    for t in range(20):
        x = max(0.0, t * 5.0 - 50)  # stays at 0 for first 10 steps, then increases
        y = min(50.0, t * 5.0)      # ramps up to 50 then stays
        pts.append((t, x, y))

    # Filter degenerate constant-stretch points so the curve is proper
    pts = [pts[0]] + [p for p in pts[1:] if p[1] != pts[0][1] or p[2] != pts[0][2]]

    fig, ax = plt.subplots(figsize=(8, 8))

    real_xs = [p[1] for p in pts]
    real_ys = [p[2] for p in pts]
    ax.plot(real_xs, real_ys, "b-o", label="Real CAV trajectory", markersize=4)
    ax.annotate("", xy=(real_xs[-1], real_ys[-1]),
                xytext=(real_xs[-2], real_ys[-2]),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))

    for side, color in [("left", "orange"), ("right", "green")]:
        mirror = mirror_trajectory(pts, lane_offset_m=3.5, side=side)
        mxs = [m[1] for m in mirror]
        mys = [m[2] for m in mirror]
        ax.plot(mxs, mys, "--s", color=color, alpha=0.7,
                label=f"Mirror side='{side}'", markersize=4)
        if len(mxs) >= 2:
            ax.annotate("", xy=(mxs[-1], mys[-1]),
                        xytext=(mxs[-2], mys[-2]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2))

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_title("Mirror trajectory geometry test\n"
                 "Real CAV in blue. side='left' (orange) = oncoming in US.\n"
                 "Arrows show direction of travel — mirror should point opposite.")
    out_path = "_test_mirror.png"
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Visual saved to {out_path}")


if __name__ == "__main__":
    print("Testing mirror_trajectory geometry:")
    test_straight_eastbound_mirrors_to_north()
    test_straight_eastbound_right_side()
    test_curved_trajectory()
    test_sharp_turn_rejected()
    test_visual()
    print("\nAll tests passed.")
