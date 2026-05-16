"""
osm_rasterizer.py
=================

Rasterize OSM Way geometry into a BEV (bird's-eye view) tensor aligned with
the PointPillars perception feature grid.

Output shape: (C, H, W) where:
  C = number of feature channels (road type, lane direction, distance to road, ...)
  H, W = grid cells (default 200x200, matching V2V4Real PointPillars BEV)

The BEV grid is centered on the ego CAV's local-frame position, with the
ego heading rotated to point along +x (so the model gets a body-frame view).

Design choices (V0 — keep simple):
  Channel 0: drivable mask (1 = on a road, 0 = off-road)
  Channel 1: road class (highway types, encoded ordinally / by importance)
  Channel 2: oneway direction along x in body frame (-1, 0, +1)
  Channel 3: signed distance to nearest road centerline (clipped to ±max_dist)

Future channels (V1):
  - Lane count
  - Road curvature
  - Distance to nearest intersection

Usage:
    >>> from osm_rasterizer import OSMRasterizer
    >>> r = OSMRasterizer(grid_size=200, resolution_m=0.5)
    >>> tensor = r.rasterize(ways, ego_lat, ego_lon, ego_heading_rad)
    >>> tensor.shape
    (4, 200, 200)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from coord_transform import LocalToWGS84
from osm_query import Way


# Importance ranking for road class channel. Higher = bigger road.
# Normalized to [0, 1] in the channel.
HIGHWAY_IMPORTANCE = {
    "motorway": 9, "motorway_link": 8,
    "trunk": 8, "trunk_link": 7,
    "primary": 7, "primary_link": 6,
    "secondary": 6, "secondary_link": 5,
    "tertiary": 5, "tertiary_link": 4,
    "unclassified": 3,
    "residential": 3,
    "living_street": 2,
    "service": 1,
}


@dataclass
class OSMRasterizer:
    """
    Rasterize OSM ways into a BEV tensor centered on the ego CAV.

    Args:
        grid_size: side length of the BEV grid in cells (default 200)
        resolution_m: meters per cell (default 0.5 → 100m x 100m FOV)
        max_distance_m: clip for the signed-distance channel
        road_width_m: half-width to inflate road centerlines into masks
        coord_transform: LocalToWGS84 instance with the dataset origin
    """

    grid_size: int = 200
    resolution_m: float = 0.5
    max_distance_m: float = 25.0
    road_width_m: float = 3.0   # Half-width of a typical lane
    coord_transform: LocalToWGS84 | None = None

    def __post_init__(self):
        if self.coord_transform is None:
            raise ValueError("OSMRasterizer requires a coord_transform (LocalToWGS84).")
        # Cell coordinates in body frame, broadcast for vectorized distance compute
        half = self.grid_size * self.resolution_m / 2
        cell_centers_x = np.linspace(
            -half + self.resolution_m / 2,
            half - self.resolution_m / 2,
            self.grid_size,
        )
        cell_centers_y = np.linspace(
            -half + self.resolution_m / 2,
            half - self.resolution_m / 2,
            self.grid_size,
        )
        # Body frame: x forward, y left
        self._grid_x, self._grid_y = np.meshgrid(cell_centers_x, cell_centers_y, indexing="xy")
        # Shape (H, W) for both

    @property
    def num_channels(self) -> int:
        return 4

    @property
    def fov_m(self) -> float:
        return self.grid_size * self.resolution_m

    # ------------------------------------------------------------------
    # Main rasterize call
    # ------------------------------------------------------------------
    def rasterize(
        self,
        ways: list[Way],
        ego_x_local: float,
        ego_y_local: float,
        ego_heading_rad: float,
        nodes_in_local_frame: bool = False,
    ) -> np.ndarray:
        """
        Args:
            ways: OSM ways from OSMQuery.fetch_roads(), OR trajectory-derived
                ways from TrajectoryRoadProvider.get_ways() (set
                nodes_in_local_frame=True for those).
            ego_x_local, ego_y_local: ego pose in V2V4Real local meters
            ego_heading_rad: ego heading (yaw) in radians (CCW from local +x)
            nodes_in_local_frame: if True, treat way.nodes as (x, y) in
                local meters directly, skipping the WGS84 -> local conversion.
                Use this with TrajectoryRoadProvider.

        Returns:
            tensor of shape (C, H, W), float32, in [-1, 1] roughly
        """
        H = W = self.grid_size

        # Channel buffers
        drivable_mask = np.zeros((H, W), dtype=np.float32)
        road_class = np.zeros((H, W), dtype=np.float32)
        oneway_dir = np.zeros((H, W), dtype=np.float32)
        signed_dist = np.full((H, W), self.max_distance_m, dtype=np.float32)

        if not ways:
            # No road in this area — all zeros, signed_dist saturates
            return self._stack_channels(drivable_mask, road_class, oneway_dir, signed_dist)

        cos_h = math.cos(-ego_heading_rad)  # rotate INTO body frame, so negate
        sin_h = math.sin(-ego_heading_rad)

        for way in ways:
            # 1. Convert way's nodes to ego-centric body frame
            local_nodes = []
            for node in way.nodes:
                if nodes_in_local_frame:
                    # Trajectory-derived ways: nodes are already (x, y) in local meters
                    lx, ly = node
                else:
                    # OSM ways: nodes are (lat, lon), convert to local
                    lat, lon = node
                    lx, ly = self.coord_transform.wgs84_to_local(lat, lon)
                # Translate so ego is at origin
                lx -= ego_x_local
                ly -= ego_y_local
                # Rotate so ego heading is +x
                bx = cos_h * lx - sin_h * ly
                by = sin_h * lx + cos_h * ly
                local_nodes.append((bx, by))

            # 2. For each segment in the way, compute distances and rasterize
            importance = HIGHWAY_IMPORTANCE.get(way.highway, 1) / 9.0

            for (x0, y0), (x1, y1) in zip(local_nodes[:-1], local_nodes[1:]):
                # Bounding-box culling: skip segments fully outside the grid
                fov = self.fov_m / 2 + self.road_width_m + self.max_distance_m
                if max(abs(x0), abs(x1)) > fov and max(abs(y0), abs(y1)) > fov:
                    continue

                # Per-segment direction in body frame:
                # +1 if segment goes mostly forward (+x), -1 if backward (-x)
                seg_dx = x1 - x0
                seg_dy = y1 - y0
                seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
                if way.oneway and seg_len > 1e-3:
                    direction = seg_dx / seg_len  # cos(angle with body +x), in [-1, 1]
                else:
                    direction = 0.0

                # Vectorized distance from each grid cell to this segment
                d = self._point_to_segment_distance(
                    self._grid_x, self._grid_y, x0, y0, x1, y1
                )

                # Update drivable mask (cells within road_width)
                seg_mask = d <= self.road_width_m
                drivable_mask[seg_mask] = 1.0
                # Higher-importance road overwrites lower
                road_class[seg_mask] = np.maximum(road_class[seg_mask], importance)
                if way.oneway:
                    # Where this segment writes its direction:
                    # use sign-dominant assignment so opposite-direction overlapping
                    # segments don't cancel each other to zero — instead the closer
                    # one wins (we already process closer cells with larger d-mask)
                    oneway_dir[seg_mask] = direction

                # Update signed distance (take min)
                np.minimum(signed_dist, d, out=signed_dist)

        # Clip signed_dist
        np.clip(signed_dist, 0, self.max_distance_m, out=signed_dist)
        # Normalize: 0 at road, 1 at max_distance
        signed_dist /= self.max_distance_m

        return self._stack_channels(drivable_mask, road_class, oneway_dir, signed_dist)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _point_to_segment_distance(
        px: np.ndarray, py: np.ndarray,
        x0: float, y0: float,
        x1: float, y1: float,
    ) -> np.ndarray:
        """Vectorized 2D point-to-segment distance."""
        dx = x1 - x0
        dy = y1 - y0
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-9:
            return np.hypot(px - x0, py - y0)
        # Project (px, py) - (x0, y0) onto the segment direction
        t = ((px - x0) * dx + (py - y0) * dy) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        # Closest point on segment
        cx = x0 + t * dx
        cy = y0 + t * dy
        return np.hypot(px - cx, py - cy)

    def _stack_channels(self, *channels: np.ndarray) -> np.ndarray:
        return np.stack(channels, axis=0).astype(np.float32)


def visualize(tensor: np.ndarray, save_path: str | None = None) -> None:
    """
    Quick matplotlib visualization of all channels for debugging.
    """
    import matplotlib.pyplot as plt
    C, H, W = tensor.shape
    fig, axes = plt.subplots(1, C, figsize=(4 * C, 4))
    if C == 1:
        axes = [axes]
    titles = ["drivable_mask", "road_class", "oneway_dir", "signed_dist (norm)"]
    for i, ax in enumerate(axes):
        im = ax.imshow(tensor[i], origin="lower", cmap="viridis")
        ax.set_title(titles[i] if i < len(titles) else f"ch{i}")
        ax.axhline(H // 2, color="r", lw=0.5)
        ax.axvline(W // 2, color="r", lw=0.5)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"Saved visualization to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    from coord_transform import V2V4REAL_PROVISIONAL_ORIGIN
    from osm_query import synthetic_grid_roads

    tx = V2V4REAL_PROVISIONAL_ORIGIN
    ways = synthetic_grid_roads(
        center_lat=tx.lat0, center_lon=tx.lon0, radius_m=300
    )
    print(f"Got {len(ways)} synthetic roads")

    rasterizer = OSMRasterizer(
        grid_size=200,
        resolution_m=0.5,
        coord_transform=tx,
    )

    # Rasterize at origin, heading along +x
    tensor = rasterizer.rasterize(
        ways=ways,
        ego_x_local=0.0,
        ego_y_local=0.0,
        ego_heading_rad=0.0,
    )
    print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"  drivable_mask: min={tensor[0].min()}, max={tensor[0].max()}, "
          f"mean={tensor[0].mean():.3f}")
    print(f"  road_class:    min={tensor[1].min()}, max={tensor[1].max()}")
    print(f"  oneway_dir:    unique={np.unique(tensor[2])}")
    print(f"  signed_dist:   min={tensor[3].min():.3f}, max={tensor[3].max():.3f}")

    # Save a debug visualization
    visualize(tensor, save_path="/home/claude/osm/_debug_synthetic.png")
