"""
coord_transform.py
==================

Convert between V2V4Real's local metric coordinate frame and WGS84 (lat/lon)
for OpenStreetMap queries.

V2V4Real positions in YAML / merge.jsonl are in a **local map frame** built by
LiDAR-SLAM and anchored to GPS/IMU. The frame is in METERS but is NOT WGS84.
You need a single (lat0, lon0) origin to convert. Once you have it, every
position in every scene maps to lat/lon with a flat-earth approximation
(accurate to ~cm at city scale).

V2V4Real was collected on highways near Columbus, Ohio (we will need to
either ask the authors for the exact origin, or recover it empirically by
matching a distinctive road feature in the LiDAR map to OSM).

Usage:
    >>> from coord_transform import LocalToWGS84
    >>> # Provisional: coords near Columbus, OH (UPDATE WHEN WE GET THE REAL ORIGIN)
    >>> tx = LocalToWGS84(lat0=39.9612, lon0=-82.9988)
    >>> lat, lon = tx.local_to_wgs84(x=143.83, y=-388.89)
    >>> print(f"{lat:.6f}, {lon:.6f}")
"""

import math
from dataclasses import dataclass


EARTH_RADIUS_M = 6_378_137.0  # WGS84 equatorial radius


@dataclass
class LocalToWGS84:
    """
    Flat-earth approximation around the dataset origin.

    Accurate to <1 cm for distances under ~5 km from origin, which covers
    a typical V2V4Real scene comfortably.

    Args:
        lat0: WGS84 latitude of the local frame origin (0,0,0)
        lon0: WGS84 longitude of the local frame origin
        x_axis: 'east' or 'north' — which way the local x-axis points.
            V2V4Real likely uses east-north-up (ENU). Verify empirically.
    """

    lat0: float
    lon0: float
    x_axis: str = "east"

    def __post_init__(self):
        self._cos_lat0 = math.cos(math.radians(self.lat0))
        self._meters_per_deg_lat = math.pi * EARTH_RADIUS_M / 180.0
        self._meters_per_deg_lon = self._meters_per_deg_lat * self._cos_lat0

    def local_to_wgs84(self, x: float, y: float) -> tuple[float, float]:
        """Local meters → (lat, lon) in degrees."""
        if self.x_axis == "east":
            dlat = y / self._meters_per_deg_lat
            dlon = x / self._meters_per_deg_lon
        elif self.x_axis == "north":
            dlat = x / self._meters_per_deg_lat
            dlon = y / self._meters_per_deg_lon
        else:
            raise ValueError(f"x_axis must be 'east' or 'north', got {self.x_axis}")
        return self.lat0 + dlat, self.lon0 + dlon

    def wgs84_to_local(self, lat: float, lon: float) -> tuple[float, float]:
        """(lat, lon) in degrees → local meters."""
        dlat_m = (lat - self.lat0) * self._meters_per_deg_lat
        dlon_m = (lon - self.lon0) * self._meters_per_deg_lon
        if self.x_axis == "east":
            return dlon_m, dlat_m
        else:
            return dlat_m, dlon_m


def pose_translation(pose_4x4_flat: list[float]) -> tuple[float, float, float]:
    """
    Extract (x, y, z) translation from a flattened 4x4 pose matrix as stored
    in V2V-GoT merge.jsonl entries (cav_ego_lidar_pose, cav_1_lidar_pose).

    The matrix layout is row-major:
        [r00, r01, r02, tx,
         r10, r11, r12, ty,
         r20, r21, r22, tz,
         0,   0,   0,   1]

    So translation lives at indices [3, 7, 11].
    """
    if len(pose_4x4_flat) != 16:
        raise ValueError(f"Expected 16-element flattened pose, got {len(pose_4x4_flat)}")
    return pose_4x4_flat[3], pose_4x4_flat[7], pose_4x4_flat[11]


# ----------------------------------------------------------------------------
# Provisional origins — UPDATE THESE WHEN YOU CONFIRM WITH AUTHORS
# ----------------------------------------------------------------------------
# V2V4Real was collected in Columbus, OH area (per the V2V4Real CVPR'23 paper).
# These are placeholders to let development proceed; they will need calibration
# before producing meaningful OSM data.

V2V4REAL_PROVISIONAL_ORIGIN = LocalToWGS84(
    lat0=39.9612,   # Columbus, OH approximate
    lon0=-82.9988,
    x_axis="east",
)

# V2X-Real is at UCLA campus (per the V2X-Real ICRA'24 paper).
V2XREAL_UCLA_ORIGIN = LocalToWGS84(
    lat0=34.0689,   # UCLA campus
    lon0=-118.4452,
    x_axis="east",
)


if __name__ == "__main__":
    # Sanity-check using the cav_ego_lidar_pose from your nq1 result file:
    sample_pose = [
        -0.8603860139846802, -0.5096380114555359, 0.0020271099638193846, -747.4990234375,
        0.5096420049667358, -0.8603760004043579, 0.004268390126526356, 246.41400146484375,
        -0.0004312550008762628, 0.004705559927970171, 0.9999889731407166, 4.646309852600098,
        0.0, 0.0, 0.0, 1.0,
    ]
    x, y, z = pose_translation(sample_pose)
    print(f"Local pose translation: x={x:.2f} m, y={y:.2f} m, z={z:.2f} m")

    tx = V2V4REAL_PROVISIONAL_ORIGIN
    lat, lon = tx.local_to_wgs84(x, y)
    print(f"WGS84 (provisional origin): {lat:.6f}, {lon:.6f}")
    print(f"OSM URL: https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=18")
    print()
    print("NOTE: The lat/lon above is a placeholder — the V2V4Real coordinate")
    print("origin is unknown. Update lat0/lon0 once we get it from the authors")
    print("or via empirical landmark matching.")

    # Round-trip test
    lat_test, lon_test = 39.9620, -82.9970
    x_back, y_back = tx.wgs84_to_local(lat_test, lon_test)
    lat_back, lon_back = tx.local_to_wgs84(x_back, y_back)
    print(f"\nRound-trip error: dlat={abs(lat_test-lat_back):.2e}, dlon={abs(lon_test-lon_back):.2e}")
