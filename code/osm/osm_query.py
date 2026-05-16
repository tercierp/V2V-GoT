"""
osm_query.py
============

Query OpenStreetMap for road geometry around a given GPS point, with disk
caching so we don't hammer the public Overpass API during training.

We fetch:
  - Road centerlines (highway=*) — primary, secondary, tertiary, residential, etc.
  - Lane direction tags (oneway, lanes) for planning context
  - Driveable surfaces only (filter out footways, cycleways, paths)

Output is a list of `Way` objects with WGS84 polylines + tag metadata.
The rasterizer (osm_rasterizer.py) consumes these.

Usage:
    >>> from osm_query import OSMQuery
    >>> q = OSMQuery(cache_dir="/scratch/izar/$USER/osm_cache")
    >>> ways = q.fetch_roads(lat=39.9612, lon=-82.9988, radius_m=200)
    >>> print(f"Got {len(ways)} road segments")
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Lazy imports — these are not stdlib
# import requests  # imported inside method


# Drivable highway types (excludes footway/cycleway/path/steps)
DRIVABLE_HIGHWAYS = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified",
    "residential",
    "service",
    "living_street",
}


@dataclass
class Way:
    """A single road segment from OSM."""
    osm_id: int
    nodes: list[tuple[float, float]]  # list of (lat, lon)
    highway: str                       # 'primary', 'residential', etc.
    oneway: bool = False
    lanes: Optional[int] = None
    name: Optional[str] = None
    raw_tags: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.nodes)


class OSMQuery:
    """
    Cached Overpass API client.

    The cache key is a hash of (lat, lon, radius). Two queries that round to
    the same key return identical results, which is what we want for training
    (we want OSM features to be deterministic per scene).

    For V2V-GoT training we will issue ~thousands of queries (one per scene
    timestamp). Caching is essential — without it we'd get rate-limited within
    minutes and training would crawl.
    """

    DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"
    # Mirrors that sometimes work better:
    #   https://overpass.kumi.systems/api/interpreter
    #   https://maps.mail.ru/osm/tools/overpass/api/interpreter

    def __init__(
        self,
        cache_dir: str | Path,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_s: int = 60,
        rate_limit_s: float = 1.0,
        round_to: int = 4,  # decimals: 4 ≈ 11m grid; cached aggressively
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.rate_limit_s = rate_limit_s
        self.round_to = round_to
        self._last_request_time = 0.0

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------
    def _cache_key(self, lat: float, lon: float, radius_m: float) -> str:
        # Quantize so nearby queries hit the same cache entry
        lat_q = round(lat, self.round_to)
        lon_q = round(lon, self.round_to)
        rad_q = round(radius_m, 0)
        s = f"{lat_q}_{lon_q}_{rad_q}"
        h = hashlib.md5(s.encode()).hexdigest()[:12]
        return f"osm_{lat_q}_{lon_q}_r{int(rad_q)}_{h}.json"

    def _cache_path(self, lat: float, lon: float, radius_m: float) -> Path:
        return self.cache_dir / self._cache_key(lat, lon, radius_m)

    # ------------------------------------------------------------------
    # Overpass query construction
    # ------------------------------------------------------------------
    @staticmethod
    def _build_overpass_query(lat: float, lon: float, radius_m: float) -> str:
        """Overpass QL query for drivable roads in radius around (lat, lon)."""
        return (
            f"[out:json][timeout:60];\n"
            f"(\n"
            f'  way["highway"](around:{radius_m},{lat},{lon});\n'
            f");\n"
            f"out geom;\n"
        )

    # ------------------------------------------------------------------
    # Main fetch
    # ------------------------------------------------------------------
    def fetch_roads(
        self,
        lat: float,
        lon: float,
        radius_m: float = 200.0,
        use_cache: bool = True,
        drivable_only: bool = True,
    ) -> list[Way]:
        """
        Fetch road segments within `radius_m` of (lat, lon).

        Returns a list of Way objects with WGS84 coordinates.
        """
        cache_path = self._cache_path(lat, lon, radius_m)

        # Try cache
        if use_cache and cache_path.exists():
            with open(cache_path) as f:
                raw = json.load(f)
            return self._parse_overpass_response(raw, drivable_only=drivable_only)

        # Rate limit
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_s:
            time.sleep(self.rate_limit_s - elapsed)

        # Hit Overpass
        import requests  # local import keeps this module importable without network deps installed
        query = self._build_overpass_query(lat, lon, radius_m)
        resp = requests.post(
            self.endpoint,
            data={"data": query},
            timeout=self.timeout_s,
        )
        self._last_request_time = time.time()
        resp.raise_for_status()
        raw = resp.json()

        # Cache
        with open(cache_path, "w") as f:
            json.dump(raw, f)

        return self._parse_overpass_response(raw, drivable_only=drivable_only)

    @staticmethod
    def _parse_overpass_response(raw: dict, drivable_only: bool = True) -> list[Way]:
        ways = []
        for elem in raw.get("elements", []):
            if elem.get("type") != "way":
                continue
            tags = elem.get("tags", {})
            highway = tags.get("highway")
            if highway is None:
                continue
            if drivable_only and highway not in DRIVABLE_HIGHWAYS:
                continue
            geom = elem.get("geometry") or []
            if len(geom) < 2:
                continue
            nodes = [(g["lat"], g["lon"]) for g in geom]
            ways.append(Way(
                osm_id=elem["id"],
                nodes=nodes,
                highway=highway,
                oneway=(tags.get("oneway") == "yes"),
                lanes=int(tags["lanes"]) if "lanes" in tags and tags["lanes"].isdigit() else None,
                name=tags.get("name"),
                raw_tags=tags,
            ))
        return ways


def synthetic_grid_roads(center_lat: float, center_lon: float, radius_m: float = 200) -> list[Way]:
    """
    Fallback: generate a synthetic grid road network for testing the rasterizer
    without hitting the Overpass API. Useful when you have no internet or when
    debugging without a known origin.
    """
    from coord_transform import LocalToWGS84
    tx = LocalToWGS84(lat0=center_lat, lon0=center_lon)

    ways = []
    # 5 east-west roads (horizontal in local frame)
    for j, y in enumerate(range(-200, 201, 100)):
        nodes = []
        for x in range(-int(radius_m), int(radius_m) + 1, 20):
            lat, lon = tx.local_to_wgs84(x, y)
            nodes.append((lat, lon))
        ways.append(Way(
            osm_id=10_000 + j,
            nodes=nodes,
            highway="secondary" if j == 2 else "residential",
            oneway=False,
            lanes=2,
            name=f"E-W road {j}",
        ))
    # 5 north-south roads
    for i, x in enumerate(range(-200, 201, 100)):
        nodes = []
        for y in range(-int(radius_m), int(radius_m) + 1, 20):
            lat, lon = tx.local_to_wgs84(x, y)
            nodes.append((lat, lon))
        ways.append(Way(
            osm_id=20_000 + i,
            nodes=nodes,
            highway="secondary" if i == 2 else "residential",
            oneway=False,
            lanes=2,
            name=f"N-S road {i}",
        ))
    return ways


if __name__ == "__main__":
    # Test the synthetic mode (no internet needed)
    ways = synthetic_grid_roads(center_lat=39.9612, center_lon=-82.9988, radius_m=200)
    print(f"Synthetic grid: {len(ways)} ways")
    for w in ways[:3]:
        print(f"  {w.name}: {len(w.nodes)} nodes, highway={w.highway}")
    print()
    print("To test the real OSM API:")
    print("  pip install requests")
    print("  q = OSMQuery(cache_dir='./osm_cache')")
    print("  ways = q.fetch_roads(lat=39.9612, lon=-82.9988, radius_m=200)")
