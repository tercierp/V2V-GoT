import os
import glob
import math
import yaml
import urllib.request
import time
from io import BytesIO

import numpy as np
from PIL import Image


# ============================================================
# USER SETTINGS
# ============================================================

DATA_ROOT = "../../../data/V2V4REAL/V2V4REAL/Data"
SPLITS = ["train_07", "train_08"]
FRAME_STEP = 30
FRAME_RANGE_MODE = "full"  # "full" or "range"
FRAME_START = 0
FRAME_END = 0
EGO_AGENT_ID = "0"

OUTPUT_ROOT = "sat_probe"
IMAGE_SIZE = 256
ZOOMS = [21, 20, 19, 18]

# Pose-to-GPS fit settings (mode C from plot_satellite)
GPS_FIT_FRAMES = 20
GPS_HEADING_FORWARD_ONLY = True
PREFER_TRUE_EGO_POS = True
POSE_ANGLES_IN_DEGREES = True

# Tile server providers
TILE_PROVIDERS = [
    {
        "name": "esri",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    },
    {
        "name": "memomaps",
        "url": "https://tileserver.memomaps.de/tilegen/{z}/{x}/{y}.png"
    }
]

# Optional providers via env vars
MAPTILER_KEY = os.getenv("MAPTILER_KEY")
if MAPTILER_KEY:
    TILE_PROVIDERS.append({
        "name": "maptiler",
        "url": f"https://api.maptiler.com/maps/satellite/{{z}}/{{x}}/{{y}}.jpg?key={MAPTILER_KEY}"
    })

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
if MAPBOX_TOKEN:
    TILE_PROVIDERS.append({
        "name": "mapbox",
        "url": f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}"
    })

TILE_SIZE = 256
USE_DISK_TILE_CACHE = True
TILE_CACHE_DIR = "tile_cache"

FALLBACK_EGO_LAT = 39.976794
FALLBACK_EGO_LON = -83.019263

VERBOSE = True


# ============================================================
# YAML / POSE UTILS
# ============================================================

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def pose6_to_matrix(pose):
    if len(pose) != 6:
        raise ValueError("Expected 6D pose [x, y, z, roll, yaw, pitch].")

    x, y, z, roll, yaw, pitch = pose

    if POSE_ANGLES_IN_DEGREES:
        roll = math.radians(roll)
        yaw = math.radians(yaw)
        pitch = math.radians(pitch)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)]
    ])

    Ry = np.array([
        [ math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T


def get_transform_from_yaml(data):
    if PREFER_TRUE_EGO_POS:
        key_order = ["true_ego_pos", "lidar_pose"]
    else:
        key_order = ["lidar_pose", "true_ego_pos"]

    for key in key_order:
        if key not in data:
            continue

        pose = np.array(data[key])

        if pose.shape == (4, 4):
            return pose.astype(float), key

        if pose.shape == (6,):
            return pose6_to_matrix(pose), key

    raise ValueError("No usable lidar_pose or true_ego_pos found in YAML.")


def get_yaml_latlon_if_available(data):
    gps = data.get("gps", None)

    if gps is None or len(gps) < 2:
        return None, None, False

    lat = float(gps[0])
    lon = float(gps[1])

    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon, True

    return None, None, False


def get_ego_latlon(data):
    lat, lon, ok = get_yaml_latlon_if_available(data)
    if ok:
        return lat, lon
    return FALLBACK_EGO_LAT, FALLBACK_EGO_LON


# ============================================================
# FIT UTILS
# ============================================================

def list_frame_ids(agent_dir):
    frame_ids = []
    for path in glob.glob(os.path.join(agent_dir, "*.yaml")):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.isdigit():
            frame_ids.append(int(name))
    return sorted(frame_ids)


def get_scenario_dirs(split_name):
    split_dir = os.path.join(DATA_ROOT, split_name)
    return sorted([
        d for d in glob.glob(os.path.join(split_dir, "*"))
        if os.path.isdir(d)
    ])


def get_frame_ids_for_scenario(scenario_dir):
    ego_dir = os.path.join(scenario_dir, EGO_AGENT_ID)
    return list_frame_ids(ego_dir)


def select_frame_ids(frame_ids):
    if not frame_ids:
        return []

    selected = frame_ids
    if FRAME_RANGE_MODE == "range":
        end = FRAME_END if FRAME_END > 0 else frame_ids[-1]
        selected = [fid for fid in frame_ids if FRAME_START <= fid <= end]

    if FRAME_STEP > 1:
        selected = selected[::FRAME_STEP]

    return selected


def latlon_to_local(lat, lon, lat0, lon0):
    lat0_rad = math.radians(lat0)
    north = (lat - lat0) * (math.pi / 180.0) * 6378137.0
    east = (lon - lon0) * (math.pi / 180.0) * 6378137.0 * math.cos(lat0_rad)
    return east, north


def enu_to_latlon(east, north, lat0, lon0):
    lat = lat0 + (north / 6378137.0) * (180.0 / math.pi)
    lon = lon0 + (east / (6378137.0 * math.cos(math.radians(lat0)))) * (180.0 / math.pi)
    return lat, lon


def fit_world_to_gps(agent_dir, frame_id, num_frames, forward_only, lat0, lon0):
    frame_ids = list_frame_ids(agent_dir)
    if not frame_ids:
        return None

    try:
        start_idx = frame_ids.index(int(frame_id))
    except ValueError:
        return None

    if forward_only:
        window = frame_ids[start_idx:start_idx + num_frames]
    else:
        half = max(1, num_frames // 2)
        window = frame_ids[max(0, start_idx - half):start_idx + half + 1]

    world_pts = []
    enu_pts = []
    for fid in window:
        yaml_path = os.path.join(agent_dir, f"{fid:06d}.yaml")
        if not os.path.exists(yaml_path):
            continue
        data = load_yaml(yaml_path)
        lat, lon, ok = get_yaml_latlon_if_available(data)
        if not ok:
            continue

        T_world, _ = get_transform_from_yaml(data)
        world_xy = T_world[:2, 3]
        east, north = latlon_to_local(lat, lon, lat0, lon0)

        world_pts.append(world_xy)
        enu_pts.append([east, north])

    if len(world_pts) < 2:
        return None

    P = np.asarray(world_pts)
    Q = np.asarray(enu_pts)

    P_center = P.mean(axis=0)
    Q_center = Q.mean(axis=0)
    H = (P - P_center).T @ (Q - Q_center)

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t = Q_center - R @ P_center
    return R, t


# ============================================================
# TILE UTILS
# ============================================================

def latlon_to_pixel(lat, lon, zoom):
    sin_lat = math.sin(math.radians(lat))
    sin_lat = min(max(sin_lat, -0.9999), 0.9999)

    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n * TILE_SIZE
    return x, y


def is_no_data_tile(image):
    arr = np.asarray(image)
    if arr.size == 0:
        return True

    mean = float(arr.mean())
    std = float(arr.std())

    # Heuristic for "Map data not yet available" placeholder tiles.
    return mean > 220 and std < 18


def fetch_tile(provider_url, z, x, y):
    key = (provider_url, z, x, y)
    if key in _tile_cache:
        return _tile_cache[key]

    if USE_DISK_TILE_CACHE:
        cache_path = os.path.join(TILE_CACHE_DIR, provider_url.replace("/", "_"), str(z), str(x), f"{y}.png")
        if os.path.exists(cache_path):
            img = Image.open(cache_path).convert("RGB")
            _tile_cache[key] = img
            return img

    url = provider_url.format(z=z, x=x, y=y)
    last_err = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            img = Image.open(BytesIO(data)).convert("RGB")
            if USE_DISK_TILE_CACHE:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                img.save(cache_path)
            _tile_cache[key] = img
            return img
        except Exception as exc:
            last_err = exc
            if attempt == 0:
                time.sleep(1)

    if VERBOSE:
        print(f"Tile fetch failed: {url} | {last_err}")
    img = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
    _tile_cache[key] = img
    return img


def fetch_satellite_image(provider_url, lat, lon, zoom, size):
    center_x, center_y = latlon_to_pixel(lat, lon, zoom)

    left = center_x - size / 2.0
    top = center_y - size / 2.0
    right = left + size
    bottom = top + size

    tile_x0 = int(math.floor(left / TILE_SIZE))
    tile_y0 = int(math.floor(top / TILE_SIZE))
    tile_x1 = int(math.floor((right - 1) / TILE_SIZE))
    tile_y1 = int(math.floor((bottom - 1) / TILE_SIZE))

    tiles_wide = tile_x1 - tile_x0 + 1
    tiles_high = tile_y1 - tile_y0 + 1

    stitched = Image.new("RGB", (tiles_wide * TILE_SIZE, tiles_high * TILE_SIZE))

    max_tile = 2 ** zoom
    for ty in range(tile_y0, tile_y1 + 1):
        if ty < 0 or ty >= max_tile:
            continue
        for tx in range(tile_x0, tile_x1 + 1):
            tx_wrap = tx % max_tile
            tile = fetch_tile(provider_url, zoom, tx_wrap, ty)
            ox = (tx - tile_x0) * TILE_SIZE
            oy = (ty - tile_y0) * TILE_SIZE
            stitched.paste(tile, (ox, oy))

    crop_left = int(round(left - tile_x0 * TILE_SIZE))
    crop_top = int(round(top - tile_y0 * TILE_SIZE))
    crop_right = crop_left + size
    crop_bottom = crop_top + size

    return stitched.crop((crop_left, crop_top, crop_right, crop_bottom))


_tile_cache = {}


# ============================================================
# MAIN
# ============================================================

for split_name in SPLITS:
    scenario_dirs = get_scenario_dirs(split_name)
    if not scenario_dirs:
        print(f"No scenarios found in split: {split_name}")
        continue

    for scenario_dir in scenario_dirs:
        scenario_name = os.path.basename(scenario_dir)
        frame_ids = select_frame_ids(get_frame_ids_for_scenario(scenario_dir))

        if not frame_ids:
            print(f"Skipping scenario {scenario_name}: no frames found")
            continue

        if VERBOSE:
            print()
            print(f"=== SCENARIO {scenario_name} ({split_name}) ===")
            print(f"Frame count: {len(frame_ids)} | step={FRAME_STEP}")

        agent_dirs = sorted([
            d for d in glob.glob(os.path.join(scenario_dir, "*"))
            if os.path.isdir(d)
        ])

        first_frame_id = frame_ids[0]
        scenario_fit_result = None
        ego_yaml_path = os.path.join(scenario_dir, EGO_AGENT_ID, f"{first_frame_id:06d}.yaml")
        if os.path.exists(ego_yaml_path):
            ego_data = load_yaml(ego_yaml_path)
            ego_lat, ego_lon = get_ego_latlon(ego_data)
            ego_dir = os.path.join(scenario_dir, EGO_AGENT_ID)
            fit = fit_world_to_gps(
                ego_dir,
                first_frame_id,
                num_frames=GPS_FIT_FRAMES,
                forward_only=GPS_HEADING_FORWARD_ONLY,
                lat0=ego_lat,
                lon0=ego_lon
            )
            if fit is not None:
                R_fit, t_fit = fit
                scenario_fit_result = (R_fit, t_fit, ego_lat, ego_lon)

        if scenario_fit_result is None and VERBOSE:
            print(f"Scenario {scenario_name}: fit unavailable, will retry per frame")

        for frame_id in frame_ids:
            ego_yaml_path = os.path.join(scenario_dir, EGO_AGENT_ID, f"{frame_id:06d}.yaml")
            if not os.path.exists(ego_yaml_path):
                if VERBOSE:
                    print(f"Skipping frame {frame_id:06d}: missing ego YAML")
                continue

            ego_data = load_yaml(ego_yaml_path)
            ego_lat, ego_lon = get_ego_latlon(ego_data)

            ego_dir = os.path.join(scenario_dir, EGO_AGENT_ID)
            fit_result = scenario_fit_result
            if fit_result is None:
                fit = fit_world_to_gps(
                    ego_dir,
                    frame_id,
                    num_frames=GPS_FIT_FRAMES,
                    forward_only=GPS_HEADING_FORWARD_ONLY,
                    lat0=ego_lat,
                    lon0=ego_lon
                )
                if fit is not None:
                    R_fit, t_fit = fit
                    fit_result = (R_fit, t_fit, ego_lat, ego_lon)

            if fit_result is None:
                if VERBOSE:
                    print(f"Frame {frame_id:06d}: fit unavailable, skipping")
                continue

            R_fit, t_fit, anchor_lat, anchor_lon = fit_result

            for agent_dir in agent_dirs:
                agent_id = os.path.basename(agent_dir)
                yaml_path = os.path.join(agent_dir, f"{frame_id:06d}.yaml")

                if not os.path.exists(yaml_path):
                    if VERBOSE:
                        print(f"Skipping agent {agent_id}: missing YAML")
                    continue

                data = load_yaml(yaml_path)
                T_world_agent, _ = get_transform_from_yaml(data)
                world_xy = T_world_agent[:2, 3]

                enu = (R_fit @ world_xy.reshape(2, 1)).reshape(2) + t_fit
                east, north = float(enu[0]), float(enu[1])
                agent_lat, agent_lon = enu_to_latlon(east, north, anchor_lat, anchor_lon)

                found_any = False
                for provider in TILE_PROVIDERS:
                    provider_name = provider["name"]
                    provider_url = provider["url"]

                    for zoom in ZOOMS:
                        image = fetch_satellite_image(provider_url, agent_lat, agent_lon, zoom, IMAGE_SIZE)
                        if is_no_data_tile(image):
                            if VERBOSE:
                                print(f"No data: {provider_name} z={zoom} agent={agent_id} frame={frame_id:06d}")
                            continue

                        out_dir = os.path.join(
                            OUTPUT_ROOT,
                            split_name,
                            scenario_name,
                            f"{frame_id:06d}",
                            f"{provider_name}_z{zoom}"
                        )
                        os.makedirs(out_dir, exist_ok=True)

                        out_path = os.path.join(out_dir, f"agent_{agent_id}.png")
                        image.save(out_path, format="PNG")

                        if VERBOSE:
                            print(f"Saved {out_path}")

                        found_any = True

                if not found_any and VERBOSE:
                    print(f"No imagery found for agent {agent_id} frame {frame_id:06d}")

print("Done.")