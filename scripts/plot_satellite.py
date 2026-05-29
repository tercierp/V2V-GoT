import os
import glob
import copy
import math
import yaml
import random
import numpy as np
import open3d as o3d
import folium
from folium.plugins import MousePosition, MeasureControl, MiniMap


# ============================================================
# USER SETTINGS
# ============================================================

DATA_ROOT = "V2V4REAL/V2V4REAL/Data"
SPLITS = ["train_01"]
FRAME_STEP = 30
FRAME_RANGE_MODE = "full"  # "full" or "range"
FRAME_START = 0
FRAME_END = 0
EGO_AGENT_ID = "0"

# Fallback GPS, used only if YAML gps is not valid lat/lon.
# Use the coordinates you already had.
FALLBACK_EGO_LAT = 39.976794
FALLBACK_EGO_LON = -83.019263

# Map/debug settings
OUTPUT_HTML = None
VOXEL_SIZE = 0.8              # increase to 1.0 / 2.0 if HTML is too heavy
MAX_POINTS_PER_AGENT = 6000   # keep HTML reasonable
Z_MIN = -3.0
Z_MAX = 3.0

# If map looks mirrored or rotated, debug these:
FLIP_X = False
FLIP_Y = False
SWAP_XY = False
MANUAL_HEADING_OFFSET_DEG = 0

# Auto-estimate heading from GPS track to avoid manual offsets.
AUTO_HEADING_FROM_GPS = True
GPS_HEADING_FRAMES = 20
GPS_HEADING_FORWARD_ONLY = True

# Rendering mode: C (pose-gps-fit)
GPS_FIT_FRAMES = 20

# Reduce console output for large runs if needed.
VERBOSE = True

# Prefer true_ego_pos when both are present (more consistent across agents in V2V4Real).
PREFER_TRUE_EGO_POS = True

# Higher zoom / overzoom.
# Satellite imagery may become blurry above native zoom, but Leaflet will let you zoom in.
MAP_ZOOM_START = 21
MAP_MAX_ZOOM = 24
MAP_MAX_NATIVE_ZOOM = 19


# ============================================================
# YAML / POSE UTILS
# ============================================================

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def angle_to_rad(a):
    """
    Accept angle in deg or rad.
    If it looks larger than pi-ish, assume degrees.
    """
    a = float(a)
    if abs(a) > 2 * math.pi:
        return math.radians(a)
    return a


def angle_to_deg(a):
    a = float(a)
    if abs(a) <= 2 * math.pi:
        return math.degrees(a)
    return a


def pose6_to_matrix(pose):
    """
    Converts OpenCOOD/V2V4Real-style pose:
        [x, y, z, roll, yaw, pitch]
    into a 4x4 transform.

    This is normally the pose of the LiDAR / ego in the world frame.
    """
    x, y, z, roll, yaw, pitch = pose

    roll = angle_to_rad(roll)
    yaw = angle_to_rad(yaw)
    pitch = angle_to_rad(pitch)

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


def yaw_from_matrix(T):
    """
    Extract yaw (heading) in degrees from a 4x4 transform.
    Assumes Z-up and yaw about Z axis.
    """
    R = np.asarray(T)[:3, :3]
    yaw = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(yaw)


def get_transform_from_yaml(data):
    """
    Prefer lidar_pose.
    Fallback to true_ego_pos.

    Handles either:
      - 4x4 matrix
      - 6D pose [x, y, z, roll, yaw, pitch]
    """
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


def get_yaw_from_pose_only(data):
    """
    Try to extract yaw/heading from pose only (no GPS fallback).
    Returns degrees.
    """
    if PREFER_TRUE_EGO_POS:
        key_order = ["true_ego_pos", "lidar_pose"]
    else:
        key_order = ["lidar_pose", "true_ego_pos"]

    for key in key_order:
        if key in data:
            pose = np.array(data[key])
            if pose.shape == (4, 4):
                return yaw_from_matrix(pose)
            if pose.shape == (6,):
                return angle_to_deg(pose[4])

    return 0.0


def get_yaw_with_source(data):
    """
    Returns yaw_deg, source_key.
    source_key is one of: true_ego_pos, lidar_pose, gps, fallback.
    """
    if PREFER_TRUE_EGO_POS:
        key_order = ["true_ego_pos", "lidar_pose"]
    else:
        key_order = ["lidar_pose", "true_ego_pos"]

    for key in key_order:
        if key in data:
            pose = np.array(data[key])
            if pose.shape == (4, 4):
                return yaw_from_matrix(pose), key
            if pose.shape == (6,):
                return angle_to_deg(pose[4]), key

    gps = data.get("gps", None)
    if gps is not None and len(gps) >= 4:
        return angle_to_deg(gps[3]), "gps"

    return 0.0, "fallback"


def get_ego_latlon(data):
    """
    If YAML gps looks like real lat/lon, use it.
    Otherwise use fallback.
    """
    gps = data.get("gps", None)

    if gps is not None and len(gps) >= 2:
        a = float(gps[0])
        b = float(gps[1])

        # Valid lat/lon range check
        if -90 <= a <= 90 and -180 <= b <= 180:
            return a, b, "yaml_gps"

    return FALLBACK_EGO_LAT, FALLBACK_EGO_LON, "fallback_manual"


def latlon_to_local(lat, lon, lat0, lon0):
    """
    Convert lat/lon to local ENU (east, north) in meters.
    """
    lat0_rad = math.radians(lat0)
    north = (lat - lat0) * (math.pi / 180.0) * EARTH_RADIUS_M
    east = (lon - lon0) * (math.pi / 180.0) * EARTH_RADIUS_M * math.cos(lat0_rad)
    return east, north


def enu_to_latlon(east, north, lat0, lon0):
    """
    Convert local ENU (east, north) in meters to lat/lon.
    """
    lat = lat0 + (north / EARTH_RADIUS_M) * (180.0 / math.pi)
    lon = lon0 + (east / (EARTH_RADIUS_M * math.cos(math.radians(lat0)))) * (180.0 / math.pi)
    return lat, lon


def bearing_deg(lat1, lon1, lat2, lon2):
    """
    Bearing from point 1 to point 2 in degrees.
    0 deg = North, 90 deg = East.
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)

    x = math.sin(dlon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)

    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def list_frame_ids(agent_dir):
    frame_ids = []
    for path in glob.glob(os.path.join(agent_dir, "*.yaml")):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.isdigit():
            frame_ids.append(int(name))
    return sorted(frame_ids)


def heading_from_gps_track(agent_dir, frame_id, num_frames, forward_only):
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

    gps_points = []
    for fid in window:
        yaml_path = os.path.join(agent_dir, f"{fid:06d}.yaml")
        if not os.path.exists(yaml_path):
            continue
        data = load_yaml(yaml_path)
        lat, lon, ok = get_yaml_latlon_if_available(data)
        if ok:
            gps_points.append((lat, lon))

    if len(gps_points) < 2:
        return None

    lat1, lon1 = gps_points[0]
    lat2, lon2 = gps_points[-1]
    return bearing_deg(lat1, lon1, lat2, lon2)


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
    return R, t, len(P)


# ============================================================
# MAP PROJECTION UTILS
# ============================================================

EARTH_RADIUS_M = 6378137.0


def ego_points_to_latlon(points_ego, ego_lat, ego_lon, heading_deg):
    """
    Project ego-frame points to GPS.

    Assumption:
      ego x = forward
      ego y = left
      ego z = up

    heading_deg:
      0 deg = facing North
      90 deg = facing East
    """

    pts = points_ego.copy()

    if SWAP_XY:
        pts[:, [0, 1]] = pts[:, [1, 0]]

    if FLIP_X:
        pts[:, 0] *= -1

    if FLIP_Y:
        pts[:, 1] *= -1

    x = pts[:, 0]
    y = pts[:, 1]

    heading = math.radians(heading_deg + MANUAL_HEADING_OFFSET_DEG)

    east = x * math.sin(heading) - y * math.cos(heading)
    north = x * math.cos(heading) + y * math.sin(heading)

    lat0_rad = math.radians(ego_lat)

    lat = ego_lat + (north / EARTH_RADIUS_M) * (180.0 / math.pi)
    lon = ego_lon + (east / (EARTH_RADIUS_M * math.cos(lat0_rad))) * (180.0 / math.pi)

    return lat, lon

def single_ego_point_to_latlon(point_ego, ego_lat, ego_lon, heading_deg):
    """
    Project one ego-frame point [x, y, z] to GPS lat/lon.
    """
    point_ego = np.asarray(point_ego).reshape(1, 3)
    lat, lon = ego_points_to_latlon(
        point_ego,
        ego_lat=ego_lat,
        ego_lon=ego_lon,
        heading_deg=heading_deg
    )
    return float(lat[0]), float(lon[0])


def get_yaml_latlon_if_available(data):
    """
    Try to get real lat/lon from YAML gps field.
    Returns:
        lat, lon, True
    or:
        None, None, False
    """
    gps = data.get("gps", None)

    if gps is None or len(gps) < 2:
        return None, None, False

    lat = float(gps[0])
    lon = float(gps[1])

    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon, True

    return None, None, False


def get_agent_distance_from_ego(T_ego_agent):
    """
    Agent origin expressed in ego frame.
    """
    agent_origin_ego = T_ego_agent[:3, 3]
    distance_xy = float(np.linalg.norm(agent_origin_ego[:2]))
    return agent_origin_ego, distance_xy

def heading_endpoint(ego_lat, ego_lon, heading_deg, length_m=35.0):
    heading = math.radians(heading_deg + MANUAL_HEADING_OFFSET_DEG)

    east = length_m * math.sin(heading)
    north = length_m * math.cos(heading)

    lat0_rad = math.radians(ego_lat)

    lat2 = ego_lat + (north / EARTH_RADIUS_M) * (180.0 / math.pi)
    lon2 = ego_lon + (east / (EARTH_RADIUS_M * math.cos(lat0_rad))) * (180.0 / math.pi)

    return lat2, lon2


# ============================================================
# PROCESS SPLITS / SCENARIOS
# ============================================================

MODE_LABEL = "C pose-gps-fit"

colors = [
    "#00FF00",  # green
    "#00BFFF",  # blue
    "#FF00FF",  # magenta
    "#FFFF00",  # yellow
    "#FFA500",  # orange
    "#00FFFF",  # cyan
    "#FFFFFF",  # white
]

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

        first_frame_id = frame_ids[0]
        ego_yaml_path = os.path.join(scenario_dir, EGO_AGENT_ID, f"{first_frame_id:06d}.yaml")
        if not os.path.exists(ego_yaml_path):
            print(f"Skipping scenario {scenario_name}: missing ego YAML {ego_yaml_path}")
            continue

        ego_data = load_yaml(ego_yaml_path)
        ego_lat, ego_lon, gps_source = get_ego_latlon(ego_data)

        if VERBOSE:
            print()
            print(f"=== SCENARIO {scenario_name} ({split_name}) ===")
            print(f"Ego lat/lon source: {gps_source}")
            print(f"Frame count: {len(frame_ids)} | step={FRAME_STEP}")

        # ============================================================
        # CREATE MAP WITH HIGH ZOOM
        # ============================================================

        m = folium.Map(
            location=[ego_lat, ego_lon],
            zoom_start=MAP_ZOOM_START,
            max_zoom=MAP_MAX_ZOOM,
            tiles=None,
            prefer_canvas=True,
            control_scale=True
        )

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Esri Satellite",
            overlay=False,
            control=True,
            max_zoom=MAP_MAX_ZOOM,
            max_native_zoom=MAP_MAX_NATIVE_ZOOM
        ).add_to(m)

        folium.TileLayer(
            tiles="OpenStreetMap",
            name="OpenStreetMap",
            overlay=False,
            control=True,
            max_zoom=MAP_MAX_ZOOM
        ).add_to(m)

        MousePosition(
            position="bottomright",
            separator=" | ",
            prefix="Lat/Lon:",
            num_digits=8
        ).add_to(m)

        MeasureControl(
            position="topleft",
            primary_length_unit="meters",
            primary_area_unit="sqmeters"
        ).add_to(m)

        MiniMap(toggle_display=True).add_to(m)

        # Debug circles around ego using first frame origin
        for r in [10, 25, 50, 100]:
            folium.Circle(
                location=[ego_lat, ego_lon],
                radius=r,
                color="white",
                fill=False,
                weight=1,
                opacity=0.6,
                tooltip=f"{r} m radius"
            ).add_to(m)

        agent_dirs = sorted([
            d for d in glob.glob(os.path.join(scenario_dir, "*"))
            if os.path.isdir(d)
        ])

        for frame_id in frame_ids:
            ego_yaml_path = os.path.join(scenario_dir, EGO_AGENT_ID, f"{frame_id:06d}.yaml")
            if not os.path.exists(ego_yaml_path):
                if VERBOSE:
                    print(f"Skipping frame {frame_id:06d}: missing ego YAML")
                continue

            ego_data = load_yaml(ego_yaml_path)
            ego_lat, ego_lon, gps_source = get_ego_latlon(ego_data)
            ego_heading_pose = get_yaw_from_pose_only(ego_data)
            ego_heading_deg = ego_heading_pose
            ego_heading_source = "pose"

            if AUTO_HEADING_FROM_GPS:
                ego_dir = os.path.join(scenario_dir, EGO_AGENT_ID)
                auto_heading = heading_from_gps_track(
                    ego_dir,
                    frame_id,
                    num_frames=GPS_HEADING_FRAMES,
                    forward_only=GPS_HEADING_FORWARD_ONLY
                )
                if auto_heading is not None:
                    ego_heading_deg = auto_heading
                    ego_heading_source = f"gps_track_{GPS_HEADING_FRAMES}"

            T_world_ego, ego_pose_key = get_transform_from_yaml(ego_data)
            T_ego_world = np.linalg.inv(T_world_ego)

            ego_dir = os.path.join(scenario_dir, EGO_AGENT_ID)
            fit_result = fit_world_to_gps(
                ego_dir,
                frame_id,
                num_frames=GPS_FIT_FRAMES,
                forward_only=GPS_HEADING_FORWARD_ONLY,
                lat0=ego_lat,
                lon0=ego_lon
            )

            if VERBOSE:
                print(f"--- Frame {frame_id:06d} ---")
                print(f"Ego heading/yaw: {ego_heading_deg:.3f} deg (source={ego_heading_source})")
                print(f"Ego transform source: {ego_pose_key}")
                if fit_result is not None:
                    _, _, fit_count = fit_result
                    print(f"Pose-to-GPS fit: ok (frames={fit_count})")
                else:
                    print("Pose-to-GPS fit: not available")

            show = frame_id == first_frame_id
            frame_group = folium.FeatureGroup(
                name=f"Frame {frame_id:06d} - {MODE_LABEL}",
                show=show
            )

            folium.Marker(
                [ego_lat, ego_lon],
                popup=f"Ego vehicle<br>agent={EGO_AGENT_ID}<br>heading={ego_heading_deg:.2f} deg",
                icon=folium.Icon(color="red", icon="car", prefix="fa")
            ).add_to(frame_group)

            heading_lat, heading_lon = heading_endpoint(ego_lat, ego_lon, ego_heading_deg)
            folium.PolyLine(
                locations=[[ego_lat, ego_lon], [heading_lat, heading_lon]],
                color="red",
                weight=5,
                tooltip="Ego heading"
            ).add_to(frame_group)

            for agent_idx, agent_dir in enumerate(agent_dirs):
                agent_id = os.path.basename(agent_dir)

                pcd_path = os.path.join(agent_dir, f"{frame_id:06d}.pcd")
                yaml_path = os.path.join(agent_dir, f"{frame_id:06d}.yaml")

                if not os.path.exists(pcd_path) or not os.path.exists(yaml_path):
                    if VERBOSE:
                        print(f"Skipping agent {agent_id}: missing PCD or YAML")
                    continue

                data = load_yaml(yaml_path)
                T_world_agent, pose_key = get_transform_from_yaml(data)
                agent_yaw_deg, agent_yaw_source = get_yaw_with_source(data)
                # Transform agent LiDAR cloud into ego LiDAR frame
                T_ego_agent = T_ego_world @ T_world_agent

                agent_origin_ego, agent_distance_xy = get_agent_distance_from_ego(T_ego_agent)

                agent_lat_from_tf, agent_lon_from_tf = single_ego_point_to_latlon(
                    agent_origin_ego,
                    ego_lat=ego_lat,
                    ego_lon=ego_lon,
                    heading_deg=ego_heading_deg
                )

                agent_gps_lat, agent_gps_lon, has_agent_gps = get_yaml_latlon_if_available(data)

                if VERBOSE:
                    print(f"Agent {agent_id}: yaw={agent_yaw_deg:.3f} ({agent_yaw_source})")

                if agent_id == EGO_AGENT_ID:
                    marker_color = "red"
                    marker_icon = "car"
                    marker_name = f"Ego Agent {agent_id}"
                else:
                    marker_color = "blue"
                    marker_icon = "car"
                    marker_name = f"CAV Agent {agent_id}"

                popup_html = f"""
                <b>{marker_name}</b><br>
                frame: {frame_id:06d}<br>
                pose source: {pose_key}<br>
                distance from ego: {agent_distance_xy:.2f} m<br>
                origin ego frame:<br>
                x={agent_origin_ego[0]:.2f} m<br>
                y={agent_origin_ego[1]:.2f} m<br>
                z={agent_origin_ego[2]:.2f} m<br>
                transform lat/lon:<br>
                {agent_lat_from_tf:.8f}, {agent_lon_from_tf:.8f}
                """

                if has_agent_gps:
                    popup_html += f"""
                    <br>YAML gps:<br>
                    {agent_gps_lat:.8f}, {agent_gps_lon:.8f}
                    """

                folium.Marker(
                    [agent_lat_from_tf, agent_lon_from_tf],
                    popup=popup_html,
                    tooltip=marker_name,
                    icon=folium.Icon(color=marker_color, icon=marker_icon, prefix="fa")
                ).add_to(frame_group)

                if agent_id != EGO_AGENT_ID:
                    folium.PolyLine(
                        locations=[
                            [ego_lat, ego_lon],
                            [agent_lat_from_tf, agent_lon_from_tf]
                        ],
                        color="blue",
                        weight=3,
                        opacity=0.8,
                        tooltip=f"Ego → Agent {agent_id}: {agent_distance_xy:.2f} m"
                    ).add_to(frame_group)

                if has_agent_gps and agent_id != EGO_AGENT_ID:
                    folium.CircleMarker(
                        location=[agent_gps_lat, agent_gps_lon],
                        radius=6,
                        color="yellow",
                        fill=True,
                        fill_color="yellow",
                        fill_opacity=0.9,
                        tooltip=f"Agent {agent_id} YAML GPS position"
                    ).add_to(frame_group)

                pcd = o3d.io.read_point_cloud(pcd_path)
                pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

                points = np.asarray(pcd.points)
                if len(points) == 0:
                    if VERBOSE:
                        print(f"Skipping agent {agent_id}: empty cloud")
                    continue

                z = points[:, 2]
                mask = (z >= Z_MIN) & (z <= Z_MAX)
                points = points[mask]

                if len(points) == 0:
                    if VERBOSE:
                        print(f"Skipping agent {agent_id}: empty after z-filter")
                    continue

                points_base = points
                if len(points_base) > MAX_POINTS_PER_AGENT:
                    idx = np.random.choice(len(points_base), MAX_POINTS_PER_AGENT, replace=False)
                    points_base = points_base[idx]

                points_base_h = np.hstack([points_base, np.ones((points_base.shape[0], 1))])

                if fit_result is None:
                    points_ego_h = (T_ego_agent @ points_base_h.T).T
                    points_proj = points_ego_h[:, :3]
                    lat, lon = ego_points_to_latlon(
                        points_proj,
                        ego_lat=ego_lat,
                        ego_lon=ego_lon,
                        heading_deg=ego_heading_pose
                    )
                else:
                    R_fit, t_fit, _ = fit_result
                    points_world_h = (T_world_agent @ points_base_h.T).T
                    points_world = points_world_h[:, :3]
                    world_xy = points_world[:, :2]
                    enu = (R_fit @ world_xy.T).T + t_fit
                    east = enu[:, 0]
                    north = enu[:, 1]
                    lat, lon = enu_to_latlon(east, north, ego_lat, ego_lon)

                color = "#FF0000" if agent_id == EGO_AGENT_ID else colors[agent_idx % len(colors)]
                for la, lo in zip(lat, lon):
                    folium.CircleMarker(
                        location=[float(la), float(lo)],
                        radius=1,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.65,
                        opacity=0.8,
                        weight=1
                    ).add_to(frame_group)

            frame_group.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        output_dir = os.path.join("maps", split_name)
        os.makedirs(output_dir, exist_ok=True)
        output_html = os.path.join(output_dir, f"fused_lidar_map_{scenario_name}.html")
        m.save(output_html)

        print(f"Saved map to: {output_html}")

print("Done.")