import open3d as o3d
import numpy as np
import yaml
import copy

# --- 1. Define Paths ---
pcd_path = "../../../data/V2V4REAL/V2V4REAL/Data/train_01/testoutput_CAV_data_2022-03-15-09-54-40_0/0/000000.pcd"
yaml_path = pcd_path.replace('.pcd', '.yaml')

# --- 2. Load YAML Data ---
with open(yaml_path, 'r') as f:
    yaml_data = yaml.load(f, Loader=yaml.UnsafeLoader)

# In V2V4Real, 'true_ego_pos' IS the pre-calculated 4x4 Transformation Matrix!
transform_matrix = np.array(yaml_data['true_ego_pos'])

# --- 3. Load and Transform the Point Cloud ---
print("Loading Local Point Cloud...")
pcd_local = o3d.io.read_point_cloud(pcd_path)

# Make a copy so we can visualize the difference
pcd_global = copy.deepcopy(pcd_local)

# Apply the mathematical transformation!
pcd_global.transform(transform_matrix)

# Paint them so we can tell them apart in the visualizer
pcd_local.paint_uniform_color([1, 0, 0])    # Red = Local (at origin 0,0)
pcd_global.paint_uniform_color([0, 1, 0])   # Green = Global (moved to real-world coordinates)

# --- 4. Save the result ---
output_filename = "fused_global_cloud.pcd"

# Combine both point clouds into one so you can see them together
combined_pcd = pcd_local + pcd_global

print(f"Saving combined point cloud to {output_filename}...")
o3d.io.write_point_cloud(output_filename, combined_pcd)
print("Done! Download this file to your laptop to view it.")