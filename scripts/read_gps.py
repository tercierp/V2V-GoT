import yaml
import numpy as np # NumPy must be imported so PyYAML knows how to decode the objects

# 1. Define the path to the YAML file
yaml_path = "V2V4REAL/V2V4REAL/Data/train_01/testoutput_CAV_data_2022-03-15-09-54-40_0/0/000000.yaml"

# 2. Load the YAML file
print("Decoding binary NumPy data...")
with open(yaml_path, 'r') as f:
    # We MUST use UnsafeLoader because the file contains !!python/object tags
    data = yaml.load(f, Loader=yaml.UnsafeLoader)

# 3. Extract the variables
ego_speed = data.get('ego_speed')
gps_data = data.get('gps', [])

# 4. Print the clean results!
print(f"Ego Speed: {ego_speed} m/s")

# Usually, GPS arrays in these datasets are formatted as: [Latitude, Longitude, Altitude, Heading/Yaw]
# Or sometimes as local [X, Y, Z, Yaw] map coordinates.
if len(gps_data) >= 2:
    print("\n--- Extracted Coordinates ---")
    print(f"Value 1 (Lat/X):  {gps_data[0]:.6f}")
    print(f"Value 2 (Lon/Y):  {gps_data[1]:.6f}")
    
    if len(gps_data) >= 3:
         print(f"Value 3 (Alt/Z):  {gps_data[2]:.6f}")
else:
    print(f"Raw GPS array: {gps_data}")