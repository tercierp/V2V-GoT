import json
import os
import glob

# --- 1. Define your exact paths ---
JSON_PATH = "data/V2V_GoT_JSONS/DMSTrack/V2V4Real/official_models/train_no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v2vllmq1.json"
RAW_DATA_ROOT = "../../../data/V2V4REAL/V2V4REAL/Data"

# Load the JSON
print("Loading JSON file...")
with open(JSON_PATH, 'r') as f:
    qa_data = json.load(f)

# --- 2. Map the Integer Indexes to String Folder Names ---
print("Scanning V2V4Real directories to build scenario mapping...")
all_scenario_folders = []

# Scan through train_01, test_01, etc., to find the actual scenario names
for split_dir in glob.glob(os.path.join(RAW_DATA_ROOT, "*_*")):
    if os.path.isdir(split_dir):
        for scenario_dir in os.listdir(split_dir):
            if os.path.isdir(os.path.join(split_dir, scenario_dir)):
                all_scenario_folders.append(scenario_dir)

# Datasets typically sort these alphabetically to assign the index 0, 1, 2, etc.
# We remove duplicates (set) and sort them alphabetically to recreate that index map!
scenario_mapping = sorted(list(set(all_scenario_folders)))

print(f"Found {len(scenario_mapping)} unique scenarios. Checking the first 5 QA pairs...\n")

# --- 3. Loop through the QA pairs ---
for item in qa_data[:5]: 
    
    scenario_idx = item.get("scenario_index")
    frame_idx = item.get("global_timestamp_index") # This acts as your frame number
    qa_id = item.get("id")
    
    # Safely get the folder name using the index
    try:
        scenario_name = scenario_mapping[scenario_idx]
    except IndexError:
        print(f"❌ Error: JSON asks for scenario_index {scenario_idx}, but we only found {len(scenario_mapping)} folders.")
        continue
        
    # Format the frame ID to 6 digits (e.g., 0 -> "000000")
    frame_str = str(frame_idx).zfill(6)
    
    # --- 4. Find the file ---
    # Looking for: RAW_DATA_ROOT / [Any Split] / [Scenario Name] / 0 / [Frame].pcd
    # We use "0" for the agent ID as that represents the Ego vehicle.
    search_pattern = os.path.join(RAW_DATA_ROOT, "*", scenario_name, "0", f"{frame_str}.pcd")
    matching_files = glob.glob(search_pattern)
    
    print(f"QA ID: {qa_id} | Scenario Index: {scenario_idx} | Frame Index: {frame_idx}")
    if matching_files:
        pcd_path = matching_files[0]
        print(f"✅ Found Match!")
        print(f"   Point Cloud: {pcd_path}\n")
    else:
        print(f"❌ Could not find raw file for Scenario: {scenario_name}, Frame: {frame_str}\n")