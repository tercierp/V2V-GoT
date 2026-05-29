import argparse
import bisect
import json
import os
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge satellite image descriptions into V2V-GoT QA JSONs."
    )
    parser.add_argument(
        "--sat-jsonl",
        type=str,
        default="data/sat_images_descriptions.jsonl",
        help="Path to JSONL descriptions file.",
    )
    parser.add_argument(
        "--qa-dir",
        type=str,
        default="data/V2V_GoT_JSONS/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm",
        help="Directory containing V2V-GoT QA JSONs to modify.",
    )
    parser.add_argument(
        "--qa-datasets",
        nargs="+",
        required=True,
        help="QA dataset names, e.g., nq1sm3w0d nq2sm3w0d",
    )
    parser.add_argument(
        "--data-root",
        default="/scratch/izar/faresse/v2v-got/data/V2V4REAL/V2V4REAL/Data",
        help="V2V4Real Data root containing split folders (train_01/test_01/etc.)",
    )
    parser.add_argument(
        "--split-names",
        default="test_01,test_02,test_03",
        help="Comma-separated split names to align with QA scenario_index ordering",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=30,
        help="Frame step used in satellite export (for diagnostics only)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite QA JSON files instead of writing *_satdesc.json",
    )
    return parser.parse_args()


def load_sat_descriptions(path: str) -> Dict[Tuple[str, str, int, str], str]:
    desc_map: Dict[Tuple[str, str, int, str], str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            split_name = item.get("split")
            scenario_name = item.get("sequence")
            frame_str = item.get("frame")
            filename = item.get("filename")
            desc = item.get("description")
            if desc is None:
                desc = ""
            desc = str(desc).strip()
            if not (split_name and scenario_name and frame_str and filename and desc):
                continue
            try:
                frame_id = int(frame_str)
            except ValueError:
                continue
            if filename.startswith("agent_"):
                agent_id = filename.split("agent_")[-1].split(".")[0]
            else:
                agent_id = "0"
            desc_map[(split_name, scenario_name, frame_id, agent_id)] = desc
    return desc_map


def build_scenario_index(data_root: str, split_names: List[str]) -> List[Tuple[int, str, str]]:
    scenario_index = []
    idx = 0

    # Accept either the V2V4Real root or the Data folder directly.
    if not any(os.path.isdir(os.path.join(data_root, s)) for s in split_names):
        data_root = os.path.join(data_root, "Data")

    for split_name in split_names:
        split_dir = os.path.join(data_root, split_name)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split not found: {split_dir}")
        scenario_dirs = sorted(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        )
        for scenario_name in scenario_dirs:
            scenario_index.append((idx, split_name, scenario_name))
            idx += 1
    return scenario_index


def build_frame_index(
    desc_map: Dict[Tuple[str, str, int, str], str]
) -> Dict[Tuple[str, str, str], List[Tuple[int, str]]]:
    frame_index: Dict[Tuple[str, str, str], List[Tuple[int, str]]] = {}
    for (split_name, scenario_name, frame_id, agent_id), desc in desc_map.items():
        key = (split_name, scenario_name, agent_id)
        frame_index.setdefault(key, []).append((frame_id, desc))
    for key in frame_index:
        frame_index[key].sort(key=lambda x: x[0])
    return frame_index


def nearest_description(
    entries: List[Tuple[int, str]], target_frame: int
) -> Tuple[str, int, str]:
    frames = [f for f, _ in entries]
    pos = bisect.bisect_left(frames, target_frame)
    if pos == 0:
        f, d = entries[0]
        return d, f, "next"
    if pos >= len(entries):
        f, d = entries[-1]
        return d, f, "prev"
    prev_f, prev_d = entries[pos - 1]
    next_f, next_d = entries[pos]
    if (target_frame - prev_f) <= (next_f - target_frame):
        return prev_d, prev_f, "prev"
    return next_d, next_f, "next"


def map_cav_to_agent_id(cav_id: str) -> str:
    if cav_id in ("ego", "0", 0):
        return "0"
    return str(cav_id)


def inject_description(prompt: str, desc: str) -> str:
    prefix = f"Satellite scene: {desc}\n"
    return prefix + prompt


def main() -> None:
    args = parse_args()

    split_names = [s.strip() for s in args.split_names.split(",") if s.strip()]
    scenario_index_map = build_scenario_index(args.data_root, split_names)
    scenario_index_lookup = {idx: (split, scenario) for idx, split, scenario in scenario_index_map}

    desc_map = load_sat_descriptions(args.sat_jsonl)
    frame_index = build_frame_index(desc_map)

    for dataset_name in args.qa_datasets:
        qa_path = os.path.join(
            args.qa_dir,
            f"v2v4real_3d_grounding_qa_dataset_{dataset_name}.json",
        )
        if not os.path.exists(qa_path):
            raise FileNotFoundError(f"QA dataset not found: {qa_path}")

        with open(qa_path, "r") as f:
            qa_data = json.load(f)

        missing = 0
        used = 0
        for item in qa_data:
            scenario_index = item.get("scenario_index")
            local_ts = item.get("local_timestamp_index")
            cav_id = item.get("asker_cav_id", "ego")

            if scenario_index is None or local_ts is None:
                missing += 1
                continue

            if scenario_index not in scenario_index_lookup:
                missing += 1
                continue

            split_name, scenario_name = scenario_index_lookup[scenario_index]
            agent_id = map_cav_to_agent_id(cav_id)
            key = (split_name, scenario_name, agent_id)

            entries = frame_index.get(key)
            if not entries:
                missing += 1
                continue

            desc, src_frame, mode = nearest_description(entries, int(local_ts))
            prompt = item["conversations"][0]["value"]
            if prompt.startswith("Satellite scene:"):
                # Avoid double-injecting
                used += 1
                continue

            item["conversations"][0]["value"] = inject_description(prompt, desc)
            item["sat_description"] = desc
            item["sat_description_source"] = {
                "split": split_name,
                "scenario": scenario_name,
                "agent_id": agent_id,
                "src_frame": src_frame,
                "mode": mode,
            }
            used += 1

        if args.inplace:
            out_path = qa_path
        else:
            out_path = qa_path.replace(".json", "_satdesc.json")

        with open(out_path, "w") as f:
            json.dump(qa_data, f)

        print(
            f"{dataset_name}: wrote {out_path} | updated={used} missing={missing}"
        )


if __name__ == "__main__":
    main()
