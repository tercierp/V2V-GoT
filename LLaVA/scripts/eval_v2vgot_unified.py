import argparse
import csv
import json
import os
import re
import sys
import random
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from eval_metrics import (
    evaluate_future_trajectory,
    evaluate_is_another_cav_notable_object,
    evaluate_notable_objects_prediction,
    evaluate_suggested_speed_steering,
    parse_notable_object_prediction_answer,
    parse_planned_future_trajectory,
    parse_suggested_speed_steering_idx,
)

Q_TO_QA_TYPE_ID = {
    1: 11,
    2: 12,
    3: 13,
    4: 14,
    5: 15,
    6: 16,
    7: 17,
    8: 18,
    9: 19,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified V2V-GoT LLM QA evaluator.")
    parser.add_argument(
        "--inference_root",
        default="LLaVA/playground/data/eval",
        help="Root folder containing *_nq* inference outputs.",
    )
    parser.add_argument(
        "--run_name",
        required=True,
        help="Run name used for output filenames and tables.",
    )
    parser.add_argument(
        "--run_variant",
        default="baseline",
        help="Variant of the run (e.g. baseline, bev30).",
    )
    parser.add_argument(
        "--gt_json",
        required=True,
        help="Path to v2v4real_3d_grounding_qa_dataset_v2vgot.json.",
    )
    parser.add_argument(
        "--gt_npy_root",
        required=True,
        help="Path to the V2V4Real npy root used by planning metrics.",
    )
    parser.add_argument(
        "--comm_mb",
        type=float,
        default=0.0,
        help="Communication cost (MB) to display in tables.",
    )
    parser.add_argument(
        "--out_dir",
        default="LLaVA/results",
        help="Output directory for CSV tables.",
    )
    parser.add_argument(
        "--qa_subtypes",
        default=None,
        help="Comma-separated list of QA types to evaluate (e.g., Q4,Q7,Q9).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=2.0,
        help="Matching radius (meters) for F1 selection in Q1-4.",
    )
    parser.add_argument(
        "--r_collision",
        type=float,
        default=None,
        help="Collision radius (unused; collision uses 3D IOU per official code).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on parsing errors instead of logging and continuing.",
    )
    return parser.parse_args()


def normalize_qa_sub_type(value) -> Tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    return (int(value),)


def get_dataset_info_from_path(path: str) -> Tuple[int, str]:
    base = os.path.basename(path)
    match = re.search(r"_nq(\d+[^/]*)$", base)
    if not match:
        return -1, ""
    dataset = match.group(0)[1:]
    q_match = re.match(r"nq(\d+)", dataset)
    if not q_match:
        return -1, dataset
    q_num = int(q_match.group(1))
    return q_num, dataset


def discover_inference_outputs(inference_root: str) -> List[Tuple[str, str]]:
    outputs = []
    for root, dirs, files in os.walk(inference_root):
        if not root.endswith(os.path.join("answers", "val", "llava-v1.5-7b")):
            continue
        run_dir = os.path.dirname(os.path.dirname(os.path.dirname(root)))
        q_num, dataset = get_dataset_info_from_path(run_dir)
        if q_num == -1:
            continue
        merge_file = os.path.join(root, "merge.jsonl")
        if os.path.isfile(merge_file):
            outputs.append((merge_file, dataset))
            continue
        for fname in files:
            if fname.endswith(".jsonl"):
                outputs.append((os.path.join(root, fname), dataset))
    return outputs


def read_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_gt_index(gt_data: List[dict]) -> Tuple[Dict[Tuple[str, int], dict], Dict[Tuple[int, int], dict]]:
    by_source = {}
    by_type = {}
    for item in gt_data:
        qa_source = item.get("qa_source")
        qa_type_id = item.get("qa_type_id")
        item_id = item.get("id")
        if qa_source is not None and item_id is not None:
            by_source[(qa_source, int(item_id))] = item
        if qa_type_id is not None and item_id is not None:
            by_type[(int(qa_type_id), int(item_id))] = item
    return by_source, by_type


def load_eval_samples(
    outputs: List[dict],
    qa_source: str,
    qa_type_id: int,
    gt_by_source: Dict[Tuple[str, int], dict],
    gt_by_type: Dict[Tuple[int, int], dict],
    parse_failures: List[dict],
    strict: bool,
) -> List[dict]:
    samples = []
    for pred in outputs:
        pred_id = pred.get("id")
        if pred_id is None:
            continue
        pred_id = int(pred_id)
        gt_item = gt_by_source.get((qa_source, pred_id))
        if gt_item is None:
            gt_item = gt_by_type.get((qa_type_id, pred_id))
        if gt_item is None:
            parse_failures.append({
                "reason": "gt_missing",
                "id": pred_id,
                "qa_source": qa_source,
                "qa_type_id": qa_type_id,
                "raw_prediction": pred.get("outputs"),
            })
            if strict:
                raise ValueError(f"GT missing for id {pred_id} ({qa_source})")
            continue
        sample = dict(gt_item)
        sample["outputs"] = pred.get("outputs", pred.get("text", ""))
        samples.append(sample)
    return samples


def log_parse_failure(parse_failures: List[dict], sample: dict, reason: str) -> None:
    parse_failures.append({
        "reason": reason,
        "global_timestamp_index": sample.get("global_timestamp_index"),
        "asker_cav_id": sample.get("asker_cav_id"),
        "qa_sub_type": normalize_qa_sub_type(sample.get("qa_sub_type")),
        "qa_type_id": sample.get("qa_type_id"),
        "qa_source": sample.get("qa_source"),
        "raw_prediction": sample.get("outputs"),
    })


def has_speed_label(text: str) -> bool:
    labels = ["fast", "moderate", "slow", "very slow", "stop"]
    return any(label in text for label in labels)


def has_steering_label(text: str) -> bool:
    labels = ["left", "slightly left", "straight", "slightly right", "right"]
    return any(label in text for label in labels)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.gt_json, "r") as f:
        gt_data = json.load(f)

    gt_by_source, gt_by_type = build_gt_index(gt_data)

    outputs = discover_inference_outputs(args.inference_root)
    if not outputs:
        raise FileNotFoundError(f"No inference outputs found under {args.inference_root}")

    qa_filter = None
    if args.qa_subtypes:
        qa_filter = {item.strip().upper() for item in args.qa_subtypes.split(",")}

    parse_failures = []
    q_data: Dict[int, List[dict]] = {q: [] for q in range(1, 10)}

    for output_file, qa_source in outputs:
        q_num_match = re.match(r"nq(\d+)", qa_source)
        if not q_num_match:
            continue
        q_num = int(q_num_match.group(1))
        if q_num not in Q_TO_QA_TYPE_ID:
            continue
        if qa_filter and f"Q{q_num}" not in qa_filter:
            continue

        qa_type_id = Q_TO_QA_TYPE_ID[q_num]
        predictions = read_jsonl(output_file)
        samples = load_eval_samples(
            predictions,
            qa_source,
            qa_type_id,
            gt_by_source,
            gt_by_type,
            parse_failures,
            args.strict,
        )
        for sample in samples:
            sample["qa_type_id"] = qa_type_id
            q_data[q_num].append(sample)

    # --- BEV Injection Diagnostic (reads per-Q satdesc files used at inference) ---
    import glob
    qa_dir = os.path.dirname(args.gt_json)
    print("\n================================================================")
    print("BEV injection diagnostic (per-Q satdesc files)")
    print("================================================================")
    warning_triggered = False
    diagnostics = {}

    for q_num in range(1, 10):
        sat_files = glob.glob(os.path.join(qa_dir, f"v2v4real_3d_grounding_qa_dataset_nq{q_num}sm*_satdesc.json"))
        if not sat_files:
            print(f"Q{q_num}: NO _satdesc.json found")
            warning_triggered = True
            diagnostics[f"Q{q_num}"] = {"status": "missing"}
            continue
        sat_path = sat_files[0]
        with open(sat_path, "r") as f:
            sat_data = json.load(f)
        n_total = len(sat_data)
        n_prefixed = sum(1 for x in sat_data if x["conversations"][0]["value"].startswith("Satellite scene:"))
        n_dud = sum(1 for x in sat_data if "not available" in x["conversations"][0]["value"][:200].lower())
        rate = n_prefixed / n_total if n_total else 0.0
        print(f"Q{q_num}: {n_prefixed}/{n_total} prefixed ({rate:.1%})  |  {n_dud} dud captions  |  {os.path.basename(sat_path)}")
        diagnostics[f"Q{q_num}"] = {
            "prefixed": n_prefixed,
            "total": n_total,
            "rate": rate,
            "dud_captions": n_dud,
            "file": os.path.basename(sat_path),
        }
        if rate < 0.5:
            warning_triggered = True

    diag_out = os.path.join(args.out_dir, f"{args.run_name}_diagnostics.json")
    with open(diag_out, "w") as f:
        json.dump({"bev_injection": diagnostics}, f, indent=2)

    if warning_triggered:
        print("\nNOTE: BEV injection rate is below 50% for one or more Q stages. "
              "Numbers below reflect partial BEV coverage.\n")
    print("================================================================\n") 

    table2 = {
        "Q1_F1": None,
        "Q2_F1": None,
        "Q3_F1": None,
        "Q4_F1": None,
        "Q5_L2": None,
        "Q6_Acc": None,
        "Q7_L2": None,
        "Q8_L1": None,
        "Q9_L2": None,
        "Q9_CR": None,
    }

    # Q1-4
    for q_num in [1, 2, 3, 4]:
        data = q_data[q_num]
        if not data:
            continue
        for sample in data:
            gt_answer = sample["conversations"][1]["value"]
            out_answer = sample.get("outputs", "")
            _, _, _, gt_loc_err, _, gt_traj_err = parse_notable_object_prediction_answer(
                Q_TO_QA_TYPE_ID[q_num], gt_answer, 3, 0
            )
            _, _, _, out_loc_err, _, out_traj_err = parse_notable_object_prediction_answer(
                Q_TO_QA_TYPE_ID[q_num], out_answer, 3, 0
            )
            if gt_loc_err:
                log_parse_failure(parse_failures, sample, "q1_4_gt_parse_location")
            if out_loc_err:
                log_parse_failure(parse_failures, sample, "q1_4_output_parse_location")
            if gt_traj_err or out_traj_err:
                log_parse_failure(parse_failures, sample, "q1_4_parse_trajectory")
        metrics = evaluate_notable_objects_prediction(
            max_num_answer_objects=3,
            num_future_waypoints=0,
            data=data,
            npy_save_path=args.gt_npy_root,
            qa_type_id=Q_TO_QA_TYPE_ID[q_num],
            match_threshold=10000,
        )
        tau_value = min(metrics["f1_per_threshold"].keys(), key=lambda x: abs(x - args.tau))
        table2[f"Q{q_num}_F1"] = metrics["f1_per_threshold"][tau_value]

    # Q5
    if q_data[5]:
        filtered = []
        for sample in q_data[5]:
            gt_answer = sample["conversations"][1]["value"]
            out_answer = sample.get("outputs", "")
            _, _, _, gt_loc_err, gt_act_err, gt_traj_err = parse_notable_object_prediction_answer(
                Q_TO_QA_TYPE_ID[5], gt_answer, 3, 1
            )
            _, _, _, out_loc_err, out_act_err, out_traj_err = parse_notable_object_prediction_answer(
                Q_TO_QA_TYPE_ID[5], out_answer, 3, 1
            )
            if gt_loc_err or gt_act_err or gt_traj_err:
                log_parse_failure(parse_failures, sample, "q5_gt_parse")
                if args.strict:
                    raise ValueError("Q5 GT parse failure")
                continue
            if out_loc_err or out_act_err or out_traj_err:
                log_parse_failure(parse_failures, sample, "q5_output_parse")
                if args.strict:
                    raise ValueError("Q5 output parse failure")
                continue
            filtered.append(sample)
        metrics = evaluate_notable_objects_prediction(
            max_num_answer_objects=3,
            num_future_waypoints=1,
            data=filtered,
            npy_save_path=args.gt_npy_root,
            qa_type_id=Q_TO_QA_TYPE_ID[5],
            match_threshold=10000,
        )
        table2["Q5_L2"] = metrics["l2_error_avg_03_all"] or metrics["l2_error_avg_3s"]

    # Q6
    if q_data[6]:
        for sample in q_data[6]:
            output = sample.get("outputs", "")
            if "not notable" not in output and "is a notable" not in output:
                log_parse_failure(parse_failures, sample, "q6_output_parse")
                if args.strict:
                    raise ValueError("Q6 output parse failure")
        table2["Q6_Acc"] = evaluate_is_another_cav_notable_object(q_data[6])

    # Q7
    if q_data[7]:
        filtered = []
        for sample in q_data[7]:
            gt_answer = sample["conversations"][1]["value"]
            out_answer = sample.get("outputs", "")
            _, _, _, gt_loc_err, gt_act_err, gt_traj_err = parse_notable_object_prediction_answer(
                Q_TO_QA_TYPE_ID[7], gt_answer, 3, 1
            )
            _, _, _, out_loc_err, out_act_err, out_traj_err = parse_notable_object_prediction_answer(
                Q_TO_QA_TYPE_ID[7], out_answer, 3, 1
            )
            if gt_loc_err or gt_act_err or gt_traj_err:
                log_parse_failure(parse_failures, sample, "q7_gt_parse")
                if args.strict:
                    raise ValueError("Q7 GT parse failure")
                continue
            if out_loc_err or out_act_err or out_traj_err:
                log_parse_failure(parse_failures, sample, "q7_output_parse")
                if args.strict:
                    raise ValueError("Q7 output parse failure")
                continue
            filtered.append(sample)
        metrics = evaluate_notable_objects_prediction(
            max_num_answer_objects=3,
            num_future_waypoints=1,
            data=filtered,
            npy_save_path=args.gt_npy_root,
            qa_type_id=Q_TO_QA_TYPE_ID[7],
            match_threshold=10000,
        )
        table2["Q7_L2"] = metrics["l2_error_avg_03_all"] or metrics["l2_error_avg_3s"]

    # Q8
    if q_data[8]:
        # Use official parser for indices, but apply max-penalty on parse failure.
        total_l1 = 0.0
        for sample in q_data[8]:
            gt_answer = sample["conversations"][1]["value"]
            out_answer = sample.get("outputs", "")

            gt_speed_idx, gt_steer_idx = parse_suggested_speed_steering_idx(gt_answer)

            if not has_speed_label(out_answer) or not has_steering_label(out_answer):
                log_parse_failure(parse_failures, sample, "q8_parse_error")
                l1 = 8.0
            else:
                out_speed_idx, out_steer_idx = parse_suggested_speed_steering_idx(out_answer)
                l1 = abs(gt_speed_idx - out_speed_idx) + abs(gt_steer_idx - out_steer_idx)
            total_l1 += l1

        table2["Q8_L1"] = total_l1 / len(q_data[8])
        # still run official helper to log its own stats
        evaluate_suggested_speed_steering(q_data[8], args.gt_npy_root)

    # Q9
    table1 = None
    if q_data[9]:
        filtered = []
        for sample in q_data[9]:
            gt_answer = sample["conversations"][1]["value"]
            out_answer = sample.get("outputs", "")
            _, gt_err = parse_planned_future_trajectory(gt_answer, 6, False, return_has_parsing_error=True)
            _, out_err = parse_planned_future_trajectory(out_answer, 6, False, return_has_parsing_error=True)
            if gt_err:
                log_parse_failure(parse_failures, sample, "q9_gt_parse_trajectory")
                if args.strict:
                    raise ValueError("Q9 GT parse failure")
                continue
            if out_err:
                log_parse_failure(parse_failures, sample, "q9_output_parse_trajectory")
                if args.strict:
                    raise ValueError("Q9 output parse failure")
                continue
            filtered.append(sample)
        metrics = evaluate_future_trajectory(
            num_future_waypoints=6,
            data=filtered,
            npy_save_path=args.gt_npy_root,
            qa_type_id=Q_TO_QA_TYPE_ID[9],
        )
        table2["Q9_L2"] = metrics["l2_error_avg_all"]
        table2["Q9_CR"] = metrics["collision_rate_avg_all"]
        table1 = {
            "L2@1s": metrics["l2_error_avg_1s"],
            "L2@2s": metrics["l2_error_avg_2s"],
            "L2@3s": metrics["l2_error_avg_3s"],
            "L2_avg": metrics["l2_error_avg_all"],
            "CR@1s": metrics["collision_rate_1s"],
            "CR@2s": metrics["collision_rate_2s"],
            "CR@3s": metrics["collision_rate_3s"],
            "CR_avg": metrics["collision_rate_avg_all"],
        }

    # Parse failure logging
    parse_failure_path = os.path.join(args.out_dir, f"{args.run_name}_parse_failures.jsonl")
    with open(parse_failure_path, "w") as f:
        for entry in parse_failures:
            f.write(json.dumps(entry) + "\n")

    # Write tables
    table1_path = os.path.join(args.out_dir, f"{args.run_name}_table1.csv")
    table2_path = os.path.join(args.out_dir, f"{args.run_name}_table2.csv")

    if table1 is not None:
        with open(table1_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Method",
                "Run_Variant",
                "L2@1s",
                "L2@2s",
                "L2@3s",
                "L2_avg",
                "CR@1s",
                "CR@2s",
                "CR@3s",
                "CR_avg",
                "Comm(MB)",
            ])
            writer.writerow([
                args.run_name,
                args.run_variant,
                table1["L2@1s"],
                table1["L2@2s"],
                table1["L2@3s"],
                table1["L2_avg"],
                table1["CR@1s"],
                table1["CR@2s"],
                table1["CR@3s"],
                table1["CR_avg"],
                args.comm_mb,
            ])

    with open(table2_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method",
            "Run_Variant",
            "Q1_F1",
            "Q2_F1",
            "Q3_F1",
            "Q4_F1",
            "Q5_L2",
            "Q6_Acc",
            "Q7_L2",
            "Q8_L1",
            "Q9_L2",
            "Q9_CR",
            "Comm(MB)",
        ])
        writer.writerow([
            args.run_name,
            args.run_variant,
            table2["Q1_F1"],
            table2["Q2_F1"],
            table2["Q3_F1"],
            table2["Q4_F1"],
            table2["Q5_L2"],
            table2["Q6_Acc"],
            table2["Q7_L2"],
            table2["Q8_L1"],
            table2["Q9_L2"],
            table2["Q9_CR"],
            args.comm_mb,
        ])

    # Print tables
    if table1 is not None:
        print("Table I (planning only)")
        print(
            f"{args.run_name}\t{table1['L2@1s']}\t{table1['L2@2s']}\t{table1['L2@3s']}\t"
            f"{table1['L2_avg']}\t{table1['CR@1s']}\t{table1['CR@2s']}\t{table1['CR@3s']}\t"
            f"{table1['CR_avg']}\t{args.comm_mb}"
        )

    print("Table II (all QA)")
    print(
        f"{args.run_name}\t{table2['Q1_F1']}\t{table2['Q2_F1']}\t{table2['Q3_F1']}\t"
        f"{table2['Q4_F1']}\t{table2['Q5_L2']}\t{table2['Q6_Acc']}\t{table2['Q7_L2']}\t"
        f"{table2['Q8_L1']}\t{table2['Q9_L2']}\t{table2['Q9_CR']}\t{args.comm_mb}"
    )


if __name__ == "__main__":
    main()
