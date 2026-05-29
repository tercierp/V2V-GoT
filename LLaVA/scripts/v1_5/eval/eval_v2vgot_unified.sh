#!/bin/bash

cd DMSTrack/DMSTrack
source dmstrack_init_env.sh
cd ../../LLaVA

MODEL=$1
CKPT_ID_INPUT=$2
GRAPH=$3
COMM_MB=$4

RUN_NAME="${MODEL}_${CKPT_ID_INPUT}_${GRAPH}"

python scripts/eval_v2vgot_unified.py \
  --inference_root playground/data/eval \
  --run_name "$RUN_NAME" \
  --gt_json ../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v2vgot.json \
  --gt_npy_root ../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy \
  --comm_mb "$COMM_MB" \
  --out_dir results
