#!/bin/bash
export PYTHONPATH=$PWD:$PYTHONPATH

MODEL="v2vgot_satdesc"
CKPT_NUMBER=$1
DATA=$2
DATA_SOURCE=$3
GRAPH=$4

NUM_LATENCY_FRAMES=0
POSITIONAL_ERROR_10_STD=0
CKPT="llava-v1.5-7b"
SPLIT='val'

for CKPT_ID in $CKPT_NUMBER
do
    EXP="v2v4real_3d_grounding_${MODEL}_${CKPT_ID}_${GRAPH}_${DATA}"
    echo $EXP
    MODEL_PATH="./checkpoints/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_${MODEL}/checkpoint-${CKPT_ID}"
    echo $MODEL_PATH

    if [ "$DATA_SOURCE" == "graph" ]; then
      QUESTION_FILE="./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/${DATA}.json"
    else
      # Fixed the path to correctly point to where the JSON QA datasets actually reside
      QUESTION_FILE="/scratch/izar/$USER/v2v-got/data/V2V-GoT-QA/V2V_GoT_JSONS/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_${DATA}.json"
    fi
    echo $QUESTION_FILE

    # Run inference sequentially on 1 node, 1 process
    ANSWER_FILE=./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/1_0.jsonl
    mkdir -p $(dirname $ANSWER_FILE)

    CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --mm_scene_projector_input_size 3072 \
        --scene_level_only False \
        --object_level_only False \
        --scene_feature_mode shallow \
        --object_feature_mode shallow \
        --num_input_frames 2 \
        --ego_only False \
        --feature_source no_fusion_keep_all \
        --num_latency_frames $NUM_LATENCY_FRAMES \
        --positional_error_10_std $POSITIONAL_ERROR_10_STD \
        --model-base liuhaotian/llava-v1.5-7b \
        --question-file "$QUESTION_FILE" \
        --answers-file "$ANSWER_FILE" \
        --num-chunks 1 \
        --chunk-idx 0 \
        --temperature 0 \
        --conv-mode vicuna_v1

    # Merge answers (only 1 chunk)
    output_file=./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/merge.jsonl
    > "$output_file"
    cat "$ANSWER_FILE" >> "$output_file"
done
