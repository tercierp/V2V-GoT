#!/bin/bash
# Run nq1-only inference with the OSM-enabled CRAFTER checkpoint.
# Usage (from LLaVA/ dir):
#   source scripts/v1_5/inference_osm_crafter.sh <ckpt_number> <data_tag>
# Example:
#   source scripts/v1_5/inference_osm_crafter.sh 1000 nq1sm3w0d gt

conda activate llava
export PYTHONPATH=$PWD:$PYTHONPATH

MODEL="crafter_osm_crafter_train01"
CKPT_NUMBER=${1:-1000}
DATA=${2:-nq1sm3w0d}
DATA_SOURCE=${3:-gt}  # 'gt' for ground-truth Q file, 'graph' for GoT-chain Q file

SAT_IMAGES_ROOT="/scratch/izar/tercier/v2v-got/sat_images"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
SPLIT="val"
EXP="v2v4real_3d_grounding_${MODEL}_${CKPT_NUMBER}_full_${DATA}"
MODEL_PATH="./checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_${MODEL}/checkpoint-${CKPT_NUMBER}"

echo "EXP:        $EXP"
echo "MODEL_PATH: $MODEL_PATH"

if [ "$DATA_SOURCE" == "graph" ]; then
    QUESTION_FILE="./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/${DATA}.json"
else
    QUESTION_FILE="../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_${DATA}.json"
fi
echo "QUESTION_FILE: $QUESTION_FILE"

for IDX in $(seq 0 $((CHUNKS-1))); do
    ANSWER_FILE="./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl"
    echo "ANSWER_FILE: $ANSWER_FILE"

    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
        --model-base liuhaotian/llava-v1.5-7b \
        --mm_scene_projector_input_size 3072 \
        --scene_level_only False \
        --object_level_only False \
        --scene_feature_mode shallow \
        --object_feature_mode shallow \
        --num_input_frames 2 \
        --ego_only False \
        --feature_source no_fusion_keep_all \
        --use_osm True \
        --osm_image_folder "$SAT_IMAGES_ROOT" \
        --question-file "$QUESTION_FILE" \
        --answers-file "$ANSWER_FILE" \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file="./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/merge.jsonl"
> "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "./playground/data/eval/$EXP/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done
echo "Merged answers → $output_file"
