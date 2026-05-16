#!/bin/bash
# Train V2V-GoT + OSM encoder on the new CRAFTER dataset (satellite images from faresse).
#
# Usage (from LLaVA/ dir, after activating llava conda env):
#   source scripts/v1_5/train_osm_crafter.sh
#
# Paths to update per user / cluster:
#   --data_path : QA JSON for CRAFTER (update once the JSON is generated)
#   --osm_image_folder : path to the sat_images root on scratch
#   --crafter_split    : which train split folder to use (e.g. train_01)

cd LLaVA
conda activate llava
nvidia-smi
export PYTHONPATH=$PWD:$PYTHONPATH
echo $PYTHONPATH

# ── Paths ─────────────────────────────────────────────────────────────────────
# UPDATE these to match your actual paths on Izar:
SAT_IMAGES_ROOT="/scratch/izar/tercier/v2v-got/sat_images"
CRAFTER_SPLIT="train_01"

# QA JSON for CRAFTER — update this once faresse's QA JSON is available.
# For now we reuse the V2V4Real QA JSON as a placeholder so the script can
# at least verify the model loads correctly. Replace with the CRAFTER JSON.
DATA_PATH="../DMSTrack/V2V4Real/official_models/train_no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v2vgot.json"
# TODO: replace with CRAFTER QA JSON once generated, e.g.:
# DATA_PATH="/scratch/tercier/v2v-got/data/crafter_qa_dataset.json"

MODEL="osm_crafter_train01"

# ── Training ──────────────────────────────────────────────────────────────────
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path "$DATA_PATH" \
    --mm_scene_projector_input_size 3072 \
    --scene_level_only False \
    --object_level_only False \
    --scene_feature_mode shallow \
    --object_feature_mode shallow \
    --num_input_frames 2 \
    --ego_only False \
    --feature_source no_fusion_keep_all \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_crafter_$MODEL \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --use_osm True \
    --osm_image_folder "$SAT_IMAGES_ROOT" \
    --crafter_split "$CRAFTER_SPLIT"
