#!/bin/bash

cd LLaVA
conda activate llava
nvidia-smi
export PYTHONPATH=$PWD:$PYTHONPATH
echo $PYTHONPATH

MODEL="v2vgot_10ep_both_shallow_f2_osm"

# OSM images: one 256x256 PNG per scenario, named {seq_id}.png (e.g. 0000.png)
# Generate with plot_satellite.py and place in this folder before launching
OSM_IMAGE_FOLDER="../DMSTrack/V2V4Real/osm_images"

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path ../DMSTrack/V2V4Real/official_models/train_no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v2vgot.json \
    --mm_scene_projector_input_size 3072 \
    --scene_level_only False \
    --object_level_only False \
    --scene_feature_mode shallow \
    --object_feature_mode shallow \
    --num_input_frames 2 \
    --ego_only False \
    --feature_source no_fusion_keep_all \
    --use_osm True \
    --osm_image_folder $OSM_IMAGE_FOLDER \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_$MODEL \
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
    --report_to wandb
