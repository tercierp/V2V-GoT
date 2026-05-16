#!/bin/bash
# ixl variant: 4× V100 GPUs on gpu-xl partition
# Effective batch size kept at 64: 4 GPUs × bs=2 × accum=8 = 64 (same as 2×2×16)

deepspeed --num_gpus 2 --master_port 29999 \
    llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3_resume.json \
    --model_name_or_path /scratch/izar/tercier/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234 \
    --version v1 \
    --data_path ../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json \
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
    --fp16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_crafter_osm_crafter_train01 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --use_osm True \
    --osm_image_folder /scratch/izar/tercier/v2v-got/sat_images \
    --train_data_split val \
    --crafter_split train_01 \
    --resume_from_checkpoint True
