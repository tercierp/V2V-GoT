#!/bin/bash
#SBATCH --job-name=v2vgot-osm-finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --output=/scratch/izar/%u/v2v-got/logs/%x-%j.out
#SBATCH --error=/scratch/izar/%u/v2v-got/logs/%x-%j.err

# Fine-tune V2V-GoT pretrained checkpoint (checkpoint-4330) with OSM token
# injection enabled, for 3 epochs on Q9 (planning) data.
#
# Strategy:
# - Start from the verified pretrained LoRA checkpoint
# - Train OSM encoder from scratch (~1.4M new params)
# - Continue LoRA training (existing adapters keep learning)
# - Effective batch size 32 maintained via grad accumulation (2 GPUs * 2 per_device * 8 grad_accum)
# - Output saved to a NEW model name to avoid overwriting baseline

set -euo pipefail

echo "=== Job info ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
echo

# === Environment ===
module purge
module load gcc cuda/11.8

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llava

ulimit -n 65536
export HF_HOME=/scratch/izar/$USER/hf_cache
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export TOKENIZERS_PARALLELISM=false

nvidia-smi
echo
echo "=== ulimit -n ==="
ulimit -n
echo

# === Run training ===
cd /scratch/izar/$USER/v2v-got/V2V-GoT
mkdir -p /scratch/izar/$USER/v2v-got/logs

# Output model name: distinct from baseline so we don't overwrite checkpoint-4330
OUTPUT_MODEL="osm_q9_v2vgot_3ep"
OUTPUT_DIR="LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_${OUTPUT_MODEL}"

# Path to pre-computed OSM features (from the precompute step)
OSM_FEATURES_ROOT=/scratch/izar/$USER/v2v-got/osm_features

# Path to the verified pretrained LoRA checkpoint (your baseline)
PRETRAINED_LORA=LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2vgot_10ep_both_shallow_f2/checkpoint-4330

echo "OSM features dir: $OSM_FEATURES_ROOT"
echo "Files in dir:     $(ls $OSM_FEATURES_ROOT | wc -l)"
echo "Pretrained LoRA:  $PRETRAINED_LORA"
echo "Output dir:       $OUTPUT_DIR"
echo

cd LLaVA
export PYTHONPATH=$PWD:$PYTHONPATH

deepspeed --num_gpus=2 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --pretrain_mm_mlp_adapter ${PRETRAINED_LORA}/non_lora_trainables.bin \
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
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --osm_features_root ${OSM_FEATURES_ROOT} \
    --use_osm_tokens True \
    --osm_num_tokens 4 \
    --osm_in_channels 4 \
    --bf16 False \
    --fp16 True \
    --output_dir ../${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

echo
echo "=== Finished ==="
echo "Ended: $(date)"
