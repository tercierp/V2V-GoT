#!/bin/bash
#SBATCH --job-name=v2vgot-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=14:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --output=/scratch/izar/%u/v2v-got/logs/%x-%j.out
#SBATCH --error=/scratch/izar/%u/v2v-got/logs/%x-%j.err

# set -euo pipefail  # disabled: inner script has harmless conda errors

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo

# --- Environment setup ---
module purge
module load gcc cuda/11.8

# Initialize conda for env switching (critical — inference_v2vgot.sh
# calls conda activate/deactivate repeatedly)
source "$(conda info --base)/etc/profile.d/conda.sh"

# File descriptor limit (needed for DataLoader)
ulimit -n 65536

# HuggingFace cache on scratch
export HF_HOME=/scratch/izar/$USER/hf_cache

# PYTHONPATH for dmstrack's temp_qa_generation.py calls
export PYTHONPATH=${PYTHONPATH:-}:/scratch/izar/$USER/v2v-got/V2V-GoT/DMSTrack/AB3DMOT
export PYTHONPATH=${PYTHONPATH}:/scratch/izar/$USER/v2v-got/V2V-GoT/DMSTrack/AB3DMOT/Xinshuo_PyToolbox
export PYTHONPATH=${PYTHONPATH}:/scratch/izar/$USER/v2v-got/V2V-GoT/DMSTrack/V2V4Real
export PYTHONPATH=${PYTHONPATH}:/scratch/izar/$USER/v2v-got/V2V-GoT/DMSTrack

# Prevent tokenizer warnings from spamming logs
export TOKENIZERS_PARALLELISM=false

# --- Sanity checks ---
nvidia-smi
echo
echo "PYTHONPATH: $PYTHONPATH"
echo "HF_HOME:    $HF_HOME"
echo "ulimit -n:  $(ulimit -n)"
echo

# --- Run full 9-stage GoT inference pipeline ---
cd /scratch/izar/$USER/v2v-got/V2V-GoT
mkdir -p /scratch/izar/$USER/v2v-got/logs

echo "=== Starting full V2V-GoT inference pipeline ==="
echo "Time: $(date)"
echo

source LLaVA/scripts/v1_5/inference_v2vgot.sh

echo
echo "=== Pipeline finished ==="
echo "Time: $(date)"
