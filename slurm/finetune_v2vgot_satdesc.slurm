#!/bin/bash
#SBATCH --job-name=v2vgot-train-satdesc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --output=/scratch/izar/%u/v2v-got/logs/%x-%j.out
#SBATCH --error=/scratch/izar/%u/v2v-got/logs/%x-%j.err

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Nodes:     $SLURM_JOB_NUM_NODES"
echo "Nodelist:  $SLURM_JOB_NODELIST"
echo "Started:   $(date)"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo

# --- Environment setup ---
module purge
module load gcc cuda/11.8

# Initialize conda for env switching
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate v2v

# File descriptor limit (needed for DataLoader)
ulimit -n 65536

# HuggingFace cache on scratch
export HF_HOME=/scratch/izar/$USER/hf_cache

# Resolve repo root explicitly to avoid issues depending on where sbatch is run from
V2VGOT_ROOT="/scratch/izar/$USER/v2v-got/cs503-v2v-project/code/V2V-GoT"

# PYTHONPATH
export PYTHONPATH=${PYTHONPATH:-}:$V2VGOT_ROOT/DMSTrack/AB3DMOT
export PYTHONPATH=${PYTHONPATH}:$V2VGOT_ROOT/DMSTrack/AB3DMOT/Xinshuo_PyToolbox
export PYTHONPATH=${PYTHONPATH}:$V2VGOT_ROOT/DMSTrack/V2V4Real
export PYTHONPATH=${PYTHONPATH}:$V2VGOT_ROOT/DMSTrack
export PYTHONPATH=${PYTHONPATH}:$V2VGOT_ROOT/LLaVA

# Prevent tokenizer warnings from spamming logs
export TOKENIZERS_PARALLELISM=false

# --- Distributed Setup ---
export MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w $MASTER_NODE hostname -i)
export MASTER_PORT=29500

echo "MASTER_NODE: $MASTER_NODE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo

# --- Sanity checks ---
nvidia-smi
echo
echo "PYTHONPATH: $PYTHONPATH"
echo "HF_HOME:    $HF_HOME"
echo "ulimit -n:  $(ulimit -n)"
echo

echo "=== Starting V2V-GoT Finetuning pipeline (with sat descriptions) ==="
echo "Time: $(date)"
echo

cd "$V2VGOT_ROOT/LLaVA"
srun bash scripts/v1_5/finetune_v2vgot_satdesc.sh

echo
echo "=== Pipeline finished ==="
echo "Time: $(date)"
