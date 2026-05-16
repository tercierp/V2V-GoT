#!/bin/bash
#SBATCH --job-name=v2vgot-osm-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=14:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --output=/scratch/izar/tercier/v2v-got/logs/%x-%j.out
#SBATCH --error=/scratch/izar/tercier/v2v-got/logs/%x-%j.err

# Usage: sbatch submit_inference_osm_crafter.sh <ckpt_number> <data_tag>
# Defaults: ckpt=1000, data=nq1sm3w0d (ground-truth Q1)
CKPT_NUMBER=${1:-1000}
DATA=${2:-nq1sm3w0d}

echo "=== Job info ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "CKPT:     $CKPT_NUMBER"
echo "DATA:     $DATA"
echo

module purge
module load gcc cuda/11.8

source "$(conda info --base)/etc/profile.d/conda.sh"

ulimit -n 65536
export HF_HOME=/scratch/izar/$USER/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi
echo

cd /scratch/izar/tercier/v2v-got/V2V-GoT/LLaVA
conda activate llava
export PYTHONPATH=$PWD:$PYTHONPATH

echo "=== Starting OSM CRAFTER inference ==="
bash scripts/v1_5/inference_osm_crafter.sh "$CKPT_NUMBER" "$DATA" gt

echo
echo "=== Done ==="
echo "Ended: $(date)"
