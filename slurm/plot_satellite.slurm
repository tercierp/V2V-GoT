#!/bin/bash
#SBATCH --job-name=plot-satellite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --output=/scratch/izar/%u/v2v-got/logs/%x-%j.out
#SBATCH --error=/scratch/izar/%u/v2v-got/logs/%x-%j.err

set -euo pipefail

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo

module purge
module load gcc cuda/11.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate v2v

ulimit -n 65536

export HF_HOME=/scratch/izar/$USER/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /scratch/izar/$USER/v2v-got/data
mkdir -p /scratch/izar/$USER/v2v-got/logs

echo "=== Running plot_satellite.py ==="
python -u plot_satellite.py

echo

echo "=== Finished ==="
echo "Ended: $(date)"
