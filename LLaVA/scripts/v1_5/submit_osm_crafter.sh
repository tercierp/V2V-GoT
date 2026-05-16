#!/bin/bash
#SBATCH --job-name=v2vgot_osm_crafter
#SBATCH --time=24:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/izar/tercier/v2v-got/logs/osm_crafter_%j.out
#SBATCH --error=/scratch/izar/tercier/v2v-got/logs/osm_crafter_%j.err

source /home/tercier/miniconda3/etc/profile.d/conda.sh
conda activate llava
export WANDB_MODE=offline

# Move all the way into the LLaVA folder so relative paths work
cd /scratch/izar/tercier/v2v-got/V2V-GoT/LLaVA
bash scripts/v1_5/train_osm_crafter.sh
