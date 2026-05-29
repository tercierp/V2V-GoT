#!/bin/bash
#SBATCH --job-name=v2vgot-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=18:00:00
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

# Resolve repo root from submission directory (SLURM runs from a spool dir)
BASE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(cd "$BASE_DIR/.." && pwd)"
V2VGOT_ROOT="$REPO_ROOT/V2V-GoT"

# PYTHONPATH for dmstrack's temp_qa_generation.py calls
export PYTHONPATH=${PYTHONPATH:-}:$V2VGOT_ROOT/DMSTrack/AB3DMOT
export PYTHONPATH=${PYTHONPATH}:$V2VGOT_ROOT/DMSTrack/AB3DMOT/Xinshuo_PyToolbox
export PYTHONPATH=${PYTHONPATH}:$V2VGOT_ROOT/DMSTrack/V2V4Real
export PYTHONPATH=${PYTHONPATH}:$V2VGOT_ROOT/DMSTrack

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
cd "$V2VGOT_ROOT"
mkdir -p /scratch/izar/$USER/v2v-got/logs

# Swap in satellite-description QA files if available (restored on exit)
QA_DIR="/scratch/izar/$USER/v2v-got/data/V2V-GoT-QA/V2V_GoT_JSONS/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm"
QA_DATASETS=(
	nq1sm3w0d
	nq2sm3w0d
	nq3sm3w0dc
	nq4sm3w0dc
	nq5sm3w1dc
	nq6sm3w1dc
	nq7sm3w1dc
	nq8sm3w6dc
	nq9sm3w6dc
)

restore_satdesc() {
	for base in "${SATDESC_BASE_FILES[@]}"; do
		if [ -f "${base}.orig_satdesc" ]; then
			rm -f "$base"
			mv "${base}.orig_satdesc" "$base"
		fi
	done
}

SATDESC_BASE_FILES=()
for dataset in "${QA_DATASETS[@]}"; do
	base_file="$QA_DIR/v2v4real_3d_grounding_qa_dataset_${dataset}.json"
	satdesc_file="$QA_DIR/v2v4real_3d_grounding_qa_dataset_${dataset}_satdesc.json"
	if [ -f "$satdesc_file" ] && [ -f "$base_file" ]; then
		if [ ! -f "${base_file}.orig_satdesc" ]; then
			mv "$base_file" "${base_file}.orig_satdesc"
		fi
		ln -sf "$satdesc_file" "$base_file"
		SATDESC_BASE_FILES+=("$base_file")
	fi
done
trap restore_satdesc EXIT

echo "=== Satellite description swap status ==="
for dataset in "${QA_DATASETS[@]}"; do
    base_file="$QA_DIR/v2v4real_3d_grounding_qa_dataset_${dataset}.json"
    if [ -L "$base_file" ]; then
        echo "  ${dataset}: SWAPPED → $(readlink "$base_file")"
    else
        echo "  ${dataset}: NOT SWAPPED"
    fi
done
echo "========================================="

echo "=== Starting full V2V-GoT inference pipeline ==="
echo "Time: $(date)"
echo

source LLaVA/scripts/v1_5/inference_v2vgot.sh

echo
echo "=== Pipeline finished ==="
echo "Time: $(date)"
