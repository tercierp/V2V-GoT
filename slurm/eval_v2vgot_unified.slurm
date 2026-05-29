#!/bin/bash
#SBATCH --job-name=v2vgot-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# set -euo pipefail  # disabled: conda activate may emit harmless warnings

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo

module purge
module load gcc cuda/11.8

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate v2v

# Resolve repo root from submission directory
BASE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(cd "$BASE_DIR/.." && pwd)"
V2VGOT_ROOT="$REPO_ROOT/V2V-GoT"

cd "$V2VGOT_ROOT"

RUN_NAME="v2vgot_10ep_both_shallow_f2_4330_full"
COMM_MB="0.4068"

QA_DIR="$V2VGOT_ROOT/../data/V2V-GoT-QA/V2V_GoT_JSONS/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm"

python LLaVA/scripts/eval_v2vgot_unified.py \
  --inference_root LLaVA/playground/data/eval \
  --run_name "$RUN_NAME" \
  --run_variant "bev30" \
  --gt_json "$QA_DIR/v2v4real_3d_grounding_qa_dataset_v2vgot.json" \
  --gt_npy_root DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy \
  --comm_mb "$COMM_MB" \
  --out_dir LLaVA/results


echo "=== Eval finished ==="
echo "Time: $(date)"
