#!/bin/bash
#SBATCH --job-name=v2vgotd-cav1-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --output=/scratch/izar/%u/v2v-got/logs/%x-%j.out
#SBATCH --error=/scratch/izar/%u/v2v-got/logs/%x-%j.err
#
# Run LLM inference on the cav1-perspective QA files produced by gen_cav1_qa.py.
#
# What this does:
#   nq8 inference: cav1-frame question → speed/steering prediction for cav1
#   nq9 inference: cav1-frame question → trajectory prediction for cav1
#
# Prerequisite (CPU, run once before submitting this job):
#   python scripts/gen_cav1_qa.py \
#       --output_dir /scratch/izar/$USER/v2v-got/outputs/cav1_perspective
#
# Submit:
#   sbatch slurm/infer_cav1.sh
#
# Outputs land in:
#   /scratch/izar/$USER/v2v-got/outputs/cav1_perspective/nq8_merge.jsonl
#   /scratch/izar/$USER/v2v-got/outputs/cav1_perspective/nq9_merge.jsonl
#
# After the job finishes, pass these to phase4_eval.py via --nq8_llm_cav1ref
# and --nq9_llm_cav1ref to get the updated comparison table.

set -euo pipefail

echo "=== Job info ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo

SCRATCH=/scratch/izar/$USER/v2v-got
REPO=$SCRATCH/V2V-GoT
# Real v2vgot_10ep baseline weights — re-downloaded from
# huggingface.co/datasets/eddyhkchiu/V2V-GoT-QA  (model_ckpt.zip, ~37 GB).
CKPT_NUM=4330
MODEL=v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2
CKPT_BASE=$SCRATCH/V2V-GoT/LLaVA/checkpoints
CAV1_DIR=$SCRATCH/outputs/cav1_perspective

mkdir -p $SCRATCH/logs $CAV1_DIR

module purge
module load gcc cuda/11.8

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llava

export HF_HOME=$SCRATCH/hf_cache
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO/LLaVA:${PYTHONPATH:-}

nvidia-smi
echo

MODEL_PATH=$CKPT_BASE/llava-v1.5-7b-task-lora/\
llava-v1.5-7b-task-lora_${MODEL}/checkpoint-${CKPT_NUM}

# model_vqa_loader.py resolves feature maps via a relative path
# '../DMSTrack/V2V4Real/official_models/' — must run from inside LLaVA/
cd $REPO/LLaVA
export PYTHONPATH=$REPO/LLaVA:${PYTHONPATH:-}

_infer() {
    local QFILE=$1 AFILE=$2 TAG=$3

    # Resume: if a partial output already exists, filter out already-done IDs
    # and run only the remaining questions, then append and sort into the final file.
    local QFILE_RUN=$QFILE
    if [[ -s "$AFILE" ]]; then
        echo "=== ${TAG}: partial output found ($(wc -l < $AFILE) lines) — resuming ==="
        local DONE_IDS=$(python3 -c "
import json, sys
done = set()
for line in open('$AFILE'):
    try: done.add(json.loads(line)['id'])
    except: pass
print(len(done), 'already done:', sorted(done)[:5], '...')
" 2>&1)
        echo "  $DONE_IDS"

        local QFILE_REMAIN=${AFILE%.jsonl}_remain.json
        local AFILE_NEW=${AFILE%.jsonl}_new.jsonl
        python3 -c "
import json
done = set()
for line in open('$AFILE'):
    try: done.add(json.loads(line)['id'])
    except: pass
qs = json.load(open('$QFILE'))
remain = [q for q in qs if q['id'] not in done]
json.dump(remain, open('$QFILE_REMAIN', 'w'))
print(f'Remaining: {len(remain)} / {len(qs)}')
"
        QFILE_RUN=$QFILE_REMAIN
        AFILE_OUT=$AFILE_NEW
    else
        echo "=== ${TAG} inference (cav1 reference frame) ==="
        AFILE_OUT=$AFILE
    fi

    python -m llava.eval.model_vqa_loader \
        --model-path  "$MODEL_PATH" \
        --model-base  liuhaotian/llava-v1.5-7b \
        --mm_scene_projector_input_size 3072 \
        --scene_level_only  False \
        --object_level_only False \
        --scene_feature_mode  shallow \
        --object_feature_mode shallow \
        --num_input_frames 2 \
        --ego_only False \
        --feature_source no_fusion_keep_all \
        --num_latency_frames 0 \
        --positional_error_10_std 0 \
        --question-file "$QFILE_RUN" \
        --answers-file  "$AFILE_OUT" \
        --num-chunks 1 \
        --chunk-idx  0 \
        --temperature 0 \
        --conv-mode vicuna_v1

    # Merge new results into the main file and sort by id
    if [[ "$AFILE_OUT" != "$AFILE" && -s "$AFILE_OUT" ]]; then
        cat "$AFILE_OUT" >> "$AFILE"
        python3 -c "
import json
lines = []
for line in open('$AFILE'):
    try: lines.append((json.loads(line)['id'], line))
    except: pass
lines.sort()
open('$AFILE', 'w').writelines(l for _, l in lines)
print(f'Merged and sorted: {len(lines)} records')
"
        rm -f "$AFILE_OUT" "$QFILE_REMAIN"
    fi

    echo "${TAG} done — $(wc -l < $AFILE) records → ${AFILE}"
    echo
}

# ── nq8 ───────────────────────────────────────────────────────────────────────
_infer "$CAV1_DIR/nq8sm3w6dc_cav1ref.json" \
       "$CAV1_DIR/nq8_merge.jsonl" \
       "nq8"

# ── nq9 ───────────────────────────────────────────────────────────────────────
_infer "$CAV1_DIR/nq9sm3w6dc_cav1ref.json" \
       "$CAV1_DIR/nq9_merge.jsonl" \
       "nq9"
echo

echo "=== All done: $(date) ==="
echo
echo "Next step:"
echo "  python scripts/phase4_eval.py \\"
echo "      --nq8_llm_cav1ref $CAV1_DIR/nq8_merge.jsonl \\"
echo "      --nq9_llm_cav1ref $CAV1_DIR/nq9_merge.jsonl \\"
echo "      --output_dir outputs/phase4_cav1ref"
