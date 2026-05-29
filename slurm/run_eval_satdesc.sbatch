#!/bin/bash
#SBATCH --job-name=eval_satdesc
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_satdesc_%j.out
#SBATCH --error=logs/eval_satdesc_%j.err

source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate v2v

cd $PWD/../_v2vgot_repo_tmp/DMSTrack/DMSTrack
export MY_PROJECT=$PWD/../
export PYTHONPATH=${PYTHONPATH}:${MY_PROJECT}/AB3DMOT
export PYTHONPATH=${PYTHONPATH}:${MY_PROJECT}/AB3DMOT/Xinshuo_PyToolbox
export PYTHONPATH=${PYTHONPATH}:${MY_PROJECT}/V2V4Real
export PYTHONPATH=${PYTHONPATH}:${MY_PROJECT}/DMSTrack
export PYTHONPATH=${PYTHONPATH}:${MY_PROJECT}/

cd ../../LLaVA

MODEL=v2vgot_satdesc
GRAPH=full
CKPT_ID=4000

declare -a DATAS=("nq1sm3w0d" "nq2sm3w0d" "nq3sm3w0dc" "nq4sm3w0dc" "nq5sm3w1dc" "nq6sm3w1dc" "nq7sm3w1dc" "nq8sm3w6dc" "nq9sm3w6dc")
declare -a QA_TYPES=("11" "12" "13" "14" "15" "16" "17" "18" "19")

for i in "${!DATAS[@]}"; do
    DATA=${DATAS[$i]}
    QA_TYPE=${QA_TYPES[$i]}
    
    OUTPUT="results/${MODEL}_${CKPT_ID}_${GRAPH}_${DATA}.txt"
    echo $MODEL > $OUTPUT
    echo $DATA >> $OUTPUT
    echo $GRAPH >> $OUTPUT
    echo $CKPT_ID >> $OUTPUT
    
    python scripts/eval_v2v4real_3d_grounding.py \
        --simplified \
        --multiple-output \
        --max-num-answer-objects 3 \
        --qa-type-id ${QA_TYPE} \
        --num-future-waypoints 0 \
        --npy-save-path ../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy \
        --answers-file ./playground/data/eval/v2v4real_3d_grounding_${MODEL}_${CKPT_ID}_${GRAPH}_${DATA}/answers/val/llava-v1.5-7b/merge.jsonl \
    >> $OUTPUT
done
