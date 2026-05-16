#!/bin/bash
# Full 9-stage V2V-GoT inference pipeline with configurable OSM integration.
#
# Usage (from V2V-GoT/ dir, after activating llava env):
#   source LLaVA/scripts/v1_5/inference_v2vgot_osm.sh <osm_ckpt> <osm_from>
#
# Arguments:
#   osm_ckpt : checkpoint step of the OSM CRAFTER model (e.g. 1000, 2000)
#   osm_from : first GoT stage to use the OSM model (1-9).
#              Stages < osm_from use the baseline (checkpoint-4330).
#              Examples:
#                9   → OSM only at planning (Q9)
#                8   → OSM at decision + planning (Q8-9)
#                7   → OSM at trajectory-merge + ... (Q7-9)
#                5   → OSM from trajectory prediction onwards (Q5-9)
#                1   → OSM for all stages (Q1-9)
#
# All inference answers are written under a unified pipeline tag so generation
# steps can always find the right merge.jsonl regardless of which model ran.

OSM_CKPT=${1:-1000}
OSM_FROM=${2:-9}

V2VGOT_DIR="$(pwd)"

BASELINE_MODEL_DIR="${V2VGOT_DIR}/LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2/checkpoint-4330"
OSM_MODEL_DIR="${V2VGOT_DIR}/LLaVA/checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_crafter_osm_crafter_train01/checkpoint-${OSM_CKPT}"
SAT_IMAGES_ROOT="/scratch/izar/tercier/v2v-got/sat_images"

# Unified tag used for ALL eval paths so generation steps always find their inputs.
# Mirrors the pattern: v2v4real_3d_grounding_{PIPE_MODEL}_{PIPE_CKPT}_full_{data}
PIPE_MODEL="osm_from_q${OSM_FROM}"
PIPE_CKPT="${OSM_CKPT}"

SPLIT="val"
BASE_MODEL_NAME="liuhaotian/llava-v1.5-7b"

echo "=== V2V-GoT OSM pipeline ==="
echo "OSM_CKPT:   $OSM_CKPT"
echo "OSM_FROM:   Q${OSM_FROM}"
echo "PIPE_TAG:   v2v4real_3d_grounding_${PIPE_MODEL}_${PIPE_CKPT}_full_*"
echo

# ── Helper: run one LLM inference stage ──────────────────────────────────────
# Args: data_tag  source  stage_num
#   data_tag  : e.g. nq1sm3w0d
#   source    : 'gt'    → read from official npy QA json
#               'graph' → read from previous stage's generated json
#   stage_num : 1-9, decides baseline vs OSM model
run_llm_stage() {
    local DATA=$1
    local SOURCE=$2
    local STAGE=$3

    echo "--- Q${STAGE} inference (${DATA}, source=${SOURCE}, model=$([ $STAGE -ge $OSM_FROM ] && echo OSM || echo baseline)) ---"

    # Pick model path and optional OSM args
    local MODEL_PATH EXTRA_ARGS
    if [ "$STAGE" -ge "$OSM_FROM" ]; then
        MODEL_PATH="$OSM_MODEL_DIR"
        EXTRA_ARGS="--use_osm True --osm_image_folder $SAT_IMAGES_ROOT"
    else
        MODEL_PATH="$BASELINE_MODEL_DIR"
        EXTRA_ARGS=""
    fi

    # Question file (absolute — script cds into LLaVA/ before launching Python)
    local QUESTION_FILE
    if [ "$SOURCE" == "gt" ]; then
        QUESTION_FILE="${V2VGOT_DIR}/DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_${DATA}.json"
    else
        QUESTION_FILE="${V2VGOT_DIR}/LLaVA/playground/data/eval/v2v4real_3d_grounding_${PIPE_MODEL}_${PIPE_CKPT}_full_${DATA}/answers/${SPLIT}/llava-v1.5-7b/${DATA}.json"
    fi

    local ANSWER_DIR="${V2VGOT_DIR}/LLaVA/playground/data/eval/v2v4real_3d_grounding_${PIPE_MODEL}_${PIPE_CKPT}_full_${DATA}/answers/${SPLIT}/llava-v1.5-7b"
    mkdir -p "$ANSWER_DIR"

    # Skip if already completed
    if [ -s "$ANSWER_DIR/merge.jsonl" ]; then
        echo "  [skip] Q${STAGE} already done ($(wc -l < "$ANSWER_DIR/merge.jsonl") lines)"
        return 0
    fi

    cd LLaVA
    conda activate llava
    export PYTHONPATH=$PWD:$PYTHONPATH

    CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
        --model-base "$BASE_MODEL_NAME" \
        --mm_scene_projector_input_size 3072 \
        --scene_level_only False \
        --object_level_only False \
        --scene_feature_mode shallow \
        --object_feature_mode shallow \
        --num_input_frames 2 \
        --ego_only False \
        --feature_source no_fusion_keep_all \
        --num_latency_frames 0 \
        --positional_error_10_std 0 \
        --question-file "$QUESTION_FILE" \
        --answers-file "$ANSWER_DIR/1_0.jsonl" \
        --num-chunks 1 \
        --chunk-idx 0 \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        $EXTRA_ARGS

    # Merge (single chunk, just copy)
    cp "$ANSWER_DIR/1_0.jsonl" "$ANSWER_DIR/merge.jsonl"
    cd ..
}

# ── Helper: run one DMSTrack generation step ─────────────────────────────────
# Args forwarded directly to temp_qa_generation.py (--input_qa_dataset, --output_qa_dataset)
# Always uses PIPE_MODEL / PIPE_CKPT so paths match the inference outputs above.
run_gen_stage() {
    # Derive output tag from --output_qa_dataset argument to check for existing output
    local OUT_DATA=""
    for arg in "$@"; do
        if [ -n "$PREV_ARG" ] && [ "$PREV_ARG" == "--output_qa_dataset" ]; then OUT_DATA="$arg"; fi
        PREV_ARG="$arg"
    done
    local GEN_OUT="${V2VGOT_DIR}/LLaVA/playground/data/eval/v2v4real_3d_grounding_${PIPE_MODEL}_${PIPE_CKPT}_full_${OUT_DATA}/answers/${SPLIT}/llava-v1.5-7b/${OUT_DATA}.json"
    if [ -n "$OUT_DATA" ] && [ -s "$GEN_OUT" ]; then
        echo "  [skip] gen ${OUT_DATA} already done"
        return 0
    fi

    local SAVED_PYTHONPATH="$PYTHONPATH"
    cd DMSTrack/DMSTrack/
    source dmstrack_init_env.sh
    cd ../V2V4Real/
    # Use dmstrack Python (has full opencood install) with local source override
    PYTHONPATH="$PWD" \
    /home/tercier/miniconda3/envs/dmstrack/bin/python opencood/tools/temp_qa_generation.py \
        --model_dir ./official_models/no_fusion_keep_all/ \
        --fusion_method no_fusion_keep_all \
        --model "$PIPE_MODEL" \
        --ckpt "$PIPE_CKPT" \
        --graph full \
        "$@"
    cd ../../
    export PYTHONPATH="$SAVED_PYTHONPATH"
}

# ── 9-stage GoT pipeline ─────────────────────────────────────────────────────

echo "nq1 inference"
run_llm_stage nq1sm3w0d gt 1

echo "nq2 inference"
run_llm_stage nq2sm3w0d gt 2

echo "nq3 generation"
run_gen_stage \
    --input_qa_dataset nq2sm3w0d \
    --output_qa_dataset nq3sm3w0dc

echo "nq3 inference"
run_llm_stage nq3sm3w0dc graph 3

echo "nq4 generation"
run_gen_stage \
    --input_qa_dataset nq1sm3w0d nq3sm3w0dc \
    --output_qa_dataset nq4sm3w0dc

echo "nq4 inference"
run_llm_stage nq4sm3w0dc graph 4

echo "nq5 generation"
run_gen_stage \
    --input_qa_dataset nq4sm3w0dc \
    --output_qa_dataset nq5sm3w1dc

echo "nq5 inference"
run_llm_stage nq5sm3w1dc graph 5

echo "nq6 generation"
run_gen_stage \
    --input_qa_dataset nq4sm3w0dc \
    --output_qa_dataset nq6sm3w1dc

echo "nq6 inference"
run_llm_stage nq6sm3w1dc graph 6

echo "nq7 generation"
run_gen_stage \
    --input_qa_dataset nq5sm3w1dc nq6sm3w1dc \
    --output_qa_dataset nq7sm3w1dc

echo "nq7 inference"
run_llm_stage nq7sm3w1dc graph 7

echo "nq8 generation"
run_gen_stage \
    --input_qa_dataset nq7sm3w1dc \
    --output_qa_dataset nq8sm3w6dc

echo "nq8 inference"
run_llm_stage nq8sm3w6dc graph 8

echo "nq9 generation"
run_gen_stage \
    --input_qa_dataset nq8sm3w6dc \
    --output_qa_dataset nq9sm3w6dc

echo "nq9 inference"
run_llm_stage nq9sm3w6dc graph 9

echo
echo "=== Pipeline done ==="
echo "Results in: LLaVA/playground/data/eval/v2v4real_3d_grounding_${PIPE_MODEL}_${PIPE_CKPT}_full_*"
