#!/bin/bash
# V2V-GoT sequential inference pipeline (satdesc) - all absolute paths
# This script is sourced from the SLURM script with CWD = V2VGOT_ROOT

set -e  # Exit on first error so we don't waste GPU time

V2VGOT_ROOT="$PWD"
LLAVA_DIR="$V2VGOT_ROOT/LLaVA"
DMSTRACK_DIR="$V2VGOT_ROOT/DMSTrack/DMSTrack"
V2V4REAL_DIR="$V2VGOT_ROOT/DMSTrack/V2V4Real"

# Ensure conda is available for env switching
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Helper: run LLaVA inference ---
run_llava_inference() {
    local ckpt=$1 data=$2 data_source=$3 graph=$4
    echo ">>> LLaVA inference: ckpt=$ckpt data=$data source=$data_source graph=$graph"
    conda activate llava
    cd "$LLAVA_DIR"
    source scripts/v1_5/inference_task_lora_satdesc.sh "$ckpt" "$data" "$data_source" "$graph"
    echo "<<< LLaVA inference done: $data"
}

# --- Helper: run QA generation ---
run_qa_generation() {
    local model=$1 ckpt=$2 graph=$3 output_qa=$4
    shift 4
    local input_qa_args=("$@")
    echo ">>> QA generation: output=$output_qa inputs=${input_qa_args[*]}"
    conda activate v2v
    cd "$V2V4REAL_DIR"
    export PYTHONPATH="$V2VGOT_ROOT/DMSTrack/AB3DMOT:$V2VGOT_ROOT/DMSTrack/AB3DMOT/Xinshuo_PyToolbox:$V2VGOT_ROOT/DMSTrack/V2V4Real:$V2VGOT_ROOT/DMSTrack/DMSTrack:$V2VGOT_ROOT/DMSTrack:$PYTHONPATH"
    python opencood/tools/temp_qa_generation.py \
        --model_dir "$V2V_LLM_DATA_ROOT/no_fusion_keep_all/" \
        --fusion_method no_fusion_keep_all \
        --model "$model" \
        --ckpt "$ckpt" \
        --input_qa_dataset "${input_qa_args[@]}" \
        --output_qa_dataset "$output_qa" \
        --graph "$graph"
    echo "<<< QA generation done: $output_qa"
}

# ===== PIPELINE =====

echo "nq1 inference"
run_llava_inference 4000 nq1sm3w0d gt full

echo "nq2 inference"
run_llava_inference 4000 nq2sm3w0d gt full

echo "nq3 generation"
run_qa_generation v2vgot_satdesc 4000 full nq3sm3w0dc nq2sm3w0d

echo "nq3 inference"
run_llava_inference 4000 nq3sm3w0dc graph full

echo "nq4 generation"
run_qa_generation v2vgot_satdesc 4000 full nq4sm3w0dc nq1sm3w0d nq3sm3w0dc

echo "nq4 inference"
run_llava_inference 4000 nq4sm3w0dc graph full

echo "nq5 generation"
run_qa_generation v2vgot_satdesc 4000 full nq5sm3w1dc nq4sm3w0dc

echo "nq5 inference"
run_llava_inference 4000 nq5sm3w1dc graph full

echo "nq6 generation"
run_qa_generation v2vgot_satdesc 4000 full nq6sm3w1dc nq4sm3w0dc

echo "nq6 inference"
run_llava_inference 4000 nq6sm3w1dc graph full

echo "nq7 generation"
run_qa_generation v2vgot_satdesc 4000 full nq7sm3w1dc nq5sm3w1dc nq6sm3w1dc

echo "nq7 inference"
run_llava_inference 4000 nq7sm3w1dc graph full

echo "nq8 generation"
run_qa_generation v2vgot_satdesc 4000 full nq8sm3w6dc nq7sm3w1dc

echo "nq8 inference"
run_llava_inference 4000 nq8sm3w6dc graph full

echo "nq9 generation"
run_qa_generation v2vgot_satdesc 4000 full nq9sm3w6dc nq8sm3w6dc

echo "nq9 inference"
run_llava_inference 4000 nq9sm3w6dc graph full

echo "=== All 9 stages complete ==="
