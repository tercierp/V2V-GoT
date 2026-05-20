#!/bin/bash
# inference_v2vgot.sh — V2V-GoT Q1..Q9 inference pipeline (full val set).
#
# Usage (from V2V-GoT repo root):
#   bash LLaVA/scripts/v1_5/inference_v2vgot.sh --mode MODE [--dup]
#
# Modes (mutually exclusive, required):
#   central   Centralized baseline. Both CAVs' LiDAR features fused at inference;
#             Q6 prompt includes GT broadcast of the other CAV's planned trajectory.
#   decent    Fully decentralized. Only the asker CAV's LiDAR is loaded
#             (asker_only=True), and the GT injection of the other CAV's
#             pose+trajectory in Q6 is stripped from the prompt.
#   q1msg     Decentralized with V2V messaging via Q1-edge. Same as `decent`
#             but: (a) Q6 GT broadcast is kept (legitimate V2V plan broadcast)
#             AND (b) the sender's Q1 output is appended to Q3's prompt as a
#             cross-CAV perception message.
#
# Options:
#   --dup     Duplicate the asker's LiDAR features into the non-asker slot
#             instead of zeroing it. Only valid with --mode decent or q1msg.
#

MODE=""
DUP=False
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)  MODE="$2"; shift 2 ;;
        --dup)   DUP=True;  shift ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument: $1"; echo "Run with --help for usage."; exit 1 ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "ERROR: --mode is required. Options: central, decent, q1msg"; exit 1
fi
case "$MODE" in
    central|decent|q1msg) ;;
    *) echo "ERROR: invalid --mode '$MODE'. Options: central, decent, q1msg"; exit 1 ;;
esac
if [ "$DUP" == "True" ] && [ "$MODE" == "central" ]; then
    echo "ERROR: --dup is only meaningful with --mode decent or --mode q1msg."; exit 1
fi

case "$MODE" in
    central) LIDAR_MODE=normal; STRIP_OTHER_CAV=False; V2V_MESSAGE_Q1=False ;;
    decent)  LIDAR_MODE=single; STRIP_OTHER_CAV=True;  V2V_MESSAGE_Q1=False ;;
    q1msg)   LIDAR_MODE=single; STRIP_OTHER_CAV=False; V2V_MESSAGE_Q1=True  ;;
esac
[ "$DUP" == "True" ] && LIDAR_MODE=duplicate

EXP_SUFFIX="_${MODE}"
[ "$DUP" == "True" ] && EXP_SUFFIX="${EXP_SUFFIX}_dup"
EXP_SUFFIX="${EXP_SUFFIX}_all"

CKPT_NUM=4330
MAX_ITEMS=-1
MAX_FRAMES_GEN=-1
[ "$STRIP_OTHER_CAV" == "True" ] && Q6_GEN_EXTRA="--strip_other_cav_context" || Q6_GEN_EXTRA=""
[ "$V2V_MESSAGE_Q1"  == "True" ] && Q3_GEN_EXTRA="--v2v_message_q1"          || Q3_GEN_EXTRA=""

echo "============================================================"
echo "mode=$MODE  dup=$DUP  suffix=$EXP_SUFFIX"
echo "  LIDAR_MODE=$LIDAR_MODE  STRIP_OTHER_CAV=$STRIP_OTHER_CAV  V2V_MESSAGE_Q1=$V2V_MESSAGE_Q1"
echo "============================================================"

echo "nq1 inference"
conda activate llava
cd LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh $CKPT_NUM nq1sm3w0d gt full $LIDAR_MODE "$EXP_SUFFIX" $MAX_ITEMS

echo "nq2 inference"
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh $CKPT_NUM nq2sm3w0d gt full $LIDAR_MODE "$EXP_SUFFIX" $MAX_ITEMS

echo "nq3 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt $CKPT_NUM --input_qa_dataset nq2sm3w0d --output_qa_dataset nq3sm3w0dc --graph full --max_frames $MAX_FRAMES_GEN --exp_suffix "$EXP_SUFFIX" $Q3_GEN_EXTRA

echo "nq3 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh $CKPT_NUM nq3sm3w0dc graph full $LIDAR_MODE "$EXP_SUFFIX" $MAX_ITEMS

echo "nq4 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt $CKPT_NUM --input_qa_dataset nq1sm3w0d nq3sm3w0dc --output_qa_dataset nq4sm3w0dc --graph full --max_frames $MAX_FRAMES_GEN --exp_suffix "$EXP_SUFFIX"

echo "nq4 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh $CKPT_NUM nq4sm3w0dc graph full $LIDAR_MODE "$EXP_SUFFIX" $MAX_ITEMS

echo "nq5 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt $CKPT_NUM --input_qa_dataset nq4sm3w0dc --output_qa_dataset nq5sm3w1dc --graph full --max_frames $MAX_FRAMES_GEN --exp_suffix "$EXP_SUFFIX"

echo "nq5 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh $CKPT_NUM nq5sm3w1dc graph full $LIDAR_MODE "$EXP_SUFFIX" $MAX_ITEMS

echo "nq6 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt $CKPT_NUM --input_qa_dataset nq4sm3w0dc --output_qa_dataset nq6sm3w1dc --graph full --max_frames $MAX_FRAMES_GEN --exp_suffix "$EXP_SUFFIX" $Q6_GEN_EXTRA

echo "nq6 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh $CKPT_NUM nq6sm3w1dc graph full $LIDAR_MODE "$EXP_SUFFIX" $MAX_ITEMS

echo "nq7 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt $CKPT_NUM --input_qa_dataset nq5sm3w1dc nq6sm3w1dc --output_qa_dataset nq7sm3w1dc --graph full --max_frames $MAX_FRAMES_GEN --exp_suffix "$EXP_SUFFIX"

echo "nq7 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh $CKPT_NUM nq7sm3w1dc graph full $LIDAR_MODE "$EXP_SUFFIX" $MAX_ITEMS

echo "nq8 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt $CKPT_NUM --input_qa_dataset nq7sm3w1dc --output_qa_dataset nq8sm3w6dc --graph full --max_frames $MAX_FRAMES_GEN --exp_suffix "$EXP_SUFFIX"

echo "nq8 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh $CKPT_NUM nq8sm3w6dc graph full $LIDAR_MODE "$EXP_SUFFIX" $MAX_ITEMS

echo "nq9 generation"
conda deactivate
cd ../DMSTrack/DMSTrack/
source dmstrack_init_env.sh
cd ../V2V4Real/
python opencood/tools/temp_qa_generation.py --model_dir ./official_models/no_fusion_keep_all/ --fusion_method no_fusion_keep_all --model v2vgot_10ep_both_shallow_f2 --ckpt $CKPT_NUM --input_qa_dataset nq8sm3w6dc --output_qa_dataset nq9sm3w6dc --graph full --max_frames $MAX_FRAMES_GEN --exp_suffix "$EXP_SUFFIX"

echo "nq9 inference"
conda deactivate
conda activate llava
cd ../../LLaVA
source scripts/v1_5/inference_task_lora_7b_my_v2v4real_3d_grounding_v2vgot_10ep_both_shallow_f2.sh $CKPT_NUM nq9sm3w6dc graph full $LIDAR_MODE "$EXP_SUFFIX" $MAX_ITEMS

echo "============================================================"
echo "DONE: mode=$MODE  dup=$DUP  suffix=$EXP_SUFFIX"
echo "============================================================"
