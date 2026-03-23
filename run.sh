#!/bin/bash
# Text-Prompted Segmentation Evaluation Script
# 
# Usage:
#   ./run.sh                     # Run with default model (biomedparse)
#   ./run.sh sam3                # Run with SAM3
#   ./run.sh biomedparse         # Run with BiomedParse v1
#   ./run.sh medisee             # Run with MediSee
#   ./run.sh dualprotoseg        # Run with DualProtoSeg
#   ./run.sh --compare           # Compare all available models
#   ./run.sh --list              # List available models

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/config.sh" ] && . "${SCRIPT_DIR}/config.sh"

# Data paths
IMAGES_DIR="/home/roba/miccai26/biomedparse_datasets/breast_bcss/test"
MASKS_DIR="/home/roba/miccai26/biomedparse_datasets/breast_bcss/test_mask"
PROMPTS_JSON="/home/roba/miccai26/test_data/test_split/instructions/breast_bcss.json"
OUTPUT_DIR="/home/roba/miccai26/results/fineTuned_full/wLoRA"

# Model repository paths
BIOMEDPARSE_PATH="/home/roba/miccai26/BiomedParse"
SAM3_PATH=""
MEDISEE_PATH="/home/roba/miccai26/MediSee"
MEDISEE_MODEL_DIR="/home/roba/miccai26/models/medisee"
DUALPROTOSEG_PATH="/home/roba/miccai26/DualProtoSeg"

# Model checkpoint paths
DUALPROTOSEG_CHECKPOINT="/home/roba/miccai26/DualProtoSeg/runs/checkpoints/2026-01-31-06-14(BaselineRun)/best_cam.pth"
# BiomedParse checkpoint from config.sh (override with BIOMEDPARSE_CHECKPOINT=...)
BIOMEDPARSE_CHECKPOINT="/home/roba/miccai26/BiomedParse/output/histopath_full/biomed_seg_lang_v1.yaml_conf~/run_2/00006160"

# Dataset configuration (for loading class prompts)
# IMPORTANT: Must match the dataset the checkpoint was trained on!
# The provided checkpoint was trained on BCSS, so use dataset="bcss"
# Using a different dataset will cause dimension mismatches
DATASET="colon"  # Options: colon, bcss, liver, prostate
PROMPTS_CONFIG="/home/roba/miccai26/configs/class_prompts.yaml"

# Parse command line arguments
MODEL="${1:-biomedparse}"  # Take from command line or default to biomedparse
DEVICE="cuda"
THRESHOLD="0.5"
METRICS="iou dice precision recall accuracy"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build model-specific arguments
build_model_args() {
    local model="$1"
    local args=""
    
    case "$model" in
        biomedparse)
            args="--biomedparse_path $BIOMEDPARSE_PATH"
            if [ -n "$BIOMEDPARSE_CHECKPOINT" ]; then
                args="$args --checkpoint $BIOMEDPARSE_CHECKPOINT"
            fi
            ;;
        sam3)
            if [ -n "$SAM3_PATH" ]; then
                args="--sam3_path $SAM3_PATH"
            fi
            ;;
        medisee)
            if [ -n "$MEDISEE_PATH" ]; then
                args="--medisee_path $MEDISEE_PATH"
            fi
            if [ -n "$MEDISEE_MODEL_DIR" ]; then
                args="$args --medisee_model_dir $MEDISEE_MODEL_DIR"
            fi
            ;;
        dualprotoseg)
            if [ -n "$DUALPROTOSEG_PATH" ]; then
                args="--dualprotoseg_path $DUALPROTOSEG_PATH"
            fi
            if [ -n "$DUALPROTOSEG_CHECKPOINT" ]; then
                args="$args --checkpoint $DUALPROTOSEG_CHECKPOINT"
            fi
            if [ -n "$DATASET" ]; then
                args="$args --dataset $DATASET"
            fi
            if [ -n "$PROMPTS_CONFIG" ]; then
                args="$args --prompts_config $PROMPTS_CONFIG"
            fi
            ;;
    esac
    
    echo "$args"
}

# Activate conda environment (clear $@ to avoid interference with conda.sh)
activate_conda() {
    source /home/roba/miniconda3/bin/activate
    conda activate dualprotoseg
}
activate_conda

# Prevent MPI initialization issues (for BiomedParse)
export OMPI_MCA_orte_tmpdir_base="/tmp"
export PMIX_MCA_gds="^ds12,ds21"
export OMPI_MCA_btl="^openib"
export OMPI_MCA_pml="ob1"

# Handle special commands
if [ "$MODEL" == "--list" ] || [ "$MODEL" == "-l" ]; then
    python "$SCRIPT_DIR/text_seg_eval_example.py" --list_models
    exit 0
fi

if [ "$MODEL" == "--help" ] || [ "$MODEL" == "-h" ]; then
    echo "Text-Prompted Segmentation Evaluation"
    echo ""
    echo "Usage:"
    echo "  ./run.sh [MODEL]         Evaluate a specific model"
    echo "  ./run.sh --list          List available models"
    echo "  ./run.sh --compare       Compare multiple models"
    echo "  ./run.sh --all           Run all models"
    echo ""
    echo "Available models:"
    echo "  sam3, biomedparse, medisee, dualprotoseg"
    exit 0
fi

if [ "$MODEL" == "--compare" ] || [ "$MODEL" == "-c" ]; then
    COMPARE_MODELS="${2:-sam3 biomedparse medisee dualprotoseg}"
    echo "Comparing models: $COMPARE_MODELS"
    
    python "$SCRIPT_DIR/text_seg_eval_example.py" \
        --compare $COMPARE_MODELS \
        --images_dir "$IMAGES_DIR" \
        --masks_dir "$MASKS_DIR" \
        --prompts_json "$PROMPTS_JSON" \
        --device "$DEVICE" \
        --output "$OUTPUT_DIR/comparison_results.json" \
        --save_predictions "$OUTPUT_DIR/comparison_predictions" \
        $(build_model_args "biomedparse") \
        $(build_model_args "sam3") \
        $(build_model_args "medisee") \
        $(build_model_args "dualprotoseg")
    
    echo ""
    echo "Results saved to: $OUTPUT_DIR/comparison_results.json"
    exit 0
fi

if [ "$MODEL" == "--all" ] || [ "$MODEL" == "-a" ]; then
    echo "Running all 4 models: sam3, biomedparse, medisee, dualprotoseg"
    echo ""
    
    for m in sam3 biomedparse medisee dualprotoseg; do
        echo ">>> Evaluating: $m"
        MODEL_ARGS=$(build_model_args "$m")
        
        python "$SCRIPT_DIR/text_seg_eval_example.py" \
            --model "$m" \
            --images_dir "$IMAGES_DIR" \
            --masks_dir "$MASKS_DIR" \
            --prompts_json "$PROMPTS_JSON" \
            --device "$DEVICE" \
            --threshold "$THRESHOLD" \
            --metrics $METRICS \
            $MODEL_ARGS \
            --output "$OUTPUT_DIR/${m}_eval_results.json" \
            --save_predictions "$OUTPUT_DIR/${m}_predictions" \
            2>&1 || echo "Warning: $m failed"
        echo ""
    done
    
    echo "All evaluations complete!"
    exit 0
fi

# Run single model evaluation
echo "========================================"
echo "Evaluating model: $MODEL"
echo "========================================"

MODEL_ARGS=$(build_model_args "$MODEL")

echo "Images:    $IMAGES_DIR"
echo "Masks:     $MASKS_DIR"
echo "Prompts:   $PROMPTS_JSON"
echo "Device:    $DEVICE"
echo ""

python "$SCRIPT_DIR/text_seg_eval_example.py" \
    --model "$MODEL" \
    --images_dir "$IMAGES_DIR" \
    --masks_dir "$MASKS_DIR" \
    --prompts_json "$PROMPTS_JSON" \
    --device "$DEVICE" \
    --threshold "$THRESHOLD" \
    --metrics $METRICS \
    $MODEL_ARGS \
    --output "$OUTPUT_DIR/${MODEL}_eval_results.json" \
    --save_predictions "$OUTPUT_DIR/${MODEL}_predictions"

echo ""
echo "Results saved to: $OUTPUT_DIR/${MODEL}_eval_results.json"
echo "Predictions: $OUTPUT_DIR/${MODEL}_predictions/"
