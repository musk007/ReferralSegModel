#!/bin/bash

# Evaluation script for DualProtoSeg
# Usage: bash eval.sh [test|valid] [--generate-cams]

CHECKPOINT="/home/roba/miccai26/DualProtoSeg/runs/checkpoints/2026-02-05-08-15/best_cam.pth"
CONFIG="config.yaml"
GPU=0

# Default to test split
SPLIT=${1:-test}

# Check if --generate-cams flag is provided
GENERATE_CAMS=""
if [[ "$@" == *"--generate-cams"* ]]; then
    GENERATE_CAMS="--generate_cams"
fi

echo "=========================================="
echo "DualProtoSeg Model Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Split:      $SPLIT"
echo "GPU:        $GPU"
echo ""

python evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --split "$SPLIT" \
    --gpu $GPU \
    $GENERATE_CAMS

echo ""
echo "Evaluation complete!"
