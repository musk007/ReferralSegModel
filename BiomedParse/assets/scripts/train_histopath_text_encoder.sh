#!/bin/bash
# Fine-tune BiomedParse text encoder on all histopathology datasets.
#
# Only the language (text) encoder is trained; the visual backbone,
# pixel decoder and transformer decoder are fully frozen.
#
# Usage (from BiomedParse root):
#   bash assets/scripts/train_histopath_text_encoder.sh
#
# Prerequisites:
#   1. Run the convert script for every dataset split first:
#        python convert_histopath_to_biomedparse.py \
#            --input  <raw_json>  --dataset_type <type>  --split <train|test> \
#            --output_json ../biomedparse_datasets/<folder>/<split>.json \
#            --images_dir  ../biomedparse_datasets/<folder>/<split>
#   2. The pretrained checkpoint is downloaded automatically from HuggingFace
#      (microsoft/BiomedParse) on first run into ./pretrained/biomedparse_v1.pt

set -e
cd "$(dirname "$0")/../.."   # run from BiomedParse root

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BIOMEDPARSE_DATASETS_DIR="$(realpath ../biomedparse_datasets)"
PRETRAINED_WEIGHTS="hf_hub:microsoft/BiomedParse"
OUTPUT_DIR="./output/histopath_text_encoder"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export PYTHONWARNINGS="ignore"
export DETECTRON2_DATASETS="${BIOMEDPARSE_DATASETS_DIR}"
export DATASET="${BIOMEDPARSE_DATASETS_DIR}"
export DATASET2="${BIOMEDPARSE_DATASETS_DIR}"
export VLDATASET="${BIOMEDPARSE_DATASETS_DIR}"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# export WANDB_KEY=YOUR_WANDB_KEY   # uncomment and set if using W&B

# ---------------------------------------------------------------------------
# GPU configuration
# mpirun -n 1 launches a single process that sees all 8 GPUs via
# CUDA_VISIBLE_DEVICES (matches the original BiomedParse train.sh convention).
# ---------------------------------------------------------------------------
BATCH_SIZE_TOTAL=16      # total samples per step across all GPUs
BATCH_SIZE_PER_GPU=2     # BATCH_SIZE_TOTAL / 8 GPUs

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "=================================================="
echo "BiomedParse – text-encoder fine-tuning"
echo "Datasets  : biomed_HistoPathAll (all histopath)"
echo "Frozen    : backbone, pixel_decoder, predictor"
echo "Trainable : lang_encoder only"
echo "Output    : ${OUTPUT_DIR}"
echo "=================================================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
mpirun -n 1 python entry.py train \
    --conf_files configs/biomed_seg_lang_v1.yaml biomed_seg_lang_v1_histopath.yaml \
    --overrides \
    SAVE_DIR "${OUTPUT_DIR}" \
    FP16 True \
    RANDOM_SEED 2024 \
    BioMed.INPUT.IMAGE_SIZE 1024 \
    MODEL.DECODER.HIDDEN_DIM 512 \
    MODEL.ENCODER.CONVS_DIM 512 \
    MODEL.ENCODER.MASK_DIM 512 \
    TRAIN.BATCH_SIZE_TOTAL ${BATCH_SIZE_TOTAL} \
    TRAIN.BATCH_SIZE_PER_GPU ${BATCH_SIZE_PER_GPU} \
    TEST.BATCH_SIZE_TOTAL ${BATCH_SIZE_TOTAL} \
    SOLVER.MAX_NUM_EPOCHS 20 \
    SOLVER.BASE_LR 0.00005 \
    MODEL.DECODER.GROUNDING.ENABLED True \
    MODEL.DECODER.SPATIAL.ENABLED True \
    MODEL.DECODER.SPATIAL.MAX_ITER 0 \
    LOADER.SAMPLE_PROB prop \
    BioMed.INPUT.RANDOM_ROTATE True \
    FIND_UNUSED_PARAMETERS True \
    ATTENTION_ARCH.SPATIAL_MEMORIES 32 \
    ATTENTION_ARCH.QUERY_NUMBER 3 \
    STROKE_SAMPLER.MAX_CANDIDATE 10 \
    WEIGHT True \
    RESUME_FROM "${PRETRAINED_WEIGHTS}"

echo ""
echo "Training complete. Checkpoints saved to: ${OUTPUT_DIR}"
