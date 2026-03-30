#!/bin/bash
# =============================================================================
# SLURM job script – BiomedParse lang_encoder + predictor fine-tuning (no LoRA)
# Only lang_encoder and predictor are trainable (full weights, no LoRA).
# backbone and pixel_decoder are frozen via SOLVER.FIX_PARAM.
#
# Usage:
#   sbatch runner_nolora.sh
#   TRAIN_DATASET=colon sbatch runner_nolora.sh   # single-dataset mode
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM directives
# ---------------------------------------------------------------------------
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:8        # full node — 8 × 140 GB H200
#SBATCH --cpus-per-task=128             # all 128 cores on the node
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --job-name=bp_nolora_ft
#SBATCH --output=/home/roba/miccai26/logs/finetune_nolora_%j.out
#SBATCH --error=/home/roba/miccai26/logs/finetune_nolora_%j.err

set -e

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BIOMEDPARSE_DIR="${BIOMEDPARSE_DIR:-/home/roba/miccai26/BiomedParse}"
DATASETS_DIR="${DATASETS_DIR:-/home/roba/miccai26/biomedparse_datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${BIOMEDPARSE_DIR}/output/histopath_nolora}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-hf_hub:microsoft/BiomedParse}"

# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------
NUM_GPUS="${SLURM_GPUS_ON_NODE:-8}"
BATCH_SIZE_PER_GPU=12
BATCH_SIZE_TOTAL=$(( NUM_GPUS * BATCH_SIZE_PER_GPU ))

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source /home/roba/miniconda3/bin/activate
conda activate dualprotoseg

export PYTHONWARNINGS="ignore"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DETECTRON2_DATASETS="${DATASETS_DIR}"
export DATASET="${DATASETS_DIR}"
export DATASET2="${DATASETS_DIR}"
export VLDATASET="${DATASETS_DIR}"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_orte_tmpdir_base="/tmp"
export PMIX_MCA_gds="^ds12,ds21"
export OMPI_MCA_btl="^openib"
export OMPI_MCA_pml="ob1"
export WANDB_KEY="${WANDB_KEY:?Set WANDB_KEY in your environment}"

mkdir -p /home/roba/miccai26/logs

# ---------------------------------------------------------------------------
# Dataset override – single-dataset mode (optional)
# ---------------------------------------------------------------------------
CONFIG_OVERRIDES_ARGS=()
if [ -n "${TRAIN_DATASET}" ]; then
  case "${TRAIN_DATASET}" in
    colon)        PREFIX="HistoPathColon" ;;
    lung)         PREFIX="HistoPathLung" ;;
    prostate)     PREFIX="HistoPathProstate" ;;
    breast_bcss)  PREFIX="HistoPathBCSS" ;;
    breast_cells) PREFIX="HistoPathCells" ;;
    *)
      echo "Unknown TRAIN_DATASET=${TRAIN_DATASET}. Use: colon, lung, prostate, breast_bcss, breast_cells"
      exit 1
      ;;
  esac
  TRAIN_DS="biomed_${PREFIX}_train"
  TEST_DS="biomed_${PREFIX}_test"
  CONFIG_OVERRIDES_ARGS=(--config_overrides "{\"DATASETS.TRAIN\": [\"${TRAIN_DS}\"], \"DATASETS.TEST\": [\"${TEST_DS}\"]}")
  OUTPUT_DIR="${OUTPUT_DIR}_${TRAIN_DATASET}"
  echo "Single-dataset mode: train=${TRAIN_DS} test=${TEST_DS} → ${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "BiomedParse – lang_encoder + predictor fine-tuning (no LoRA)"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPUs       : ${NUM_GPUS}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Batch total: ${BATCH_SIZE_TOTAL}  (${BATCH_SIZE_PER_GPU} per GPU)"
echo "Output     : ${OUTPUT_DIR}"
echo "Datasets   : ${DATASETS_DIR}"
echo "Trainable  : lang_encoder + predictor (backbone + pixel_decoder FROZEN)"
echo "LR         : predictor=1e-5 | lang_encoder=1e-5  (backbone/pixel_decoder=0)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Run
# Three config files are stacked in order (each overrides the previous):
#   1. configs/biomed_seg_lang_v1.yaml          – base BiomedParse config
#   2. biomed_seg_lang_v1_histopath_full.yaml   – dataset + training settings
#   3. biomed_seg_lang_v1_histopath_nolora.yaml – disables LoRA, freezes backbone
# ---------------------------------------------------------------------------
cd "${BIOMEDPARSE_DIR}"

mpirun -n ${NUM_GPUS} --oversubscribe --bind-to none python entry.py train \
    --conf_files configs/biomed_seg_lang_v1.yaml \
                 biomed_seg_lang_v1_histopath_full.yaml \
                 biomed_seg_lang_v1_histopath_nolora.yaml \
    "${CONFIG_OVERRIDES_ARGS[@]}" \
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
    SOLVER.MAX_NUM_EPOCHS 50 \
    SOLVER.BASE_LR 0.000001 \
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
    WANDB True \
    RESUME_FROM "${PRETRAINED_WEIGHTS}"

echo ""
echo "Training complete. Checkpoints saved to: ${OUTPUT_DIR}"
