#!/bin/bash
# =============================================================================
# SLURM job script – BiomedParse FULL model fine-tuning
# All modules trainable: backbone, pixel decoder, predictor, lang encoder.
#
# Node specs: 8 × H200 (140 GB VRAM each), 128 CPUs, 1.5 TB RAM.
# Defaults to a full dedicated node (8 GPUs).  worker-0 and worker-2 are
# both idle, so both jobs can run at the same time on separate nodes:
#
#   sbatch runner.sh        # text encoder → e.g. worker-0 (8 GPUs)
#   sbatch runner_full.sh   # full model   → e.g. worker-2 (8 GPUs)
#
# To share one node instead (4 GPUs each):
#   sbatch --gres=gpu:nvidia_h200:4 runner.sh
#   sbatch --gres=gpu:nvidia_h200:4 runner_full.sh
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM directives
# ---------------------------------------------------------------------------
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:8        # full node — 8 × 140 GB H200
#SBATCH --cpus-per-task=128             # all 128 cores on the node
#SBATCH --mem=500G                      # generous but far below 1.5 TB node RAM
#SBATCH --time=48:00:00
#SBATCH --job-name=bp_full_ft
#SBATCH --output=/home/roba/miccai26/logs/finetune_full_%j.out
#SBATCH --error=/home/roba/miccai26/logs/finetune_full_%j.err

set -e

# ---------------------------------------------------------------------------
# Paths  – override any of these via environment variables, e.g.:
#   OUTPUT_DIR=./output/run_full_B sbatch runner_full.sh
#
# Single-dataset fine-tuning (optional):
#   TRAIN_DATASET=colon sbatch runner_full.sh       # colon, lung, prostate, breast_bcss, breast_cells
#   TRAIN_DATASET=lung sbatch runner_full.sh
#   etc.
# If TRAIN_DATASET is unset, uses biomed_CombinedFull (BiomedParse official + your histopath).
# ---------------------------------------------------------------------------
BIOMEDPARSE_DIR="${BIOMEDPARSE_DIR:-/home/roba/miccai26/BiomedParse}"
DATASETS_DIR="${DATASETS_DIR:-/home/roba/miccai26/biomedparse_datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${BIOMEDPARSE_DIR}/output/histopath_full}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-hf_hub:microsoft/BiomedParse}"

# ---------------------------------------------------------------------------
# GPU setup – SLURM sets CUDA_VISIBLE_DEVICES and SLURM_GPUS_ON_NODE for us
# ---------------------------------------------------------------------------
NUM_GPUS="${SLURM_GPUS_ON_NODE:-8}"

# Full model fine-tuning stores backward activations through ALL layers,
# including the deep FocalNet backbone at 1024×1024 resolution.
# Per-image overhead is ~33 GB; batch=2 per GPU → ~7 GB fixed + 2×33 ≈ 73 GB/GPU.
# Empirically: batch=6 OOM'd (138 GB), batch=4 OOM'd (138.7 GB) on a single GPU.
# With DDP (mpirun -n 8), each GPU sees only BATCH_SIZE_PER_GPU images;
# the remaining 7 GPUs are no longer idle.
BATCH_SIZE_PER_GPU=12
BATCH_SIZE_TOTAL=$(( NUM_GPUS * BATCH_SIZE_PER_GPU ))   # effective global batch

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source /home/roba/miniconda3/bin/activate
conda activate dualprotoseg

export PYTHONWARNINGS="ignore"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # reduces fragmentation
export HF_HUB_OFFLINE=1          # worker nodes have no internet — use local cache
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
# Dataset override – when TRAIN_DATASET is set, train on that single dataset only
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
echo "BiomedParse – FULL model fine-tuning"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPUs       : ${NUM_GPUS}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Batch total: ${BATCH_SIZE_TOTAL}  (${BATCH_SIZE_PER_GPU} per GPU)"
echo "Output     : ${OUTPUT_DIR}"
echo "Datasets   : ${DATASETS_DIR}"
echo "Trainable  : ALL (backbone + pixel_decoder + predictor + lang_encoder)"
echo "LR         : backbone=1e-6 | pixel_decoder=2e-6 | predictor=5e-6 | lang_encoder=1e-5"
echo "============================================================"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "${BIOMEDPARSE_DIR}"

mpirun -n ${NUM_GPUS} --oversubscribe --bind-to none python entry.py train \
    --conf_files configs/biomed_seg_lang_v1.yaml biomed_seg_lang_v1_histopath_full.yaml \
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
    SOLVER.MAX_NUM_EPOCHS 20 \
    SOLVER.BASE_LR 0.00000001 \
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
