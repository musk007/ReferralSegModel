#!/bin/bash
# =============================================================================
# SLURM job script – staged partial fine-tuning for BiomedParse
#
# Stages:
#   1) Warm-up (limited trainable subset; image encoder frozen)
#   2) Selective unfreezing (top text blocks + decoder cross-attn)
#   3) Optional full decoder fine-tuning (image encoder still frozen)
#
# Notes:
# - This script is standalone and does not modify any existing runner/config.
# - All augmentation-related options are explicitly disabled in every stage.
#
# Usage:
#   sbatch runner_staged_partial_ft.sh
#   TRAIN_DATASET=colon sbatch runner_staged_partial_ft.sh
#   RUN_STAGE3=1 sbatch runner_staged_partial_ft.sh
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM directives
# ---------------------------------------------------------------------------
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --job-name=bp_staged_ft
#SBATCH --output=/home/roba/miccai26/logs/finetune_staged_%j.out
#SBATCH --error=/home/roba/miccai26/logs/finetune_staged_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BIOMEDPARSE_DIR="${BIOMEDPARSE_DIR:-/home/roba/miccai26/BiomedParse}"
DATASETS_DIR="${DATASETS_DIR:-/adialab/usr/roba/biomedparse_datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/adialab/usr/roba/train_output/histopath_staged_partial}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-hf_hub:microsoft/BiomedParse}"
BASE_CONF="${BASE_CONF:-configs/biomed_seg_lang_v1.yaml}"
HISTO_CONF="${HISTO_CONF:-biomed_seg_lang_v1_histopath_full.yaml}"

# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
STAGE1_BS_PER_GPU="${STAGE1_BS_PER_GPU:-8}"
STAGE2_BS_PER_GPU="${STAGE2_BS_PER_GPU:-8}"
STAGE3_BS_PER_GPU="${STAGE3_BS_PER_GPU:-4}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
GRAD_CLIP="${GRAD_CLIP:-False}"

STAGE1_BS_TOTAL=$(( NUM_GPUS * STAGE1_BS_PER_GPU ))
STAGE2_BS_TOTAL=$(( NUM_GPUS * STAGE2_BS_PER_GPU ))
STAGE3_BS_TOTAL=$(( NUM_GPUS * STAGE3_BS_PER_GPU ))

# ---------------------------------------------------------------------------
# Stage hyperparameters (from recommendation)
# ---------------------------------------------------------------------------
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-20}"

STAGE1_LR="${STAGE1_LR:-0.0001}"
STAGE2_LR="${STAGE2_LR:-0.00005}"
STAGE3_LR="${STAGE3_LR:-0.00001}"

STAGE1_WARMUP_ITERS="${STAGE1_WARMUP_ITERS:-100}"
STAGE2_WARMUP_ITERS="${STAGE2_WARMUP_ITERS:-50}"
STAGE3_WARMUP_ITERS="${STAGE3_WARMUP_ITERS:-50}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
RUN_STAGE3="${RUN_STAGE3:-0}"   # optional stage 3 (0=off, 1=on)

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
export WANDB_KEY="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"

mkdir -p /home/roba/miccai26/logs
mkdir -p "${OUTPUT_ROOT}"

# ---------------------------------------------------------------------------
# Optional single-dataset mode
# ---------------------------------------------------------------------------
TRAIN_DS=""
EVAL_DS=""
TEST_DS=""
if [ -n "${TRAIN_DATASET:-}" ]; then
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
  EVAL_DS="biomed_${PREFIX}_eval"
  TEST_DS="biomed_${PREFIX}_test"
  OUTPUT_ROOT="${OUTPUT_ROOT}_${TRAIN_DATASET}"
  mkdir -p "${OUTPUT_ROOT}"
  echo "Single-dataset mode: train=${TRAIN_DS} eval=${EVAL_DS} test=${TEST_DS} -> ${OUTPUT_ROOT}"
fi

# Merge stage-specific JSON (flat dotted keys) with optional dataset selection.
# NOTE: SOLVER.FIX_PARAM.* cannot be passed via --overrides when FIX_PARAM is {}
# in the YAML — utilities/arguments.py looks up existing leaf keys for typing and
# raises KeyError. JSON config_overrides runs first and creates those keys.
merge_config_overrides_json() {
  local stage_solver_json="$1"
  STAGE_SOLVER_JSON="${stage_solver_json}" TRAIN_DATASET_FLAG="${TRAIN_DATASET:-}" \
  TRAIN_DS_VAL="${TRAIN_DS:-}" EVAL_DS_VAL="${EVAL_DS:-}" TEST_DS_VAL="${TEST_DS:-}" \
  python3 - <<'PY'
import json, os
stage = json.loads(os.environ["STAGE_SOLVER_JSON"])
if os.environ.get("TRAIN_DATASET_FLAG"):
    stage["DATASETS.TRAIN"] = [os.environ["TRAIN_DS_VAL"]]
    stage["DATASETS.EVAL"] = [os.environ["EVAL_DS_VAL"]]
    stage["DATASETS.TEST"] = [os.environ["TEST_DS_VAL"]]
print(json.dumps(stage))
PY
}

cd "${BIOMEDPARSE_DIR}"

echo "============================================================"
echo "BiomedParse staged partial fine-tuning"
echo "Job ID          : ${SLURM_JOB_ID:-local}"
echo "Node            : $(hostname)"
echo "GPUs            : ${NUM_GPUS}"
echo "Output root     : ${OUTPUT_ROOT}"
echo "Image encoder   : frozen in all stages"
echo "Augmentations   : disabled in all stages"
echo "Run Stage 3     : ${RUN_STAGE3}"
echo "============================================================"

# Shared no-augmentation overrides
NO_AUG_OVERRIDES=(
  BioMed.INPUT.AUGMENT False
  BioMed.INPUT.CROP.ENABLED False
  BioMed.INPUT.RANDOM_FLIP none
  BioMed.INPUT.RANDOM_ROTATE False
  BioMed.INPUT.COLOR_AUG_SSD False
)

run_stage() {
  local stage_name="$1"
  local stage_dir="$2"
  local resume_from="$3"
  local bs_per_gpu="$4"
  local bs_total="$5"
  local epochs="$6"
  local lr="$7"
  local warmup_iters="$8"
  local config_overrides_json="$9"

  mkdir -p "${stage_dir}"

  echo ""
  echo "------------------------------------------------------------"
  echo "Running ${stage_name}"
  echo "Save dir       : ${stage_dir}"
  echo "Resume from    : ${resume_from}"
  echo "Batch per GPU  : ${bs_per_gpu}"
  echo "Batch total    : ${bs_total}"
  echo "Epochs         : ${epochs}"
  echo "LR             : ${lr}"
  echo "Warmup iters   : ${warmup_iters}"
  echo "------------------------------------------------------------"

  mpirun -n ${NUM_GPUS} --oversubscribe --bind-to none python entry.py train \
    --conf_files "${BASE_CONF}" "${HISTO_CONF}" \
    --config_overrides "${config_overrides_json}" \
    --overrides \
    SAVE_DIR "${stage_dir}" \
    FP16 True \
    RANDOM_SEED 2024 \
    BioMed.INPUT.IMAGE_SIZE 1024 \
    MODEL.DECODER.HIDDEN_DIM 512 \
    MODEL.ENCODER.CONVS_DIM 512 \
    MODEL.ENCODER.MASK_DIM 512 \
    TRAIN.BATCH_SIZE_TOTAL ${bs_total} \
    TRAIN.BATCH_SIZE_PER_GPU ${bs_per_gpu} \
    TEST.BATCH_SIZE_TOTAL ${bs_total} \
    SOLVER.MAX_NUM_EPOCHS ${epochs} \
    SOLVER.BASE_LR ${lr} \
    SOLVER.WARMUP_ITERS ${warmup_iters} \
    SOLVER.WARMUP_FACTOR 1 \
    SOLVER.OPTIMIZER ADAMW \
    SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
    GRADIENT_ACCUMULATE_STEP ${GRAD_ACC_STEPS} \
    GRAD_CLIP ${GRAD_CLIP} \
    MODEL.DECODER.GROUNDING.ENABLED True \
    MODEL.DECODER.SPATIAL.ENABLED True \
    MODEL.DECODER.SPATIAL.MAX_ITER 0 \
    LOADER.SAMPLE_PROB prop \
    FIND_UNUSED_PARAMETERS True \
    ATTENTION_ARCH.SPATIAL_MEMORIES 32 \
    ATTENTION_ARCH.QUERY_NUMBER 3 \
    STROKE_SAMPLER.MAX_CANDIDATE 10 \
    WEIGHT True \
    WANDB True \
    RESUME_FROM "${resume_from}" \
    "${NO_AUG_OVERRIDES[@]}"
}

# Stage directories/checkpoints
STAGE1_DIR="${OUTPUT_ROOT}/stage1_warmup"
STAGE2_DIR="${OUTPUT_ROOT}/stage2_selective"
STAGE3_DIR="${OUTPUT_ROOT}/stage3_decoder_full"

STAGE1_BEST="${STAGE1_DIR}/run_1/best_model"
STAGE2_BEST="${STAGE2_DIR}/run_1/best_model"

# Flat JSON for SOLVER.FIX_PARAM / IGNORE_FIX / LR_MULTIPLIER (must use --config_overrides; see header comment above).
STAGE1_SOLVER_JSON=$(cat <<'EOF'
{
  "SOLVER.FIX_PARAM.backbone": true,
  "SOLVER.FIX_PARAM.pixel_decoder": true,
  "SOLVER.FIX_PARAM.predictor": true,
  "SOLVER.FIX_PARAM.lang_encoder": true,
  "SOLVER.IGNORE_FIX": [
    "predictor.transformer_cross_attention_layers",
    "lang_encoder.lang_encoder.resblocks.10",
    "lang_encoder.lang_encoder.resblocks.11"
  ],
  "SOLVER.LR_MULTIPLIER.backbone": 1.0,
  "SOLVER.LR_MULTIPLIER.pixel_decoder": 1.0,
  "SOLVER.LR_MULTIPLIER.predictor": 0.2,
  "SOLVER.LR_MULTIPLIER.lang_encoder": 1.0
}
EOF
)

STAGE2_SOLVER_JSON=$(cat <<'EOF'
{
  "SOLVER.FIX_PARAM.backbone": true,
  "SOLVER.FIX_PARAM.pixel_decoder": true,
  "SOLVER.FIX_PARAM.predictor": true,
  "SOLVER.FIX_PARAM.lang_encoder": true,
  "SOLVER.IGNORE_FIX": [
    "predictor.transformer_cross_attention_layers",
    "lang_encoder.lang_encoder.resblocks.8",
    "lang_encoder.lang_encoder.resblocks.9",
    "lang_encoder.lang_encoder.resblocks.10",
    "lang_encoder.lang_encoder.resblocks.11"
  ],
  "SOLVER.LR_MULTIPLIER.backbone": 1.0,
  "SOLVER.LR_MULTIPLIER.pixel_decoder": 1.0,
  "SOLVER.LR_MULTIPLIER.predictor": 0.2,
  "SOLVER.LR_MULTIPLIER.lang_encoder": 1.0
}
EOF
)

STAGE3_SOLVER_JSON=$(cat <<'EOF'
{
  "SOLVER.FIX_PARAM.backbone": true,
  "SOLVER.FIX_PARAM.pixel_decoder": true,
  "SOLVER.FIX_PARAM.predictor": false,
  "SOLVER.FIX_PARAM.lang_encoder": true,
  "SOLVER.IGNORE_FIX": [
    "lang_encoder.lang_encoder.resblocks.8",
    "lang_encoder.lang_encoder.resblocks.9",
    "lang_encoder.lang_encoder.resblocks.10",
    "lang_encoder.lang_encoder.resblocks.11"
  ],
  "SOLVER.LR_MULTIPLIER.backbone": 1.0,
  "SOLVER.LR_MULTIPLIER.pixel_decoder": 1.0,
  "SOLVER.LR_MULTIPLIER.predictor": 0.2,
  "SOLVER.LR_MULTIPLIER.lang_encoder": 1.0
}
EOF
)

CONFIG_STAGE1=$(merge_config_overrides_json "${STAGE1_SOLVER_JSON}")
CONFIG_STAGE2=$(merge_config_overrides_json "${STAGE2_SOLVER_JSON}")
CONFIG_STAGE3=$(merge_config_overrides_json "${STAGE3_SOLVER_JSON}")

# ---------------------------------------------------------------------------
# Stage 1 — warm-up with minimal trainable subset
#   - freeze backbone + pixel decoder + predictor + full text encoder
#   - unfreeze only decoder cross-attention and top text blocks
# ---------------------------------------------------------------------------
run_stage \
  "Stage 1 (warm-up)" \
  "${STAGE1_DIR}" \
  "${PRETRAINED_WEIGHTS}" \
  "${STAGE1_BS_PER_GPU}" \
  "${STAGE1_BS_TOTAL}" \
  "${STAGE1_EPOCHS}" \
  "${STAGE1_LR}" \
  "${STAGE1_WARMUP_ITERS}" \
  "${CONFIG_STAGE1}"

if [ ! -e "${STAGE1_BEST}" ]; then
  echo "ERROR: Stage 1 best checkpoint not found: ${STAGE1_BEST}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Stage 2 — selective unfreezing
#   - keep image encoder + pixel decoder frozen
#   - unfreeze top 4 text blocks + decoder cross-attention layers
# ---------------------------------------------------------------------------
run_stage \
  "Stage 2 (selective unfreezing)" \
  "${STAGE2_DIR}" \
  "${STAGE1_BEST}" \
  "${STAGE2_BS_PER_GPU}" \
  "${STAGE2_BS_TOTAL}" \
  "${STAGE2_EPOCHS}" \
  "${STAGE2_LR}" \
  "${STAGE2_WARMUP_ITERS}" \
  "${CONFIG_STAGE2}"

if [ ! -e "${STAGE2_BEST}" ]; then
  echo "ERROR: Stage 2 best checkpoint not found: ${STAGE2_BEST}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Stage 3 — optional full decoder fine-tuning
#   - keep image encoder + pixel decoder frozen
#   - unfreeze full predictor (decoder + classifier), keep top 4 text blocks
# ---------------------------------------------------------------------------
if [ "${RUN_STAGE3}" = "1" ]; then
  run_stage \
    "Stage 3 (full decoder fine-tuning)" \
    "${STAGE3_DIR}" \
    "${STAGE2_BEST}" \
    "${STAGE3_BS_PER_GPU}" \
    "${STAGE3_BS_TOTAL}" \
    "${STAGE3_EPOCHS}" \
    "${STAGE3_LR}" \
    "${STAGE3_WARMUP_ITERS}" \
    "${CONFIG_STAGE3}"
fi

echo ""
echo "============================================================"
echo "Staged fine-tuning complete."
echo "Stage 1: ${STAGE1_DIR}"
echo "Stage 2: ${STAGE2_DIR}"
if [ "${RUN_STAGE3}" = "1" ]; then
  echo "Stage 3: ${STAGE3_DIR}"
fi
echo "============================================================"
