#!/bin/bash
# =============================================================================
# SLURM job script – staged partial fine-tuning for BiomedParse  [v2]
#
# Changes from v1 (based on validation curve analysis):
#   - Stage 1: unfreeze ONLY decoder cross-attention (removed text blocks)
#   - Stage 1: LR reduced 10x (1e-4 -> 1e-5) to prevent catastrophic forgetting
#   - Stage 1: batch size per GPU halved (8 -> 4) to suit 1436-sample dataset
#   - Stage 1: warmup reduced to ~22 iters (matches actual iters/epoch)
#   - Stage 1: LR multipliers set to 0.0 for all frozen components
#   - All stages: gradient clipping enabled (norm 1.0)
#   - Stage 2: text encoder blocks 8-11 unfrozen (previously 10-11 only in S1)
#   - Stage 2: LR multiplier for lang_encoder set to 5x predictor LR
#     (text encoder needs more adaptation for OOD clinical free-text)
#   - Stage 3: unchanged from v1
#
# Stages:
#   1) Warm-up  — freeze everything; unfreeze only decoder cross-attention
#   2) Selective — unfreeze top 4 text blocks + decoder cross-attention
#   3) Optional  — full decoder fine-tuning (image encoder still frozen)
#
# Usage:
#   sbatch runner_staged_partial_ft_v2.sh
#   TRAIN_DATASET=colon sbatch runner_staged_partial_ft_v2.sh
#   RUN_STAGE3=1 sbatch runner_staged_partial_ft_v2.sh
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
#SBATCH --job-name=bp_staged_ft_v2
#SBATCH --output=/home/roba/miccai26/logs/finetune_staged_v2_%j.out
#SBATCH --error=/home/roba/miccai26/logs/finetune_staged_v2_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BIOMEDPARSE_DIR="${BIOMEDPARSE_DIR:-/home/roba/miccai26/BiomedParse}"
DATASETS_DIR="${DATASETS_DIR:-/adialab/usr/roba/biomedparse_datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/adialab/usr/roba/train_output/histopath_staged_partial_v2}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-hf_hub:microsoft/BiomedParse}"
BASE_CONF="${BASE_CONF:-configs/biomed_seg_lang_v1.yaml}"
HISTO_CONF="${HISTO_CONF:-biomed_seg_lang_v1_histopath_full.yaml}"

# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"

# v2: batch size per GPU halved in S1 and S2 (small dataset guard)
STAGE1_BS_PER_GPU="${STAGE1_BS_PER_GPU:-4}"
STAGE2_BS_PER_GPU="${STAGE2_BS_PER_GPU:-4}"
STAGE3_BS_PER_GPU="${STAGE3_BS_PER_GPU:-4}"

GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"

# v2: gradient clipping enabled globally
GRAD_CLIP="${GRAD_CLIP:-True}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

STAGE1_BS_TOTAL=$(( NUM_GPUS * STAGE1_BS_PER_GPU ))
STAGE2_BS_TOTAL=$(( NUM_GPUS * STAGE2_BS_PER_GPU ))
STAGE3_BS_TOTAL=$(( NUM_GPUS * STAGE3_BS_PER_GPU ))

# ---------------------------------------------------------------------------
# Stage hyperparameters
# ---------------------------------------------------------------------------
# v2 Stage 1: extended to 35 epochs, LR reduced 10x, warmup matched to
# actual iters/epoch (1436 samples / 32 batch = ~45 iters/epoch; 5% = ~22)
STAGE1_EPOCHS="${STAGE1_EPOCHS:-35}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-20}"

STAGE1_LR="${STAGE1_LR:-0.00001}"     # v2: was 0.0001
STAGE2_LR="${STAGE2_LR:-0.000025}"    # v2: slightly lower than before
STAGE3_LR="${STAGE3_LR:-0.000005}"    # v2: proportionally lower

STAGE1_WARMUP_ITERS="${STAGE1_WARMUP_ITERS:-22}"   # v2: was 100
STAGE2_WARMUP_ITERS="${STAGE2_WARMUP_ITERS:-22}"   # v2: was 50
STAGE3_WARMUP_ITERS="${STAGE3_WARMUP_ITERS:-22}"   # v2: was 50

WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
RUN_STAGE3="${RUN_STAGE3:-0}"

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
echo "BiomedParse staged partial fine-tuning [v2]"
echo "Job ID          : ${SLURM_JOB_ID:-local}"
echo "Node            : $(hostname)"
echo "GPUs            : ${NUM_GPUS}"
echo "Output root     : ${OUTPUT_ROOT}"
echo "Image encoder   : frozen in all stages"
echo "Augmentations   : disabled in all stages"
echo "Gradient clip   : ${GRAD_CLIP} (norm ${GRAD_CLIP_NORM})"
echo "Run Stage 3     : ${RUN_STAGE3}"
echo ""
echo "v2 key changes vs v1:"
echo "  S1 LR          : 0.0001 -> ${STAGE1_LR}"
echo "  S1 BS/GPU      : 8 -> ${STAGE1_BS_PER_GPU}"
echo "  S1 warmup      : 100 -> ${STAGE1_WARMUP_ITERS} iters"
echo "  S1 epochs      : 20 -> ${STAGE1_EPOCHS}"
echo "  S1 unfreeze    : cross-attn only (removed text resblocks 10-11)"
echo "  All LR mults   : frozen components set to 0.0"
echo "  Grad clipping  : enabled globally"
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
    SOLVER.CLIP_GRADIENTS.CLIP_VALUE ${GRAD_CLIP_NORM} \
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

# ---------------------------------------------------------------------------
# Stage 1 JSON — v2 changes:
#   - IGNORE_FIX: cross-attention ONLY (text resblocks removed)
#   - LR multipliers: 0.0 for all frozen components (extra safety layer)
#   - predictor LR multiplier: reduced from 0.2 to 0.1 (more conservative)
# ---------------------------------------------------------------------------
STAGE1_SOLVER_JSON=$(cat <<'EOF'
{
  "SOLVER.FIX_PARAM.backbone": true,
  "SOLVER.FIX_PARAM.pixel_decoder": true,
  "SOLVER.FIX_PARAM.predictor": true,
  "SOLVER.FIX_PARAM.lang_encoder": true,
  "SOLVER.IGNORE_FIX": [
    "predictor.transformer_cross_attention_layers"
  ],
  "SOLVER.LR_MULTIPLIER.backbone": 0.0,
  "SOLVER.LR_MULTIPLIER.pixel_decoder": 0.0,
  "SOLVER.LR_MULTIPLIER.predictor": 0.1,
  "SOLVER.LR_MULTIPLIER.lang_encoder": 0.0
}
EOF
)

# ---------------------------------------------------------------------------
# Stage 2 JSON — v2 changes:
#   - IGNORE_FIX: top 4 text blocks (8-11) + cross-attention
#   - lang_encoder LR multiplier: 0.5 (5x predictor LR) to push text
#     encoder adaptation harder for OOD clinical free-text prompts
#   - backbone/pixel_decoder LR multipliers: remain 0.0
# ---------------------------------------------------------------------------
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
  "SOLVER.LR_MULTIPLIER.backbone": 0.0,
  "SOLVER.LR_MULTIPLIER.pixel_decoder": 0.0,
  "SOLVER.LR_MULTIPLIER.predictor": 0.1,
  "SOLVER.LR_MULTIPLIER.lang_encoder": 0.5
}
EOF
)

# ---------------------------------------------------------------------------
# Stage 3 JSON — unchanged from v1 in structure; LR multipliers updated
#   - Full predictor unfrozen
#   - Top 4 text blocks remain unfrozen
#   - Image encoder + pixel decoder remain frozen throughout
# ---------------------------------------------------------------------------
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
  "SOLVER.LR_MULTIPLIER.backbone": 0.0,
  "SOLVER.LR_MULTIPLIER.pixel_decoder": 0.0,
  "SOLVER.LR_MULTIPLIER.predictor": 0.1,
  "SOLVER.LR_MULTIPLIER.lang_encoder": 0.5
}
EOF
)

CONFIG_STAGE1=$(merge_config_overrides_json "${STAGE1_SOLVER_JSON}")
CONFIG_STAGE2=$(merge_config_overrides_json "${STAGE2_SOLVER_JSON}")
CONFIG_STAGE3=$(merge_config_overrides_json "${STAGE3_SOLVER_JSON}")

# ---------------------------------------------------------------------------
# Stage 1 — minimal warm-up
#   Freeze: backbone, pixel_decoder, full predictor, full lang_encoder
#   Unfreeze: decoder cross-attention layers ONLY
#   Goal: stabilise loss without forgetting pretrained representations
# ---------------------------------------------------------------------------
run_stage \
  "Stage 1 (warm-up) [v2 — cross-attn only, LR=1e-5]" \
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
  echo "Check WandB logs — if eval/mIoU is still declining at epoch 35,"
  echo "consider reducing STAGE1_LR further to 5e-6 before running Stage 2."
  exit 1
fi

# ---------------------------------------------------------------------------
# Stage 2 — selective unfreezing
#   Freeze: backbone, pixel_decoder
#   Unfreeze: top 4 text blocks (8-11), decoder cross-attention
#   lang_encoder LR = 5x predictor LR (OOD text adaptation)
#   Goal: adapt text encoder to clinical free-text while preserving
#         image features
# ---------------------------------------------------------------------------
run_stage \
  "Stage 2 (selective unfreezing) [v2 — text blocks 8-11, lang LR x5]" \
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
#   Freeze: backbone, pixel_decoder (image encoder frozen throughout)
#   Unfreeze: full predictor + top 4 text blocks
#   Only run if Stage 2 eval/mIoU has plateaued and val/train gap is <5%
# ---------------------------------------------------------------------------
if [ "${RUN_STAGE3}" = "1" ]; then
  echo ""
  echo "WARNING: Only run Stage 3 if Stage 2 val/train Dice gap < 5%."
  echo "If gap is larger, the dataset is too small for further unfreezing."
  run_stage \
    "Stage 3 (full decoder fine-tuning) [v2]" \
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
echo "Staged fine-tuning [v2] complete."
echo "Stage 1 : ${STAGE1_DIR}"
echo "Stage 2 : ${STAGE2_DIR}"
if [ "${RUN_STAGE3}" = "1" ]; then
  echo "Stage 3 : ${STAGE3_DIR}"
fi
echo ""
echo "Next steps:"
echo "  1. Check WandB: eval/mIoU and eval/mDice should be stable or rising"
echo "  2. Check eval/precision@0.5 — if still flat after Stage 1, text"
echo "     prompts may need GPT-4 harmonisation before Stage 2"
echo "  3. Only proceed to Stage 3 if val/train Dice gap < 5%"
echo "============================================================"