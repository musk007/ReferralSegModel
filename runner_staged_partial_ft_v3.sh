#!/bin/bash
# =============================================================================
# SLURM job script – staged partial fine-tuning for BiomedParse  [v3]
#
# Changes from v2 (based on comparative v1 vs v2 validation curve analysis):
#
#   ROOT CAUSE SUMMARY:
#     v1: LR too high + too many unfrozen params -> catastrophic forgetting
#     v2: LR too low + too few unfrozen params   -> underfitting / slow learning
#     v3: balanced middle ground
#
#   Stage 1 changes:
#     - LR raised from v2's 1e-5 to 5e-5 (midpoint between v1 and v2)
#     - pixel_decoder unfrozen (FIX_PARAM.pixel_decoder: false)
#       with conservative LR multiplier 0.05
#     - SOLVER.MAX_ITER explicitly set to match actual training steps
#       (fixes the abrupt LR cliff seen in both v1 and v2 at step ~18)
#     - IGNORE_FIX: cross-attention + pixel_decoder (no text blocks yet)
#
#   Stage 2 changes:
#     - LR raised proportionally: 2.5e-5 -> 1.5e-5 (was 2.5e-5 in v2)
#     - lang_encoder LR multiplier kept at 0.5 (5x predictor)
#     - SOLVER.MAX_ITER explicitly set for Stage 2
#
#   Stage 3: unchanged from v2
#
#   Global:
#     - Gradient clipping kept (norm 1.0)
#     - Batch size kept at 4/GPU (32 total)
#     - SOLVER.MAX_ITER computed per stage from epochs and dataset size
#
#   NOTE on eval/precision@0.5:
#     Both v1 and v2 show this metric completely flat at ~3.7.
#     This is a text OOD issue — not fixable by hyperparameters alone.
#     Before running Stage 2, harmonise your clinical text prompts using
#     GPT-4 to loosely align with BiomedParse ontology structure.
#     See bottom of this script for a minimal GPT-4 harmonisation prompt.
#
# Usage:
#   sbatch runner_staged_partial_ft_v3.sh
#   TRAIN_DATASET=colon sbatch runner_staged_partial_ft_v3.sh
#   RUN_STAGE3=1 sbatch runner_staged_partial_ft_v3.sh
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
#SBATCH --job-name=bp_staged_ft_v3
#SBATCH --output=/home/roba/miccai26/logs/finetune_staged_v3_%j.out
#SBATCH --error=/home/roba/miccai26/logs/finetune_staged_v3_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BIOMEDPARSE_DIR="${BIOMEDPARSE_DIR:-/home/roba/miccai26/BiomedParse}"
DATASETS_DIR="${DATASETS_DIR:-/adialab/usr/roba/biomedparse_datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/adialab/usr/roba/train_output/histopath_staged_partial_v3}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-hf_hub:microsoft/BiomedParse}"
BASE_CONF="${BASE_CONF:-configs/biomed_seg_lang_v1.yaml}"
HISTO_CONF="${HISTO_CONF:-biomed_seg_lang_v1_histopath_full.yaml}"

# ---------------------------------------------------------------------------
# GPU / batch setup
# ---------------------------------------------------------------------------
NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"

STAGE1_BS_PER_GPU="${STAGE1_BS_PER_GPU:-4}"
STAGE2_BS_PER_GPU="${STAGE2_BS_PER_GPU:-4}"
STAGE3_BS_PER_GPU="${STAGE3_BS_PER_GPU:-4}"

GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
GRAD_CLIP="${GRAD_CLIP:-True}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

STAGE1_BS_TOTAL=$(( NUM_GPUS * STAGE1_BS_PER_GPU ))
STAGE2_BS_TOTAL=$(( NUM_GPUS * STAGE2_BS_PER_GPU ))
STAGE3_BS_TOTAL=$(( NUM_GPUS * STAGE3_BS_PER_GPU ))

# ---------------------------------------------------------------------------
# Dataset size — used to compute MAX_ITER per stage
# 1436 training samples / 32 total batch size = 45 iters/epoch (ceil)
# Adjust DATASET_TRAIN_SIZE if your train split differs
# ---------------------------------------------------------------------------
DATASET_TRAIN_SIZE="${DATASET_TRAIN_SIZE:-1005}"   # 70% of 1436
ITERS_PER_EPOCH=$(python3 -c "import math; print(math.ceil(${DATASET_TRAIN_SIZE} / ${STAGE1_BS_TOTAL}))")

# ---------------------------------------------------------------------------
# Stage hyperparameters
# ---------------------------------------------------------------------------
STAGE1_EPOCHS="${STAGE1_EPOCHS:-35}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-20}"

# v3: LR is midpoint between v1 (1e-4) and v2 (1e-5)
STAGE1_LR="${STAGE1_LR:-0.00005}"
STAGE2_LR="${STAGE2_LR:-0.000015}"
STAGE3_LR="${STAGE3_LR:-0.000005}"

# v3: warmup = ~5% of total stage iters, matched to actual schedule
STAGE1_WARMUP_ITERS="${STAGE1_WARMUP_ITERS:-22}"
STAGE2_WARMUP_ITERS="${STAGE2_WARMUP_ITERS:-22}"
STAGE3_WARMUP_ITERS="${STAGE3_WARMUP_ITERS:-22}"

# v3: explicit MAX_ITER per stage to fix abrupt LR cliff seen in v1 and v2
STAGE1_MAX_ITER=$(( ITERS_PER_EPOCH * STAGE1_EPOCHS ))
STAGE2_MAX_ITER=$(( ITERS_PER_EPOCH * STAGE2_EPOCHS ))
STAGE3_MAX_ITER=$(( ITERS_PER_EPOCH * STAGE3_EPOCHS ))

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
echo "BiomedParse staged partial fine-tuning [v3]"
echo "Job ID          : ${SLURM_JOB_ID:-local}"
echo "Node            : $(hostname)"
echo "GPUs            : ${NUM_GPUS}"
echo "Output root     : ${OUTPUT_ROOT}"
echo "Dataset size    : ${DATASET_TRAIN_SIZE} training samples"
echo "Iters/epoch     : ${ITERS_PER_EPOCH}"
echo "Gradient clip   : ${GRAD_CLIP} (norm ${GRAD_CLIP_NORM})"
echo "Run Stage 3     : ${RUN_STAGE3}"
echo ""
echo "v3 key changes vs v2:"
echo "  S1 LR          : 1e-5 -> ${STAGE1_LR}  (midpoint v1/v2)"
echo "  S1 unfreeze    : cross-attn + pixel_decoder (LR mult 0.05)"
echo "  S1 MAX_ITER    : explicitly set to ${STAGE1_MAX_ITER}"
echo "  S2 MAX_ITER    : explicitly set to ${STAGE2_MAX_ITER}"
echo "  S2 LR          : 2.5e-5 -> ${STAGE2_LR}"
echo "  LR cliff fix   : SOLVER.MAX_ITER set per stage"
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
  local max_iter="$9"
  local config_overrides_json="${10}"

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
  echo "MAX_ITER       : ${max_iter}"
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
    SOLVER.MAX_ITER ${max_iter} \
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
# Stage 1 JSON — v3 changes:
#   - pixel_decoder unfrozen (key addition vs v2)
#   - pixel_decoder LR multiplier: 0.05 (conservative — 0.5x predictor)
#   - IGNORE_FIX: cross-attention + pixel_decoder (no text blocks)
#   - backbone and lang_encoder remain fully frozen (LR mult 0.0)
#   - predictor LR multiplier: 0.1 (same as v2)
# ---------------------------------------------------------------------------
STAGE1_SOLVER_JSON=$(cat <<'EOF'
{
  "SOLVER.FIX_PARAM.backbone": true,
  "SOLVER.FIX_PARAM.pixel_decoder": false,
  "SOLVER.FIX_PARAM.predictor": true,
  "SOLVER.FIX_PARAM.lang_encoder": true,
  "SOLVER.IGNORE_FIX": [
    "predictor.transformer_cross_attention_layers"
  ],
  "SOLVER.LR_MULTIPLIER.backbone": 0.0,
  "SOLVER.LR_MULTIPLIER.pixel_decoder": 0.05,
  "SOLVER.LR_MULTIPLIER.predictor": 0.1,
  "SOLVER.LR_MULTIPLIER.lang_encoder": 0.0
}
EOF
)

# ---------------------------------------------------------------------------
# Stage 2 JSON — same as v2 but with proportionally reduced LR
#   - Top 4 text blocks (8-11) unfrozen for OOD clinical text adaptation
#   - lang_encoder LR multiplier: 0.5 (5x predictor) — kept from v2
#   - pixel_decoder remains unfrozen from Stage 1
#   - backbone frozen throughout
# ---------------------------------------------------------------------------
STAGE2_SOLVER_JSON=$(cat <<'EOF'
{
  "SOLVER.FIX_PARAM.backbone": true,
  "SOLVER.FIX_PARAM.pixel_decoder": false,
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
  "SOLVER.LR_MULTIPLIER.pixel_decoder": 0.05,
  "SOLVER.LR_MULTIPLIER.predictor": 0.1,
  "SOLVER.LR_MULTIPLIER.lang_encoder": 0.5
}
EOF
)

# ---------------------------------------------------------------------------
# Stage 3 JSON — full decoder fine-tuning (optional)
#   - Full predictor unfrozen
#   - Top 4 text blocks remain unfrozen
#   - pixel_decoder remains unfrozen
#   - backbone frozen throughout all stages
# ---------------------------------------------------------------------------
STAGE3_SOLVER_JSON=$(cat <<'EOF'
{
  "SOLVER.FIX_PARAM.backbone": true,
  "SOLVER.FIX_PARAM.pixel_decoder": false,
  "SOLVER.FIX_PARAM.predictor": false,
  "SOLVER.FIX_PARAM.lang_encoder": true,
  "SOLVER.IGNORE_FIX": [
    "lang_encoder.lang_encoder.resblocks.8",
    "lang_encoder.lang_encoder.resblocks.9",
    "lang_encoder.lang_encoder.resblocks.10",
    "lang_encoder.lang_encoder.resblocks.11"
  ],
  "SOLVER.LR_MULTIPLIER.backbone": 0.0,
  "SOLVER.LR_MULTIPLIER.pixel_decoder": 0.05,
  "SOLVER.LR_MULTIPLIER.predictor": 0.1,
  "SOLVER.LR_MULTIPLIER.lang_encoder": 0.5
}
EOF
)

CONFIG_STAGE1=$(merge_config_overrides_json "${STAGE1_SOLVER_JSON}")
CONFIG_STAGE2=$(merge_config_overrides_json "${STAGE2_SOLVER_JSON}")
CONFIG_STAGE3=$(merge_config_overrides_json "${STAGE3_SOLVER_JSON}")

# ---------------------------------------------------------------------------
# Stage 1 — balanced warm-up
#   Freeze  : backbone, lang_encoder
#   Unfreeze: pixel_decoder (LR x0.05), decoder cross-attention (LR x0.1)
#   Goal    : learn faster than v2 without catastrophic forgetting as in v1
#   Watch   : eval/mIoU should be stable or rising from epoch 5 onwards
#             train/epoch_loss_avg should descend (not flat as in v2)
#             train/lr_default should show smooth cosine decay (not cliff)
# ---------------------------------------------------------------------------
run_stage \
  "Stage 1 (balanced warm-up) [v3]" \
  "${STAGE1_DIR}" \
  "${PRETRAINED_WEIGHTS}" \
  "${STAGE1_BS_PER_GPU}" \
  "${STAGE1_BS_TOTAL}" \
  "${STAGE1_EPOCHS}" \
  "${STAGE1_LR}" \
  "${STAGE1_WARMUP_ITERS}" \
  "${STAGE1_MAX_ITER}" \
  "${CONFIG_STAGE1}"

if [ ! -e "${STAGE1_BEST}" ]; then
  echo "ERROR: Stage 1 best checkpoint not found: ${STAGE1_BEST}"
  echo ""
  echo "Diagnostic checklist:"
  echo "  1. Check train/epoch_loss_avg — if still flat, raise STAGE1_LR to 7e-5"
  echo "  2. Check train/lr_default — should now show smooth cosine, not cliff"
  echo "  3. Check eval/mIoU — should be stable or rising (not declining as v1)"
  echo "  4. If eval/precision@0.5 still flat, run GPT-4 prompt harmonisation"
  echo "     before Stage 2 (see note at bottom of this script)"
  exit 1
fi

echo ""
echo "------------------------------------------------------------"
echo "Stage 1 complete. Before launching Stage 2:"
echo "  1. Verify eval/mIoU is stable or rising in WandB"
echo "  2. Verify train/epoch_loss_avg is descending (not flat)"
echo "  3. Verify train/lr_default shows smooth cosine decay"
echo "  4. If eval/precision@0.5 is still flat at ~3.7, harmonise"
echo "     your clinical text prompts before Stage 2 (see note below)"
echo "------------------------------------------------------------"

# ---------------------------------------------------------------------------
# Stage 2 — selective text encoder unfreezing
#   Freeze  : backbone
#   Unfreeze: pixel_decoder, top 4 text blocks (8-11), cross-attention
#   lang LR : 5x predictor LR for stronger OOD text adaptation
#   Goal    : adapt text encoder to clinical free-text descriptions
#   Watch   : eval/precision@0.5 should start moving (if prompts harmonised)
#             eval/mIoU and eval/mDice should continue rising
# ---------------------------------------------------------------------------
run_stage \
  "Stage 2 (selective text unfreezing) [v3]" \
  "${STAGE2_DIR}" \
  "${STAGE1_BEST}" \
  "${STAGE2_BS_PER_GPU}" \
  "${STAGE2_BS_TOTAL}" \
  "${STAGE2_EPOCHS}" \
  "${STAGE2_LR}" \
  "${STAGE2_WARMUP_ITERS}" \
  "${STAGE2_MAX_ITER}" \
  "${CONFIG_STAGE2}"

if [ ! -e "${STAGE2_BEST}" ]; then
  echo "ERROR: Stage 2 best checkpoint not found: ${STAGE2_BEST}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Stage 3 — optional full decoder fine-tuning
#   Only run if:
#     - Stage 2 eval/mIoU has plateaued for >5 epochs
#     - val/train Dice gap is <5%
#     - eval/precision@0.5 has started moving (text encoder adapted)
#   If val/train gap >5% at end of Stage 2, do NOT run Stage 3 —
#   the dataset is too small for further unfreezing.
# ---------------------------------------------------------------------------
if [ "${RUN_STAGE3}" = "1" ]; then
  echo ""
  echo "WARNING: Only run Stage 3 if:"
  echo "  - Stage 2 eval/mIoU has plateaued for >5 epochs"
  echo "  - val/train Dice gap < 5%"
  echo "  - eval/precision@0.5 has moved above its flat baseline"
  run_stage \
    "Stage 3 (full decoder fine-tuning) [v3]" \
    "${STAGE3_DIR}" \
    "${STAGE2_BEST}" \
    "${STAGE3_BS_PER_GPU}" \
    "${STAGE3_BS_TOTAL}" \
    "${STAGE3_EPOCHS}" \
    "${STAGE3_LR}" \
    "${STAGE3_WARMUP_ITERS}" \
    "${STAGE3_MAX_ITER}" \
    "${CONFIG_STAGE3}"
fi

echo ""
echo "============================================================"
echo "Staged fine-tuning [v3] complete."
echo "Stage 1 : ${STAGE1_DIR}"
echo "Stage 2 : ${STAGE2_DIR}"
if [ "${RUN_STAGE3}" = "1" ]; then
  echo "Stage 3 : ${STAGE3_DIR}"
fi
echo ""
echo "Next steps:"
echo "  1. Compare v3 vs v2 in WandB — look for:"
echo "     - train/epoch_loss_avg descending (not flat as v2)"
echo "     - eval/mIoU rising (not declining as v1)"
echo "     - train/lr_default smooth cosine (not cliff as v1/v2)"
echo "  2. If eval/precision@0.5 is still flat after Stage 2,"
echo "     the text OOD gap is too large for adapter-only finetuning."
echo "     Consider full text encoder unfreezing in a Stage 2b run."
echo "============================================================"

# =============================================================================
# NOTE: GPT-4 prompt harmonisation for eval/precision@0.5
#
# Both v1 and v2 show eval/precision@0.5 completely flat at ~3.7.
# This means the model cannot produce confident predictions (>0.5)
# for your clinical free-text prompts — a text OOD issue that
# hyperparameter tuning alone cannot fix.
#
# Before Stage 2, run your clinical descriptions through GPT-4
# using this prompt template to loosely align them with BiomedParse's
# ontology while preserving clinical specificity:
#
#   System: You are a biomedical text harmonisation assistant.
#   Convert the following clinical description into a structured
#   biomedical object description following this format:
#   "[tissue type] showing [object type] in [anatomical site]"
#   Use only terms consistent with histopathology ontology.
#   Preserve the clinical meaning. Do not invent information.
#
#   User: <your clinical free-text description here>
#
# Store the harmonised prompts alongside your original ones and
# use them as your primary text input for Stage 2 onwards.
# =============================================================================