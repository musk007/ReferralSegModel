#!/bin/bash
# =============================================================================
# SLURM job script – staged partial fine-tuning for BiomedParse  [v3 + ES]
#
# Changes from v3:
#   - Stage 2 now runs epoch-by-epoch with early stopping
#   - A background early_stopping_monitor.py process watches eval/best_mIoU
#     via WandB API and writes a STOP file when patience is exhausted
#   - The epoch loop checks for the STOP file after every checkpoint interval
#     and exits Stage 2 cleanly, preserving the best checkpoint
#   - Stage 1 and Stage 3 are unchanged from v3
#
# Early stopping settings (Stage 2):
#   PATIENCE        : stop after N epochs with no improvement (default 5)
#   MIN_EPOCHS_S2   : never stop before this epoch (default 10)
#   ES_MIN_DELTA    : minimum mIoU gain to count as improvement (default 0.1)
#   ES_POLL_INTERVAL: seconds between WandB polls (default 60)
#
# WandB settings (required for early stopping):
#   WANDB_ENTITY    : your WandB username or team
#   WANDB_PROJECT   : your WandB project name
#   These are read from environment or set below.
#
# Usage:
#   sbatch runner_staged_partial_ft_v4.sh
#   TRAIN_DATASET=colon sbatch runner_staged_partial_ft_v4.sh
#   RUN_STAGE3=1 sbatch runner_staged_partial_ft_v4.sh
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
#SBATCH --job-name=bp_staged_ft_v4
#SBATCH --output=/home/roba/miccai26/logs/finetune_staged_v4_%j.out
#SBATCH --error=/home/roba/miccai26/logs/finetune_staged_v4_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BIOMEDPARSE_DIR="${BIOMEDPARSE_DIR:-/home/roba/miccai26/BiomedParse}"
DATASETS_DIR="${DATASETS_DIR:-/adialab/usr/roba/biomedparse_datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/adialab/usr/roba/train_output/histopath_staged_v4}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-hf_hub:microsoft/BiomedParse}"
BASE_CONF="${BASE_CONF:-configs/biomed_seg_lang_v1.yaml}"
HISTO_CONF="${HISTO_CONF:-biomed_seg_lang_v1_histopath_full.yaml}"

# Path to the early stopping monitor script (copy it next to this script)
ES_MONITOR_SCRIPT="${ES_MONITOR_SCRIPT:-/home/roba/miccai26/early_stopping_monitor.py}"

# ---------------------------------------------------------------------------
# WandB settings — required for early stopping monitor
# ---------------------------------------------------------------------------
WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"      # <-- set your entity
WANDB_PROJECT="${WANDB_PROJECT:-your_wandb_project}"   # <-- set your project
export WANDB_KEY="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"

# ---------------------------------------------------------------------------
# Early stopping hyperparameters (Stage 2 only)
# ---------------------------------------------------------------------------
ES_PATIENCE="${ES_PATIENCE:-5}"          # epochs without improvement before stop
ES_MIN_EPOCHS="${ES_MIN_EPOCHS:-10}"     # never stop before this epoch
ES_MIN_DELTA="${ES_MIN_DELTA:-0.1}"      # minimum mIoU gain to count as improvement
ES_POLL_INTERVAL="${ES_POLL_INTERVAL:-60}"  # seconds between WandB polls
ES_CHECK_INTERVAL="${ES_CHECK_INTERVAL:-2}" # check stop file every N epochs

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
# Dataset size and iter computation
# ---------------------------------------------------------------------------
DATASET_TRAIN_SIZE="${DATASET_TRAIN_SIZE:-1005}"
ITERS_PER_EPOCH=$(python3 -c \
  "import math; print(math.ceil(${DATASET_TRAIN_SIZE} / ${STAGE1_BS_TOTAL}))")

# ---------------------------------------------------------------------------
# Stage hyperparameters
# ---------------------------------------------------------------------------
STAGE1_EPOCHS="${STAGE1_EPOCHS:-35}"
STAGE2_MAX_EPOCHS="${STAGE2_MAX_EPOCHS:-30}"  # upper bound; ES may stop earlier
STAGE3_EPOCHS="${STAGE3_EPOCHS:-20}"

STAGE1_LR="${STAGE1_LR:-0.00005}"
STAGE2_LR="${STAGE2_LR:-0.000015}"
STAGE3_LR="${STAGE3_LR:-0.000005}"

STAGE1_WARMUP_ITERS="${STAGE1_WARMUP_ITERS:-22}"
STAGE2_WARMUP_ITERS="${STAGE2_WARMUP_ITERS:-22}"
STAGE3_WARMUP_ITERS="${STAGE3_WARMUP_ITERS:-22}"

STAGE1_MAX_ITER=$(( ITERS_PER_EPOCH * STAGE1_EPOCHS ))
STAGE2_MAX_ITER=$(( ITERS_PER_EPOCH * STAGE2_MAX_EPOCHS ))
STAGE3_MAX_ITER=$(( ITERS_PER_EPOCH * STAGE3_EPOCHS ))

WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
RUN_STAGE3="${RUN_STAGE3:-0}"

# ---------------------------------------------------------------------------
# Stop file paths
# ---------------------------------------------------------------------------
ES_STOP_FILE="/tmp/bp_stage2_stop_${SLURM_JOB_ID:-local}"
ES_LOG_FILE="${OUTPUT_ROOT}/stage2_es_monitor.log"
ES_MONITOR_PID_FILE="/tmp/bp_stage2_monitor_pid_${SLURM_JOB_ID:-local}"

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

mkdir -p /home/roba/miccai26/logs
mkdir -p "${OUTPUT_ROOT}"

# Clean up any stale stop file from a previous run
rm -f "${ES_STOP_FILE}"

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
      echo "Unknown TRAIN_DATASET=${TRAIN_DATASET}."
      exit 1
      ;;
  esac
  TRAIN_DS="biomed_${PREFIX}_train"
  EVAL_DS="biomed_${PREFIX}_eval"
  TEST_DS="biomed_${PREFIX}_test"
  OUTPUT_ROOT="${OUTPUT_ROOT}_${TRAIN_DATASET}"
  mkdir -p "${OUTPUT_ROOT}"
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
echo "BiomedParse staged partial fine-tuning [v3 + early stopping]"
echo "Job ID          : ${SLURM_JOB_ID:-local}"
echo "Node            : $(hostname)"
echo "GPUs            : ${NUM_GPUS}"
echo "Output root     : ${OUTPUT_ROOT}"
echo "Iters/epoch     : ${ITERS_PER_EPOCH}"
echo "Grad clip       : ${GRAD_CLIP} (norm ${GRAD_CLIP_NORM})"
echo "Run Stage 3     : ${RUN_STAGE3}"
echo ""
echo "Early stopping (Stage 2 only):"
echo "  Metric        : eval/best_mIoU"
echo "  Patience      : ${ES_PATIENCE} epochs"
echo "  Min epochs    : ${ES_MIN_EPOCHS}"
echo "  Min delta     : ${ES_MIN_DELTA}"
echo "  Poll interval : ${ES_POLL_INTERVAL}s"
echo "  Stop file     : ${ES_STOP_FILE}"
echo "============================================================"

# Shared no-augmentation overrides
NO_AUG_OVERRIDES=(
  BioMed.INPUT.AUGMENT False
  BioMed.INPUT.CROP.ENABLED False
  BioMed.INPUT.RANDOM_FLIP none
  BioMed.INPUT.RANDOM_ROTATE False
  BioMed.INPUT.COLOR_AUG_SSD False
)

# ---------------------------------------------------------------------------
# Standard run_stage — used for Stage 1 and Stage 3
# ---------------------------------------------------------------------------
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
  echo "Save dir    : ${stage_dir}"
  echo "Resume from : ${resume_from}"
  echo "BS/GPU      : ${bs_per_gpu}  |  BS total: ${bs_total}"
  echo "Epochs      : ${epochs}  |  MAX_ITER: ${max_iter}"
  echo "LR          : ${lr}  |  Warmup: ${warmup_iters}"
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

# ---------------------------------------------------------------------------
# run_stage2_with_early_stopping
# Runs Stage 2 in ES_CHECK_INTERVAL-epoch chunks, polling the stop file
# between chunks. The background monitor writes the stop file when patience
# is exhausted. Training resumes from the latest checkpoint each chunk.
# ---------------------------------------------------------------------------
run_stage2_with_early_stopping() {
  local stage_dir="$1"
  local resume_from="$2"
  local config_overrides_json="$3"

  mkdir -p "${stage_dir}"

  echo ""
  echo "------------------------------------------------------------"
  echo "Running Stage 2 with early stopping"
  echo "Save dir      : ${stage_dir}"
  echo "Resume from   : ${resume_from}"
  echo "Max epochs    : ${STAGE2_MAX_EPOCHS}"
  echo "Check interval: every ${ES_CHECK_INTERVAL} epochs"
  echo "------------------------------------------------------------"

  # Fetch the WandB run ID for the Stage 2 run.
  # BiomedParse writes the run ID to wandb/latest-run after the first epoch.
  # We poll for it to start the monitor once training has initialised.
  get_wandb_run_id() {
    local wandb_dir="${stage_dir}/wandb/latest-run"
    if [ -f "${wandb_dir}" ]; then
      basename "$(cat "${wandb_dir}")" | sed 's/run-[0-9]*-//'
    fi
  }

  local current_resume="${resume_from}"
  local epoch_start=1
  local stopped=0
  local monitor_pid=""

  while [ "${epoch_start}" -le "${STAGE2_MAX_EPOCHS}" ]; do

    local epoch_end=$(( epoch_start + ES_CHECK_INTERVAL - 1 ))
    if [ "${epoch_end}" -gt "${STAGE2_MAX_EPOCHS}" ]; then
      epoch_end="${STAGE2_MAX_EPOCHS}"
    fi
    local chunk_iters=$(( ITERS_PER_EPOCH * epoch_end ))

    echo ""
    echo "  Stage 2 chunk: epochs ${epoch_start} -> ${epoch_end}"
    echo "  Resuming from: ${current_resume}"

    # Run this chunk — MAX_NUM_EPOCHS controls when it stops
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
      TRAIN.BATCH_SIZE_TOTAL ${STAGE2_BS_TOTAL} \
      TRAIN.BATCH_SIZE_PER_GPU ${STAGE2_BS_PER_GPU} \
      TEST.BATCH_SIZE_TOTAL ${STAGE2_BS_TOTAL} \
      SOLVER.MAX_NUM_EPOCHS ${epoch_end} \
      SOLVER.MAX_ITER ${chunk_iters} \
      SOLVER.BASE_LR ${STAGE2_LR} \
      SOLVER.WARMUP_ITERS ${STAGE2_WARMUP_ITERS} \
      SOLVER.WARMUP_FACTOR 1 \
      SOLVER.OPTIMIZER ADAMW \
      SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
      SOLVER.CLIP_GRAD_NORM ${GRAD_CLIP_NORM} \
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
      RESUME_FROM "${current_resume}" \
      "${NO_AUG_OVERRIDES[@]}"

    # Start the background monitor after the first chunk
    # (WandB run ID is now available)
    if [ -z "${monitor_pid}" ]; then
      local wandb_run_id
      wandb_run_id=$(get_wandb_run_id)
      if [ -n "${wandb_run_id}" ]; then
        echo "  Starting early stopping monitor (WandB run: ${wandb_run_id})"
        python3 "${ES_MONITOR_SCRIPT}" \
          --wandb_run_id    "${wandb_run_id}" \
          --wandb_project   "${WANDB_PROJECT}" \
          --wandb_entity    "${WANDB_ENTITY}" \
          --metric          "eval/best_mIoU" \
          --patience        "${ES_PATIENCE}" \
          --min_epochs      "${ES_MIN_EPOCHS}" \
          --min_delta       "${ES_MIN_DELTA}" \
          --poll_interval   "${ES_POLL_INTERVAL}" \
          --stop_file       "${ES_STOP_FILE}" \
          --log_file        "${ES_LOG_FILE}" &
        monitor_pid=$!
        echo "${monitor_pid}" > "${ES_MONITOR_PID_FILE}"
        echo "  Monitor PID: ${monitor_pid}"
      else
        echo "  WARNING: Could not determine WandB run ID — early stopping disabled"
      fi
    fi

    # Update resume checkpoint to latest saved checkpoint
    local latest_ckpt="${stage_dir}/run_1/model_$(printf '%010d' ${chunk_iters}).pt"
    local best_ckpt="${stage_dir}/run_1/best_model"
    if [ -e "${latest_ckpt}" ]; then
      current_resume="${latest_ckpt}"
    elif [ -e "${best_ckpt}" ]; then
      current_resume="${best_ckpt}"
    fi

    # Check stop file
    if [ -f "${ES_STOP_FILE}" ]; then
      echo ""
      echo "  *** EARLY STOPPING TRIGGERED ***"
      echo "  Stop file contents:"
      cat "${ES_STOP_FILE}"
      echo ""
      echo "  Best checkpoint: ${best_ckpt}"
      stopped=1
      break
    fi

    epoch_start=$(( epoch_end + 1 ))

  done

  # Clean up monitor process
  if [ -n "${monitor_pid}" ] && kill -0 "${monitor_pid}" 2>/dev/null; then
    echo "  Stopping early stopping monitor (PID ${monitor_pid})"
    kill "${monitor_pid}" 2>/dev/null || true
    rm -f "${ES_MONITOR_PID_FILE}"
  fi

  rm -f "${ES_STOP_FILE}"

  if [ "${stopped}" -eq 1 ]; then
    echo "Stage 2 ended via early stopping."
  else
    echo "Stage 2 completed all ${STAGE2_MAX_EPOCHS} epochs."
  fi
}

# ---------------------------------------------------------------------------
# Config JSONs — identical to v3
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

# Stage directories/checkpoints
STAGE1_DIR="${OUTPUT_ROOT}/stage1_warmup"
STAGE2_DIR="${OUTPUT_ROOT}/stage2_selective"
STAGE3_DIR="${OUTPUT_ROOT}/stage3_decoder_full"

STAGE1_BEST="${STAGE1_DIR}/run_1/best_model"
STAGE2_BEST="${STAGE2_DIR}/run_1/best_model"

# ---------------------------------------------------------------------------
# Stage 1 — balanced warm-up (unchanged from v3)
# ---------------------------------------------------------------------------
run_stage \
  "Stage 1 (balanced warm-up) [v3+ES]" \
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
  exit 1
fi

echo ""
echo "------------------------------------------------------------"
echo "Stage 1 complete. Launching Stage 2 with early stopping."
echo "Monitor: ${ES_LOG_FILE}"
echo "------------------------------------------------------------"

# ---------------------------------------------------------------------------
# Stage 2 — selective unfreezing with early stopping
# ---------------------------------------------------------------------------
run_stage2_with_early_stopping \
  "${STAGE2_DIR}" \
  "${STAGE1_BEST}" \
  "${CONFIG_STAGE2}"

if [ ! -e "${STAGE2_BEST}" ]; then
  echo "ERROR: Stage 2 best checkpoint not found: ${STAGE2_BEST}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Stage 3 — optional full decoder fine-tuning (unchanged from v3)
# ---------------------------------------------------------------------------
if [ "${RUN_STAGE3}" = "1" ]; then
  echo ""
  echo "WARNING: Only run Stage 3 if:"
  echo "  - Stage 2 val/train Dice gap < 5%"
  echo "  - eval/precision@0.5 has moved above baseline"
  run_stage \
    "Stage 3 (full decoder fine-tuning) [v3+ES]" \
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
echo "Staged fine-tuning [v3 + early stopping] complete."
echo "Stage 1  : ${STAGE1_DIR}"
echo "Stage 2  : ${STAGE2_DIR}"
if [ "${RUN_STAGE3}" = "1" ]; then
  echo "Stage 3  : ${STAGE3_DIR}"
fi
echo ""
echo "Stage 2 early stopping log: ${ES_LOG_FILE}"
echo "============================================================"