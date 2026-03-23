#!/bin/bash
# =============================================================================
# SLURM job script – Text-Prompted Segmentation Evaluation
#
# Evaluates one model on one dataset. Override MODEL and DATASET via
# environment variables before submitting, e.g.:
#
#   sbatch eval.sh                                  # biomedparse on lung (default)
#   MODEL=medisee DATASET=colon sbatch eval.sh
#   MODEL=biomedparse DATASET=breast_bcss sbatch eval.sh
#
# Supported models  : biomedparse, medisee, sam3, dualprotoseg
# Supported datasets: lung, colon, prostate, breast_bcss, breast_cells
#
# To evaluate on BiomedParse official dataset (ACDC, DRIVE, ISIC, etc.):
#   BIOMEDPARSE_OFFICIAL=1 sbatch eval.sh           # all official test sets
#   BIOMEDPARSE_OFFICIAL=1 BIOMEDPARSE_OFFICIAL_DATASETS=ACDC,DRIVE sbatch eval.sh
#
# To compare all models on a single dataset:
#   COMPARE=1 DATASET=lung sbatch eval.sh
#
# To run all (model × dataset) combinations:
#   ALL=1 sbatch eval.sh
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM directives
# ---------------------------------------------------------------------------
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:1        # 1 GPU is enough for inference
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --job-name=bp_eval
#SBATCH --output=/home/roba/miccai26/logs/eval_%j.out
#SBATCH --error=/home/roba/miccai26/logs/eval_%j.err

set -e

# ---------------------------------------------------------------------------
# Configurable parameters  (override via environment or --export)
# ---------------------------------------------------------------------------
MODEL="${MODEL:-biomedparse}"
DATASET="${DATASET:-lung}"
DEVICE="${DEVICE:-cuda}"
THRESHOLD="${THRESHOLD:-0.5}"
METRICS="${METRICS:-iou dice precision recall accuracy giou ciou}"
COMPARE="${COMPARE:-0}"   # set to 1 to compare all models on DATASET
ALL="${ALL:-0}"           # set to 1 to run every model × dataset combination
# Evaluate on BiomedParse official dataset (biomedparse_datasets/biomedParse/BiomedParseData)
BIOMEDPARSE_OFFICIAL="${BIOMEDPARSE_OFFICIAL:-0}"
BIOMEDPARSE_OFFICIAL_DATASETS="${BIOMEDPARSE_OFFICIAL_DATASETS:-all}"  # all | ACDC,DRIVE,ISIC,...

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="/home/roba/miccai26"
[ -f "${SCRIPT_DIR}/config.sh" ] && . "${SCRIPT_DIR}/config.sh"
BIOMEDPARSE_PATH="${SCRIPT_DIR}/BiomedParse"
MEDISEE_PATH="${SCRIPT_DIR}/MediSee"
MEDISEE_MODEL_DIR="${SCRIPT_DIR}/models/medisee"
DUALPROTOSEG_PATH="${SCRIPT_DIR}/DualProtoSeg"
DUALPROTOSEG_CHECKPOINT="${DUALPROTOSEG_PATH}/runs/checkpoints/2026-01-31-06-14(BaselineRun)/best_cam.pth"
PROMPTS_CONFIG="${SCRIPT_DIR}/configs/class_prompts.yaml"
DUALPROTOSEG_DATASET="bcss"   # checkpoint was trained on BCSS
# BiomedParse checkpoint from config.sh (shared with run.sh; override via env)

# Histopath-style datasets (lung, colon, etc.) use contoured_instructions.json from test_data
# BiomedParse official (ACDC, DRIVE, etc.) use BIOMEDPARSE_OFFICIAL=1
BASE_DATA="${SCRIPT_DIR}/test_data/data_test"
BASE_RESULTS="${SCRIPT_DIR}/results/eval"
BIOMEDPARSE_OFFICIAL_DATA="${SCRIPT_DIR}/biomedparse_datasets/biomedParse/BiomedParseData"
BIOMEDPARSE_CHECKPOINT="/home/roba/miccai26/BiomedParse/output/histopath_full/biomed_seg_lang_v1.yaml_conf~/run_2/00006160"
# ---------------------------------------------------------------------------
# Dataset path table
# Each dataset entry: IMAGES_DIR  MASKS_DIR  PROMPTS_JSON
# ---------------------------------------------------------------------------
resolve_dataset() {
    local ds="$1"
    case "$ds" in
        lung)
            IMAGES_DIR="${BASE_DATA}/lung/all_images"
            MASKS_DIR="${BASE_DATA}/lung/all_masks"
            PROMPTS_JSON="${BASE_DATA}/lung/contoured_instructions.json"
            ;;
        colon)
            IMAGES_DIR="${BASE_DATA}/colon/all_images"
            MASKS_DIR="${BASE_DATA}/colon/all_masks"
            PROMPTS_JSON="${BASE_DATA}/colon/contoured_instructions.json"
            ;;
        prostate)
            IMAGES_DIR="${BASE_DATA}/prostate/all_images"
            MASKS_DIR="${BASE_DATA}/prostate/labeled_masks"
            PROMPTS_JSON="${BASE_DATA}/prostate/contoured_instructions.json"
            ;;
        breast_bcss)
            IMAGES_DIR="${BASE_DATA}/breast/bcss/all_images"
            MASKS_DIR="${BASE_DATA}/breast/bcss/all_masks"
            PROMPTS_JSON="${BASE_DATA}/breast/bcss/contoured_instructions.json"
            ;;
        breast_cells)
            IMAGES_DIR="${BASE_DATA}/breast/cells/all_images"
            MASKS_DIR="${BASE_DATA}/breast/cells/all_masks"
            PROMPTS_JSON="${BASE_DATA}/breast/cells/contoured_instructions.json"
            ;;
        *)
            echo "ERROR: Unknown dataset '${ds}'. Choose: lung, colon, prostate, breast_bcss, breast_cells"
            exit 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Model-specific CLI arguments
# ---------------------------------------------------------------------------
model_args() {
    local m="$1"
    case "$m" in
        biomedparse)
            echo "--biomedparse_path ${BIOMEDPARSE_PATH}" ${BIOMEDPARSE_CHECKPOINT:+--checkpoint "${BIOMEDPARSE_CHECKPOINT}"}
            ;;
        medisee)
            echo "--medisee_path ${MEDISEE_PATH} --medisee_model_dir ${MEDISEE_MODEL_DIR}"
            ;;
        dualprotoseg)
            echo "--dualprotoseg_path ${DUALPROTOSEG_PATH} --checkpoint ${DUALPROTOSEG_CHECKPOINT} --dataset ${DUALPROTOSEG_DATASET} --prompts_config ${PROMPTS_CONFIG}"
            ;;
        sam3)
            echo ""
            ;;
        *)
            echo ""
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source /home/roba/miniconda3/bin/activate
conda activate dualprotoseg

export OMPI_MCA_orte_tmpdir_base="/tmp"
export PMIX_MCA_gds="^ds12,ds21"
export OMPI_MCA_btl="^openib"
export OMPI_MCA_pml="ob1"

mkdir -p /home/roba/miccai26/logs

# ---------------------------------------------------------------------------
# Helper: run evaluation for one (model, dataset) pair
# ---------------------------------------------------------------------------
run_eval() {
    local m="$1"
    local ds="$2"

    resolve_dataset "$ds"

    local out_dir="${BASE_RESULTS}/${ds}/${m}"
    mkdir -p "${out_dir}"

    echo "------------------------------------------------------------"
    echo "Model  : ${m}"
    echo "Dataset: ${ds}"
    echo "Images : ${IMAGES_DIR}"
    echo "Masks  : ${MASKS_DIR}"
    echo "Output : ${out_dir}"
    echo "------------------------------------------------------------"

    # shellcheck disable=SC2046
    python "${SCRIPT_DIR}/text_seg_eval_example.py" \
        --model "${m}" \
        --images_dir "${IMAGES_DIR}" \
        --masks_dir  "${MASKS_DIR}" \
        --prompts_json "${PROMPTS_JSON}" \
        --device "${DEVICE}" \
        --threshold "${THRESHOLD}" \
        --metrics ${METRICS} \
        $(model_args "${m}") \
        --output "${out_dir}/${m}_results.json" \
        --save_predictions "${out_dir}/predictions"

    echo "Saved → ${out_dir}/${m}_results.json"
    echo ""
}

# ---------------------------------------------------------------------------
# Helper: compare all models on one dataset
# ---------------------------------------------------------------------------
run_compare() {
    local ds="$1"

    resolve_dataset "$ds"

    local out_dir="${BASE_RESULTS}/${ds}/comparison"
    mkdir -p "${out_dir}"

    echo "------------------------------------------------------------"
    echo "Compare: biomedparse sam3 medisee dualprotoseg"
    echo "Dataset: ${ds}"
    echo "Output : ${out_dir}"
    echo "------------------------------------------------------------"

    python "${SCRIPT_DIR}/text_seg_eval_example.py" \
        --compare biomedparse sam3 medisee dualprotoseg \
        --images_dir "${IMAGES_DIR}" \
        --masks_dir  "${MASKS_DIR}" \
        --prompts_json "${PROMPTS_JSON}" \
        --device "${DEVICE}" \
        --metrics ${METRICS} \
        --output "${out_dir}/comparison_results.json" \
        --save_predictions "${out_dir}/predictions" \
        --biomedparse_path "${BIOMEDPARSE_PATH}" \
        --medisee_path "${MEDISEE_PATH}" \
        --medisee_model_dir "${MEDISEE_MODEL_DIR}" \
        --dualprotoseg_path "${DUALPROTOSEG_PATH}" \
        --checkpoint "${DUALPROTOSEG_CHECKPOINT}" \
        --dataset "${DUALPROTOSEG_DATASET}" \
        --prompts_config "${PROMPTS_CONFIG}" \
        ${BIOMEDPARSE_CHECKPOINT:+--biomedparse_checkpoint "${BIOMEDPARSE_CHECKPOINT}"}

    echo "Saved → ${out_dir}/comparison_results.json"
}

# ---------------------------------------------------------------------------
# BiomedParse official dataset evaluation
# Uses text_seg_eval_example.py (same as run.sh) with COCO-format BiomedParseData
# Evaluates each test set separately and saves per-dataset results.
# ---------------------------------------------------------------------------
run_biomedparse_official() {
    if [ ! -d "${BIOMEDPARSE_OFFICIAL_DATA}" ]; then
        echo "ERROR: BiomedParse official data not found at ${BIOMEDPARSE_OFFICIAL_DATA}"
        exit 1
    fi

    if [ -z "${BIOMEDPARSE_CHECKPOINT}" ]; then
        echo "ERROR: BIOMEDPARSE_CHECKPOINT must be set for official dataset evaluation"
        exit 1
    fi

    local out_dir="${BASE_RESULTS}/biomedparse_official/biomedparse"
    mkdir -p "${out_dir}"

    # Datasets: all registered or comma-separated list
    local all_names="ACDC BreastUS CDD-CESM DRIVE G1020 ISIC LGG OCT-CME PanNuke UWaterlooSkinCancer"
    local datasets_to_run
    if [ "${BIOMEDPARSE_OFFICIAL_DATASETS}" = "all" ]; then
        datasets_to_run="${all_names}"
    else
        datasets_to_run="$(echo "${BIOMEDPARSE_OFFICIAL_DATASETS}" | tr ',' ' ')"
    fi

    echo "------------------------------------------------------------"
    echo "BiomedParse evaluation on official dataset(s) (per test set)"
    echo "Data      : ${BIOMEDPARSE_OFFICIAL_DATA}"
    echo "Checkpoint: ${BIOMEDPARSE_CHECKPOINT}"
    echo "Datasets  : ${datasets_to_run}"
    echo "Output    : ${out_dir}"
    echo "------------------------------------------------------------"

    for ds_name in ${datasets_to_run}; do
        ds_name=$(echo "$ds_name" | xargs)
        [ -z "$ds_name" ] && continue

        local json_path="${BIOMEDPARSE_OFFICIAL_DATA}/${ds_name}/test.json"
        if [ ! -f "${json_path}" ]; then
            echo "  [SKIP] ${ds_name}: test.json not found"
            continue
        fi

        echo ""
        echo ">>> Evaluating on ${ds_name}..."
        cd "${SCRIPT_DIR}"
        python "${SCRIPT_DIR}/text_seg_eval_example.py" \
            --model biomedparse \
            --biomedparse_path "${BIOMEDPARSE_PATH}" \
            --checkpoint "${BIOMEDPARSE_CHECKPOINT}" \
            --biomedparse_official_data "${BIOMEDPARSE_OFFICIAL_DATA}" \
            --biomedparse_official_datasets "${ds_name}" \
            --device "${DEVICE}" \
            --threshold "${THRESHOLD}" \
            --metrics ${METRICS} \
            --output "${out_dir}/${ds_name}_results.json" \
            --save_predictions "${out_dir}/predictions/${ds_name}" \
            || echo "  [FAIL] ${ds_name}"
    done

    echo ""
    echo "Per-dataset results saved to ${out_dir}/*_results.json"
    echo ""
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Text-Prompted Segmentation Evaluation"
echo "Job ID  : ${SLURM_JOB_ID:-local}"
echo "Node    : $(hostname)"
echo "Metrics : ${METRICS}"
echo "============================================================"
echo ""

ALL_DATASETS="lung colon prostate breast_bcss breast_cells"
ALL_MODELS="biomedparse sam3 medisee dualprotoseg"

if [ "${BIOMEDPARSE_OFFICIAL}" = "1" ]; then
    run_biomedparse_official

elif [ "${ALL}" = "1" ]; then
    echo "Running all model × dataset combinations..."
    echo ""
    for ds in ${ALL_DATASETS}; do
        for m in ${ALL_MODELS}; do
            run_eval "${m}" "${ds}" || echo "  [SKIP] ${m} on ${ds} failed, continuing..."
        done
    done

elif [ "${COMPARE}" = "1" ]; then
    run_compare "${DATASET}"

else
    run_eval "${MODEL}" "${DATASET}"
fi

echo "============================================================"
echo "Evaluation complete."
echo "============================================================"
