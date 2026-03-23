# Shared configuration for run.sh and eval.sh
# Update BIOMEDPARSE_CHECKPOINT here to use the same model in both scripts.

SCRIPT_DIR="${SCRIPT_DIR:-/home/roba/miccai26}"
BIOMEDPARSE_PATH="${SCRIPT_DIR}/BiomedParse"

# BiomedParse fine-tuned checkpoint (dir containing model_state_dict.pt, or path to .pt file).
# Empty = pretrained from HuggingFace.
BIOMEDPARSE_CHECKPOINT="${BIOMEDPARSE_CHECKPOINT:-${SCRIPT_DIR}/BiomedParse/output/histopath_full/biomed_seg_lang_v1.yaml_conf~/run_2/00006160/default}"
