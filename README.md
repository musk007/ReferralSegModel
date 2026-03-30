# BiomedParse Fine-Tuning & Evaluation for Histopathology

Fine-tuning [BiomedParse](https://github.com/microsoft/BiomedParse) on histopathology datasets (colon, lung, prostate, breast BCSS, breast cells) with text-prompted segmentation.

## Prerequisites

```bash
conda activate dualprotoseg
```

Cluster nodes: 8 x NVIDIA H200 (140 GB VRAM each), 128 CPUs.

## Repository Structure

```
/home/roba/miccai26/
├── runner_full.sh                          # SLURM script — LoRA fine-tuning
├── runner_nolora.sh                        # SLURM script — full-weight fine-tuning (no LoRA)
├── eval.sh                                 # Evaluation on histopath + BiomedParse official test sets
├── text_seg_eval_example.py                # Evaluation framework
├── wrappers/
│   └── biomedparse_wrapper.py              # Inference wrapper for eval
├── BiomedParse/                            # BiomedParse source (modified)
│   ├── configs/
│   │   └── biomed_seg_lang_v1.yaml         # Base model config
│   ├── biomed_seg_lang_v1_histopath_full.yaml    # Training config (LoRA)
│   ├── biomed_seg_lang_v1_histopath_nolora.yaml  # Override: no LoRA, selective freezing
│   ├── modeling/
│   │   ├── BaseModel.py                    # Model loading (auto-detects LoRA checkpoints)
│   │   └── utils/lora.py                   # LoRA adapter implementation
│   ├── trainer/
│   │   ├── default_trainer.py              # Training loop, per-epoch eval, W&B logging
│   │   ├── xdecoder_trainer.py             # Optimizer setup, param freezing
│   │   └── utils_trainer.py                # Checkpoint saving
│   ├── datasets/
│   │   ├── registration/
│   │   │   └── register_biomed_datasets.py # Dataset registration (histopath + official)
│   │   └── evaluation/
│   │       └── grounding_evaluation.py     # mIoU, mDice, cIoU, cDice, precision@k
│   └── output/                             # Training outputs (checkpoints, best_model)
├── biomedparse_datasets/                   # Dataset root
│   ├── colon/
│   │   ├── train/ train_mask/ train.json
│   │   ├── eval/  eval_mask/  eval.json
│   │   └── test/  test_mask/  test.json
│   ├── lung/          (same structure)
│   ├── prostate/      (same structure)
│   ├── breast_bcss/   (same structure)
│   ├── breast_cells/  (same structure)
│   └── biomedParse/BiomedParseData/        # Official BiomedParse datasets
├── test_data/instructions/test/            # Per-dataset instruction JSONs for eval
├── results/eval/                           # Evaluation results
└── logs/                                   # SLURM job logs
```

## Fine-Tuning

### Configuration

Training behaviour is controlled by YAML config files stacked in order (each overrides the previous):

1. `configs/biomed_seg_lang_v1.yaml` — base BiomedParse architecture config
2. `biomed_seg_lang_v1_histopath_full.yaml` — datasets, training params, LoRA settings
3. `biomed_seg_lang_v1_histopath_nolora.yaml` *(optional)* — disables LoRA, freezes backbone/pixel_decoder

Three knobs control what gets fine-tuned:

| Setting | Purpose |
|---|---|
| `SOLVER.LORA.ENABLED` | When `true`, wraps target `nn.Linear` layers with rank-R LoRA adapters. Only `lora_A`/`lora_B` are trainable. |
| `SOLVER.LORA.TARGET_MODULES` | List of module name substrings to apply LoRA to (e.g. `["lang_encoder", "predictor"]`). |
| `SOLVER.FIX_PARAM` | Freeze entire modules by name when LoRA is disabled (e.g. `backbone: true`). |
| `SOLVER.LR_MULTIPLIER` | Per-module learning rate scaling. Set to `0.0` to soft-freeze a module. |

### Option A: LoRA Fine-Tuning (`runner_full.sh`)

Trains LoRA adapters (rank 8) on `lang_encoder` + `predictor` + `sem_seg_head`. Backbone and pixel_decoder receive reduced LR but are not frozen.

```bash
sbatch runner_full.sh
```

Key settings in `biomed_seg_lang_v1_histopath_full.yaml`:

```yaml
SOLVER.LORA:
  ENABLED: true
  R: 8
  ALPHA: 16
  TARGET_MODULES: ["sem_seg_head", "lang_encoder", "predictor"]

SOLVER.FIX_PARAM: {}           # nothing frozen

SOLVER.LR_MULTIPLIER:
  backbone: 0.1                # 10% of BASE_LR
  pixel_decoder: 0.2           # 20% of BASE_LR
  predictor: 0.5               # 50% of BASE_LR
  lang_encoder: 1.0            # 100% of BASE_LR
```

### Option B: Full-Weight Fine-Tuning — No LoRA (`runner_nolora.sh`)

Directly fine-tunes all weights in `lang_encoder` + `predictor`. Backbone and pixel_decoder are completely frozen.

```bash
sbatch runner_nolora.sh
```

This stacks an additional override config `biomed_seg_lang_v1_histopath_nolora.yaml`:

```yaml
SOLVER.LORA:
  ENABLED: false

SOLVER.FIX_PARAM:
  backbone: true               # frozen
  pixel_decoder: true           # frozen

SOLVER.LR_MULTIPLIER:
  backbone: 0.0
  pixel_decoder: 0.0
  predictor: 1.0
  lang_encoder: 1.0
```

### Customising What Gets Fine-Tuned

To create a new training configuration, copy one of the existing YAML files and adjust the three knobs. Examples:

**LoRA on all four modules:**
```yaml
SOLVER.LORA:
  ENABLED: true
  TARGET_MODULES: ["backbone", "pixel_decoder", "predictor", "lang_encoder"]
SOLVER.FIX_PARAM: {}
```

**Full-weight fine-tuning on text encoder only:**
```yaml
SOLVER.LORA:
  ENABLED: false
SOLVER.FIX_PARAM:
  backbone: true
  pixel_decoder: true
  predictor: true
SOLVER.LR_MULTIPLIER:
  lang_encoder: 1.0
```

### Overriding Output Directory and Epochs

Environment variables override defaults in the runner scripts:

```bash
OUTPUT_DIR=/home/roba/miccai26/BiomedParse/output/my_experiment \
SOLVER.MAX_NUM_EPOCHS=50 \
sbatch runner_full.sh
```

### Single-Dataset Fine-Tuning

```bash
TRAIN_DATASET=colon sbatch runner_full.sh
TRAIN_DATASET=lung sbatch runner_nolora.sh
# Options: colon, lung, prostate, breast_bcss, breast_cells
```

### Monitoring

Training logs to W&B (project: `BiomedParseFineTune`) and to SLURM log files:

```bash
# Watch SLURM queue
watch squeue

# Tail training logs
tail -f /home/roba/miccai26/logs/finetune_full_<JOBID>.err
tail -f /home/roba/miccai26/logs/finetune_nolora_<JOBID>.err
```

Per-epoch eval metrics logged: `mIoU`, `mDice`, `cIoU`, `cDice`, `precision@0.5` (on 0–100 scale).

Best model is saved automatically based on `SOLVER.BEST_METRIC` (default: `mIoU`).

### Training Output Structure

```
output/histopath_full/biomed_seg_lang_v1.yaml_conf~/run_N/
├── 00000740/                   # Epoch checkpoint
│   └── default/
│       └── model_state_dict.pt
├── best_model/                 # Best eval-split checkpoint
│   ├── default/
│   │   └── model_state_dict.pt
│   └── best_meta.json          # { epoch, score, metric }
└── wandb/                      # W&B run data
```

## Evaluation

### Histopathology Test Sets

Evaluates on lung, colon, prostate, breast_bcss, breast_cells using instruction-based prompts:

```bash
HISTOPATH=1 bash eval.sh
```

### BiomedParse Official Datasets

Evaluates on ACDC, BreastUS, CAMUS, CDD-CESM, etc.:

```bash
BIOMEDPARSE_OFFICIAL=1 bash eval.sh
```

### Both Together

```bash
BIOMEDPARSE_OFFICIAL=1 HISTOPATH=1 bash eval.sh
```

### Single Dataset

```bash
DATASET=colon bash eval.sh
DATASET=lung bash eval.sh
```

### Specifying a Checkpoint

Edit `BIOMEDPARSE_CHECKPOINT` in `eval.sh` (line 75) to point to the desired checkpoint directory:

```bash
BIOMEDPARSE_CHECKPOINT="/home/roba/miccai26/BiomedParse/output/histopath_full/.../best_model"
```

The loader auto-detects LoRA checkpoints and applies LoRA wrappers before loading weights.

### Evaluation Metrics

All metrics are reported on a **0–100 percentage scale**:

| Metric | Description |
|---|---|
| `mIoU` | Mean Intersection over Union (per-sample, averaged) |
| `cIoU` | Cumulative IoU (global intersection / global union) |
| `mDice` | Mean Dice coefficient |
| `cDice` | Cumulative Dice coefficient |
| `precision@0.5` | Fraction of samples with IoU >= 0.5 |

### Evaluation Output

```
results/eval/<dataset>/biomedparse/
├── biomedparse_results.json    # Quantitative metrics
└── predictions/                # Predicted mask PNGs
```

## Troubleshooting

### NCCL Timeout During Training

Caused by rank divergence in distributed training. The training code includes barriers at epoch boundaries. If issues recur, check `trainer/default_trainer.py`.

### `$UNUSED$` Warnings When Loading Checkpoint

If you see `$UNUSED$ ... lora_A` warnings during evaluation, the LoRA wrappers were not applied before loading. This is handled automatically in `BaseModel.from_pretrained()` — ensure you're using the latest code.

### Image Loading Errors

The dataset registration code (`register_biomed_datasets.py`) tries alternate extensions (`.png` <-> `.jpg`) if the referenced file is not found. Check that image/mask files exist in the dataset directories.

### HuggingFace Timeout on Worker Nodes

Worker nodes may lack internet access. The runner scripts set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to use cached models. Ensure the cache is populated on a login node first.
