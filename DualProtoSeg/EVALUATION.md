# DualProtoSeg Evaluation

This directory contains scripts for evaluating trained DualProtoSeg models.

## Quick Start

### Using the Shell Script (Easiest)

```bash
# Activate the environment
conda activate dualprotoseg

# Evaluate on test set
bash eval.sh test

# Evaluate on validation set
bash eval.sh valid

# Evaluate with CAM generation
bash eval.sh test --generate-cams
```

### Using Python Directly

```bash
conda activate dualprotoseg

python evaluate.py \
    --checkpoint /home/roba/miccai26/DualProtoSeg/runs/checkpoints/2026-01-31-06-14/best_cam.pth \
    --config config.yaml \
    --split test \
    --gpu 0
```

## Command-Line Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--checkpoint` | Path to model checkpoint (.pth file) | - | ✓ |
| `--config` | Path to config file | `config.yaml` | ✗ |
| `--split` | Dataset split (`test`, `valid`, `val`) | `test` | ✗ |
| `--gpu` | GPU ID to use | `0` | ✗ |
| `--batch_size` | Batch size for evaluation | From config | ✗ |
| `--output_dir` | Output directory for results | Next to checkpoint | ✗ |
| `--generate_cams` | Generate CAM visualizations | False | ✗ |

## Output Files

The evaluation script creates the following outputs:

### Default Location
`runs/checkpoints/<timestamp>/eval_<split>/`

### Files Generated
- `evaluation_results.txt` - Detailed metrics (IoU, Dice, accuracy, AUC)
- `cams/*.png` - CAM visualizations (if `--generate_cams` is used)

## Metrics Computed

### Classification Metrics
- **Loss**: BCE loss on image-level predictions
- **Accuracy**: Percentage of correctly classified images
- **AUC**: Area under ROC curve (macro-averaged)

### Segmentation Metrics
- **mIoU**: Mean Intersection over Union
- **Mean Dice**: Average Dice coefficient across classes
- **FwIU**: Frequency-weighted IoU

### Per-Class Metrics
- IoU and Dice scores for each class:
  - Tumor
  - Stroma
  - Inflammatory
  - Necrosis
  - Background

## Example Output

```
================================================================================
EVALUATION RESULTS
================================================================================

                         CLASSIFICATION METRICS                         
--------------------------------------------------------------------------------
  Loss:     0.2345
  Accuracy: 87.50%
  AUC:      0.9234

                          SEGMENTATION METRICS                          
--------------------------------------------------------------------------------
  mIoU:       0.6543
  Mean Dice:  0.7234
  FwIU:       0.6789

                           PER-CLASS METRICS                            
--------------------------------------------------------------------------------
  Class           IoU        Dice      
  -----------------------------------
  Tumor           0.7234    0.8123
  Stroma          0.6543    0.7456
  Inflammatory    0.5678    0.6789
  Necrosis        0.6012    0.7012
  Background      0.7890    0.8567
```

## Troubleshooting

### Error: Checkpoint not found
Make sure the checkpoint path in `eval.sh` points to your actual checkpoint file.

### Error: CUDA out of memory
Reduce the batch size:
```bash
python evaluate.py --checkpoint <path> --batch_size 16
```

### Error: Dataset not found
Ensure the data paths in `config.yaml` are correct:
- `dataset.train_root`
- `dataset.val_root`

## Notes

- The script automatically loads the CONCH text encoder with the class prompts used during training
- Evaluation is performed with the same preprocessing as training
- CAM generation can be memory-intensive; consider using `batch_size=1` if needed
- Results are saved automatically; no need to redirect output
