# Text-Prompted Image Segmentation Evaluation Pipeline

A unified evaluation framework for text-prompted medical image segmentation models.

## Supported Models

1. **SAM3** - Meta's Segment Anything Model 3
2. **BiomedParse** - Biomedical segmentation with text prompts
3. **MediSee** - Medical image reasoning and segmentation
4. **DualProtoSeg** - Weakly supervised histopathology segmentation with CONCH

## Quick Start

### Setup

All models use the unified `dualprotoseg` conda environment:

```bash
conda activate dualprotoseg
```

### Run Single Model

```bash
# Evaluate specific model
./run.sh sam3
./run.sh biomedparse
./run.sh medisee
./run.sh dualprotoseg
```

### Run All Models

```bash
# Evaluate all 4 models sequentially
./run.sh --all
```

### Compare Models

```bash
# Compare all 4 models side-by-side
./run.sh --compare
```

### List Available Models

```bash
./run.sh --list
```

## Configuration

Edit `run.sh` to configure:

### Data Paths
```bash
IMAGES_DIR="/home/roba/miccai26/test_data/test_examples/colon_example"
MASKS_DIR="/home/roba/miccai26/test_data/test_examples/colon_example/masks"
PROMPTS_JSON="/home/roba/miccai26/test_data/test_examples/instructions/GlaS_gpt-5.1_0_original.json"
OUTPUT_DIR="/home/roba/miccai26/results/segmentations_allDim"
```

### Model Paths
```bash
BIOMEDPARSE_PATH="/home/roba/miccai26/BiomedParse"
SAM3_PATH=""  # Uses default installation
MEDISEE_PATH="/home/roba/miccai26/MediSee"
MEDISEE_MODEL_DIR="/home/roba/miccai26/models/medisee"
DUALPROTOSEG_PATH="/home/roba/miccai26/DualProtoSeg"
```

### DualProtoSeg Specific
```bash
DUALPROTOSEG_CHECKPOINT="/path/to/checkpoint/best_cam.pth"
DATASET="bcss"  # Options: colon, bcss, liver, prostate
PROMPTS_CONFIG="/home/roba/miccai26/configs/class_prompts.yaml"
```

**Important**: For DualProtoSeg, the `DATASET` must match what the checkpoint was trained on to avoid dimension mismatches.

## File Structure

```
/home/roba/miccai26/
├── run.sh                          # Main evaluation script
├── text_seg_eval_example.py        # Evaluation framework
├── configs/
│   └── class_prompts.yaml         # Class prompts for DualProtoSeg
├── wrappers/
│   ├── sam3_wrapper.py
│   ├── biomedparse_wrapper.py
│   ├── medisee_wrapper.py
│   └── dualprotoseg_wrapper.py
├── BiomedParse/                   # Model repositories
├── MediSee/
├── DualProtoSeg/
├── models/                        # Model weights
│   ├── medisee/
│   └── ...
└── results/                       # Evaluation outputs
    └── segmentations_allDim/
```

## Metrics

The evaluation computes the following metrics:

- **IoU** (Intersection over Union)
- **Dice** coefficient
- **Precision**
- **Recall**
- **Accuracy**

Optional metrics:
- **Specificity**
- **Boundary IoU**
- **Hausdorff** distance

## Output

Each evaluation produces:

1. **JSON results** - Quantitative metrics summary
2. **Predictions** - Segmentation masks as PNG files
3. **Console output** - Real-time progress and statistics

Example:
```
results/segmentations_allDim/
├── sam3_eval_results.json
├── sam3_predictions/
│   ├── image1_pred.png
│   ├── image2_pred.png
│   └── ...
├── biomedparse_eval_results.json
├── biomedparse_predictions/
└── ...
```

## Running on GPU

For GPU execution (required for actual inference):

```bash
# Interactive session
srun --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash
cd /home/roba/miccai26
./run.sh sam3

# Or direct execution
srun --gres=gpu:1 --mem=32G ./run.sh --all
```

## Model-Specific Notes

### SAM3
- Requires: `sam3` package installed
- Dependencies: `einops`, `decord`, `pycocotools`, `psutil`

### BiomedParse
- Requires: `detectron2`, `mpi4py`, `kornia`
- Path: `/home/roba/miccai26/BiomedParse`

### MediSee
- Requires: `sentencepiece`, `protobuf`
- Model dir: `/home/roba/miccai26/models/medisee`
- Based on LLaVA-Med architecture

### DualProtoSeg
- Requires: CONCH model and class prompts configuration
- **Important**: Dataset must match checkpoint training dataset
- Default: BCSS dataset (4 classes: Tumor, Stroma, Inflammatory, Necrosis)
- Checkpoint: `/home/roba/miccai26/DualProtoSeg/runs/checkpoints/.../best_cam.pth`

## Troubleshooting

### "ModuleNotFoundError"
Ensure the correct conda environment is activated:
```bash
conda activate dualprotoseg
```

### "Found no NVIDIA driver"
You're running on a login node. Use `srun` to get a GPU node.

### DualProtoSeg dimension mismatch
Ensure `DATASET` in `run.sh` matches the checkpoint's training dataset.

### Missing model weights
Check paths in `run.sh` and ensure models are downloaded:
- MediSee: weights should be in `/home/roba/miccai26/models/medisee`
- DualProtoSeg: checkpoint path must be valid

## Advanced Usage

### Custom Dataset

1. For DualProtoSeg, add your dataset to `configs/class_prompts.yaml`:

```yaml
my_dataset:
  num_classes: 3
  class_names:
    - class1
    - class2
    - class3
  class_prompts:
    0:
      - "detailed description of class 1"
      - "another description of class 1"
      # Add 4-6 prompts per class
    1:
      - "description of class 2"
    2:
      - "description of class 3"
```

2. Update `DATASET="my_dataset"` in `run.sh`

### Custom Metrics

Edit `text_seg_eval_example.py` to add metrics:
```python
METRICS = "iou dice precision recall accuracy specificity boundary_iou hausdorff"
```

### Batch Processing

Process multiple datasets:
```bash
for dataset in dataset1 dataset2 dataset3; do
    IMAGES_DIR="/path/to/$dataset/images"
    MASKS_DIR="/path/to/$dataset/masks"
    ./run.sh --all
done
```

## Citation

If you use this evaluation framework, please cite the respective model papers:

- **SAM3**: Meta's Segment Anything Model
- **BiomedParse**: [Citation needed]
- **MediSee**: [Citation needed]
- **DualProtoSeg**: Weakly supervised histopathology segmentation

## License

See individual model repositories for licensing information.
