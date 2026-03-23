# Unified Segmentation Model Evaluation System

A modular, extensible framework for evaluating multiple text-prompted segmentation models using a unified interface.

## Overview

This system provides:
- **Unified API** for different segmentation models
- **Automatic model registration** and discovery
- **Command-line interface** for easy evaluation
- **Model comparison** capabilities
- **Standardized metrics** (IoU, Dice, Precision, Recall, etc.)
- **Flexible dataset formats**

## Quick Start

### 1. Installation

```bash
# Clone your evaluation repository
git clone <your-repo>
cd <your-repo>

# Install base dependencies
pip install numpy pillow scipy torch torchvision

# Install text_seg_eval framework
# (copy text_seg_eval.py to your directory)
```

### 2. Install Model Repositories

Install the models you want to evaluate:

```bash
# BiomedParse (v1 for 2D, v2 for 3D)
git clone https://github.com/microsoft/BiomedParse.git
cd BiomedParse && git checkout v1  # For histopathology
pip install -r assets/requirements/requirements.txt

# SAM3
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .

# DualProtoSeg
git clone https://github.com/maianhpuco/DualProtoSeg.git
cd DualProtoSeg && pip install -r requirements.txt

# MediSee 
git clone https://github.com/Edisonhimself/MediSee
cd MediSee
conda env update -n <env_name> -f environment.yml

# SAT : Segment Anything with Text
git clone https://github.com/zhaoziheng/SAT
cd SAT/model
pip install -e dynamic-network-architectures-main



# Add others as needed...
```

### 3. List Available Models

```bash
python evaluate_models.py --list
```

### 4. Run Evaluation

```bash
python evaluate_models.py \
    --model biomedparse_v1 \
    --images_dir /path/to/images \
    --masks_dir /path/to/masks \
    --prompts_json /path/to/prompts.json \
    --output results.json
```

## Supported Models

### Currently Implemented

| Model | Status | Modalities | Best For |
|-------|--------|------------|----------|
| BiomedParse v1 | ✅ Complete | CT, MRI, X-Ray, Pathology, etc. | Multi-modal 2D segmentation |
| BiomedParse v2 | ✅ Complete | CT, MRI, PET, 3D Microscopy | 3D volumetric segmentation |
| SAM3 | ✅ Complete | General, Pathology, Radiology | Text and visual prompts |
| DualProtoSeg | ✅ Complete | Pathology | Histopathology weakly supervised |

### To Be Implemented

| Model | Status | Priority | Notes |
|-------|--------|----------|-------|
| MediSee | 📝 Template | High | Reasoning-based queries |
| TextDiff | 📝 Template | Medium | Label-efficient learning |
| TissueLab | 📝 Template | Medium | Agentic AI system |
| SAT | 📝 Template | Low | 3D radiology only (not for histopathology) |

## File Structure

```
.
├── text_seg_eval.py              # Core evaluation framework
├── model_registry.py             # Model registration system
├── evaluate_models.py            # Unified CLI tool
│
├── biomedparse_wrapper.py        # ✅ BiomedParse wrappers (v1 & v2)
├── sam3_wrapper.py               # ✅ SAM3 wrapper
├── dualprotoseg_wrapper.py       # ✅ DualProtoSeg wrapper
│
├── wrapper_templates.py          # 📝 Templates for remaining models
│
├── BIOMEDPARSE_README.md         # BiomedParse specific docs
└── README_MODELS.md              # This file
```

## Usage Examples

### Example 1: Evaluate Single Model

```bash
python evaluate_models.py \
    --model biomedparse_v1 \
    --images_dir ./histopathology/images \
    --masks_dir ./histopathology/masks \
    --prompts_json ./histopathology/prompts.json \
    --device cuda \
    --threshold 0.5 \
    --metrics iou dice precision recall \
    --output biomedparse_results.json
```

### Example 2: Compare Multiple Models

```bash
python evaluate_models.py \
    --compare biomedparse_v1 sam3 dualprotoseg \
    --images_dir ./histopathology/images \
    --masks_dir ./histopathology/masks \
    --prompts_json ./histopathology/prompts.json \
    --output model_comparison.json
```

### Example 3: Filter by Modality

```bash
# List only pathology models
python evaluate_models.py --list --modality pathology

# Output:
# Models supporting pathology:
#   biomedparse_v1
#   sam3
#   dualprotoseg
#   medisee
#   textdiff
```

### Example 4: Model-Specific Arguments

```bash
python evaluate_models.py \
    --model biomedparse_v1 \
    --images_dir ./images \
    --masks_dir ./masks \
    --prompts_json ./prompts.json \
    --model_args "biomedparse_path=/path/to/BiomedParse,threshold=0.6"
```

## Dataset Format

### JSON Format 1: Simple List

```json
[
    {
        "id": "sample_001",
        "image": "img001.png",
        "mask": "mask001.png",
        "prompt": "tumor cells"
    },
    {
        "id": "sample_002",
        "image": "img002.png",
        "mask": "mask002.png",
        "prompt": "inflammatory cells"
    }
]
```

### JSON Format 2: With Regions

```json
{
    "slide_001": {
        "regions": {
            "region_001": {
                "gpt": {
                    "diagnosis": "adenocarcinoma"
                }
            },
            "region_002": {
                "gpt": {
                    "diagnosis": "normal glands"
                }
            }
        }
    }
}
```

## Implementing New Model Wrappers

### Step 1: Create Wrapper File

Create `{model_name}_wrapper.py`:

```python
from text_seg_eval import TextSegmentationModel
import numpy as np
from PIL import Image
from typing import Tuple, Optional

class YourModelWrapper(TextSegmentationModel):
    def __init__(self, checkpoint_path, device="cuda", threshold=0.5):
        self.device = device
        self.threshold = threshold
        # Load your model
        self.model = load_model(checkpoint_path)
        self.model = self.model.to(device).eval()
    
    def segment(
        self, 
        image: Image.Image, 
        text_prompt: str
    ) -> Tuple[np.ndarray, Optional[float]]:
        # 1. Preprocess image
        # 2. Encode text
        # 3. Run inference
        # 4. Post-process mask
        # 5. Return binary mask (H, W) and confidence
        
        mask = your_segmentation_logic(image, text_prompt)
        confidence = calculate_confidence(mask)
        
        return mask.astype(np.uint8), confidence
```

### Step 2: Register Model

The model is automatically registered if the wrapper is importable. Alternatively, manually register:

```python
from model_registry import ModelRegistry
from your_wrapper import YourModelWrapper

ModelRegistry.register(
    name="your_model",
    wrapper_class=YourModelWrapper,
    description="Description of your model",
    modalities=["Pathology", "Radiology"],
    paper_url="https://arxiv.org/...",
    github_url="https://github.com/..."
)
```

### Step 3: Test

```python
from model_registry import get_model

model = get_model("your_model", checkpoint_path="path/to/checkpoint.pth")
mask, conf = model.segment(test_image, "test prompt")

assert mask.shape == (test_image.size[1], test_image.size[0])
assert mask.dtype == np.uint8
assert 0 <= conf <= 1
```

## Implementation Priority for Histopathology

Based on your histopathology use case, here's the recommended implementation order:

1. ✅ **BiomedParse v1** - Multi-modality foundation model
2. ✅ **SAM3** - Versatile text-prompted segmentation
3. ✅ **DualProtoSeg** - Histopathology-specific
4. 🔄 **MediSee** - For complex reasoning queries
5. 🔄 **TextDiff** - For few-shot scenarios
6. ⏭️ **TissueLab** - Agentic AI platform (complex integration)
7. ❌ **SAT** - Skip (designed for 3D radiology, not 2D histopathology)

## Metrics

Available metrics (from `text_seg_eval.py`):
- **iou**: Intersection over Union (Jaccard Index)
- **dice**: Dice Coefficient (F1 Score)
- **precision**: Pixel-wise precision
- **recall**: Pixel-wise recall (sensitivity)
- **specificity**: Pixel-wise specificity
- **accuracy**: Pixel-wise accuracy
- **boundary_iou**: Boundary IoU (requires scipy)
- **hausdorff**: Hausdorff Distance (requires scipy)

## Troubleshooting

### Model Not Found

```bash
python evaluate_models.py --list
# Check if your model is registered
```

### Import Errors

Make sure the model repository is installed and in your Python path:

```python
import sys
sys.path.insert(0, "/path/to/model/repo")
```

Or use the wrapper's path parameter:

```bash
python evaluate_models.py \
    --model biomedparse_v1 \
    --model_args "biomedparse_path=/path/to/BiomedParse" \
    ...
```

### CUDA Out of Memory

```bash
# Use CPU
python evaluate_models.py --device cpu ...

# Or reduce batch size (model-specific)
--model_args "batch_size=1"
```

## Advanced Usage

### Programmatic API

```python
from text_seg_eval import evaluate_model
from model_registry import get_model

# Load model
model = get_model(
    "biomedparse_v1",
    device="cuda",
    threshold=0.5
)

# Prepare dataset
dataset = [
    {
        "id": "sample_001",
        "image_path": "path/to/image.png",
        "mask_path": "path/to/mask.png",
        "prompt": "tumor cells"
    },
    # ... more samples
]

# Evaluate
results = evaluate_model(
    model=model,
    dataset=dataset,
    metrics=["iou", "dice", "precision", "recall"],
    verbose=True
)

# Print results
results.print_summary()

# Save to JSON
results.save_json("results.json")
```

### Batch Processing

```python
from model_registry import get_model
from PIL import Image

model = get_model("biomedparse_v1")

images = [Image.open(f"img_{i}.png") for i in range(10)]
prompts = ["tumor"] * 10

# Process in batch (if model supports it)
if hasattr(model, 'batch_segment'):
    results = model.batch_segment(images, prompts)
else:
    results = [model.segment(img, prompt) 
               for img, prompt in zip(images, prompts)]
```

## Contributing

To add a new model wrapper:

1. Create wrapper file following the template
2. Implement the `segment()` method
3. Test with dummy data
4. Add to model registry
5. Update this README

See `wrapper_templates.py` for detailed templates and implementation notes.

## Citation

If you use this evaluation framework, please cite the original models:

```bibtex
@article{zhao2025biomedparse,
  title={A foundation model for joint segmentation, detection and recognition of biomedical objects across nine modalities},
  author={Zhao, Theodore and ...},
  journal={Nature methods},
  year={2025}
}

@article{carion2025sam3,
  title={SAM 3: Segment Anything with Concepts},
  author={Carion, Nicolas and ...},
  journal={arXiv preprint arXiv:2511.16719},
  year={2025}
}

% Add citations for other models as used
```

## License

This evaluation framework is provided for research purposes. Individual models may have their own licenses - please check each model's repository for license information.

##Contacts

For questions or issues:
- Open an issue in this repository
- Check individual model repositories for model-specific questions
