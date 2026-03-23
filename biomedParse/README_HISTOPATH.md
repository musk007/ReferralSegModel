# Histopathology Referral Segmentation — BiomedParse v1 Integration

## Overview

This module integrates four multi-dimensional histopathology segmentation datasets
(colon, BCSS breast, lung, prostate) into the [BiomedParse v1](https://github.com/microsoft/BiomedParse/tree/main)
fine-tuning pipeline. Each region is described by up to four reasoning dimensions:

| Dimension | Key in JSON | Description |
|---|---|---|
| Histological | `Histological Reasoning` | Morphological/cytological features |
| Spatial | `Spatial Reasoning` | Location relative to landmarks |
| Hierarchical | `Hierarchical Reasoning` | Rank within the image (largest, most complex) |
| Ambiguity | `Ambiguity Reasoning` | Disambiguation from similar neighbours |

---

## Files Added / Modified

```
BiomedParse/
├── biomedparse_datasets/
│   ├── convert_histopath_to_biomedparse.py    ← NEW: converts your JSON to BiomedParse format
│   └── run_ablation_conversion.py             ← NEW: batch-generates dimension-specific JSONs
├── datasets/
│   ├── registration/
│   │   └── register_biomed_datasets.py        ← MODIFIED: adds HistoPath* dataset registration
│   └── dataset_mappers/
│       └── biomed_dataset_mapper.py           ← MODIFIED: multi-dimensional prompt support
└── configs/
    └── biomed_seg_lang_v1_histopath.yaml      ← NEW: training config for custom data
```

---

## Step-by-Step Setup

### 1. Prepare image and mask files

BiomedParse requires **1024×1024 PNG** images and binary masks.

For each dataset, create:
```
biomedparse_datasets/
    HistoPathColon/
        train/         ← images named  <image_id>_pathology_colon.png
        train_mask/    ← masks  named  <image_id>_pathology_colon_<target>.png
        test/
        test_mask/
    HistoPathBCSS/
        train/         ← images named  <image_id>_pathology_breast.png
        train_mask/    ← masks  named  <image_id>_pathology_breast_<target>.png
        test/
        test_mask/
    HistoPathLung/
        train/         ← images named  <image_id>_pathology_lung.png
        ...
    HistoPathProstate/
        train/         ← images named  <image_id>_pathology_prostate.png
        ...
```

> **Target name convention:** spaces → `+`, underscores → `+`, no special chars.
> Example: `region class "invasive tumor"` → target `invasive+tumor`

---

### 2. Convert annotation JSONs to BiomedParse format

#### Option A: Full multi-dimensional JSON (recommended for training)

```bash
cd biomedparse_datasets

# Colon
python convert_histopath_to_biomedparse.py \
    --input   /path/to/colon_contoured_gpt-5_1_0_original.json \
    --dataset_type colon \
    --split   train \
    --output_json HistoPathColon/train.json \
    --prompt_mode all          # stores all dimension sentences per entry

# BCSS breast
python convert_histopath_to_biomedparse.py \
    --input   /path/to/bcss_contoured_gpt-5_1_0_original.json \
    --dataset_type bcss \
    --split   train \
    --output_json HistoPathBCSS/train.json \
    --prompt_mode all

# Lung
python convert_histopath_to_biomedparse.py \
    --input   /path/to/lung_contoured_terminology_gpt-5_1_0_original.json \
    --dataset_type lung \
    --split   train \
    --output_json HistoPathLung/train.json \
    --prompt_mode all

# Prostate
python convert_histopath_to_biomedparse.py \
    --input   /path/to/prostate_contoured_gpt-5_1_0_original.json \
    --dataset_type prostate \
    --split   train \
    --output_json HistoPathProstate/train.json \
    --prompt_mode all

# Repeat for --split test
```

#### Option B: Ablation JSONs (one per dimension, generated automatically)

```bash
python run_ablation_conversion.py \
    --colon_json    /path/to/colon_contoured_gpt-5_1_0_original.json \
    --bcss_json     /path/to/bcss_contoured_gpt-5_1_0_original.json \
    --lung_json     /path/to/lung_contoured_terminology_gpt-5_1_0_original.json \
    --prostate_json /path/to/prostate_contoured_gpt-5_1_0_original.json \
    --splits train test \
    --output_root   .
```

This generates e.g.:
```
HistoPathColon/train.json
HistoPathColon/train_histological.json
HistoPathColon/train_spatial.json
HistoPathColon/train_hierarchical.json
HistoPathColon/train_ambiguity.json
```

---

### 3. Register datasets

The modified `register_biomed_datasets.py` auto-registers all datasets when imported.
No extra action needed. Verify with:

```python
from detectron2.data import DatasetCatalog
datasets = [d for d in DatasetCatalog.list() if "HistoPath" in d]
print(datasets)
# Expected:
# ['biomed_HistoPathColon_train', 'biomed_HistoPathColon_test', ...]
```

For ablation datasets, add this call to your training script:

```python
from datasets.registration.register_biomed_datasets import register_ablation_json_datasets

register_ablation_json_datasets(
    "HistoPathColon",
    "biomedparse_datasets/HistoPathColon",
)
```

---

### 4. Configure training

Edit `configs/biomed_seg_lang_v1.yaml` (or use the provided
`configs/biomed_seg_lang_v1_histopath.yaml`):

```yaml
DATASETS:
  # Fine-tune on custom data only:
  TRAIN: ["biomed_HistoPathColon_train", "biomed_HistoPathBCSS_train",
          "biomed_HistoPathLung_train",  "biomed_HistoPathProstate_train"]

  # Fine-tune on custom + original data:
  # TRAIN: ["biomed_BiomedParseData-Demo_train", "biomed_HistoPathAll_train"]

  PROMPT_DIMENSION: "all"   # or "histological" | "spatial" | "hierarchical" | "ambiguity"
  PROMPT_COMBINE: False
```

---

### 5. Run training (unchanged from original)

```bash
bash assets/scripts/train.sh
```

---

## Ablation Study

To replicate the ablation over reasoning dimensions:

| Config | TRAIN dataset | PROMPT_DIMENSION |
|---|---|---|
| All dims | `biomed_HistoPathAll_train` | `all` |
| Histological only | `biomed_HistoPathAll_train` | `histological` |
| Spatial only | `biomed_HistoPathAll_train` | `spatial` |
| Hierarchical only | `biomed_HistoPathAll_train` | `hierarchical` |
| Ambiguity only | `biomed_HistoPathAll_train` | `ambiguity` |

Alternatively, use dimension-specific JSON datasets for cleaner ablation:

```yaml
DATASETS:
  TRAIN: ["biomed_HistoPathColon_histological_train"]
  TEST:  ["biomed_HistoPathColon_histological_test"]
```

---

## JSON Format Reference

### Input format (your GPT annotation files)

Four dataset-specific variants are handled automatically by `convert_histopath_to_biomedparse.py`:

**Colon:**
```json
{ "testA_1": { "diagnosis": "adenomatous", "regions": {
    "testA_1_query_mask_largest_contoured": { "gpt": { "glands": {
        "testA_1_query_mask_largest_contoured": {
            "Histological Reasoning": ["..."],
            "Spatial Reasoning":      ["..."],
            "Hierarchical Reasoning": ["..."],
            "Ambiguity Reasoning":    ["..."]
        }
    }}}
}}}
```

**BCSS / Prostate (flat):**
```json
{ "image_key": { "regions": {
    "region_key": { "gpt": {
        "Histological Reasoning": ["..."],
        "Ambiguity Reasoning":    ["..."],
        "region_class": "invasive tumor"
    }}
}}}
```

**Lung (one nesting level):**
```json
{ "00": { "regions": {
    "image_00_0": { "gpt": {
        "Tumor epithelium": {
            "Histological Reasoning": ["..."],
            "Ambiguity Reasoning":    ["..."]
        },
        "region_class": "tumor epithelium"
    }}
}}}
```

### Output format (BiomedParse train.json / test.json)

```json
[
  {
    "image_name": "testA_1_pathology_colon.png",
    "mask_name":  "testA_1_pathology_colon_adenomatous+gland.png",
    "sentences": [
      {"sent": "Segment the large, branching adenomatous gland..."},
      {"sent": "Select the dominant, centrally located villous gland..."},
      {"sent": "Isolate the largest and most expansive glandular structure..."},
      {"sent": "Identify the villous gland whose fronds contain..."}
    ]
  }
]
```
