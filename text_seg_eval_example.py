"""
Text-Prompted Segmentation Evaluation Pipeline

This script evaluates text-prompted segmentation models using the text_seg_eval framework.
All model wrappers are imported from the wrappers/ directory.

Supported models:
- SAM3 (Meta's Segment Anything 3)
- BiomedParse v1/v2 (Microsoft's biomedical segmentation)
- DualProtoSeg (Histopathology prototype learning)
- MediSee (Reasoning-based Pixel-level Perception in Medical Images)

Usage:

    bash run.sh
            
    # List available models
    python text_seg_eval_example.py --list_models
"""

import os
import sys
import json
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Any, Optional

# Add current directory and wrappers to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WRAPPERS_DIR = os.path.join(SCRIPT_DIR, "wrappers")
BIOMEDPARSE_PATH = os.path.join(SCRIPT_DIR, "BiomedParse")

for path in [SCRIPT_DIR, WRAPPERS_DIR, BIOMEDPARSE_PATH]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import the evaluation framework
from text_seg_eval import (
    TextSegmentationModel,
    evaluate_model,
    compare_models,
    EvaluationResults,
    AVAILABLE_METRICS,
    DEFAULT_METRICS,
)

REASONING_KEYS = [
    "Histological Reasoning",
    "Spatial Reasoning",
    "Hierarchical Reasoning",
    "Ambiguity Reasoning",
]
# =============================================================================
# Model Imports from Wrappers
# =============================================================================

# Dictionary to hold available models
AVAILABLE_MODELS = {}

# Try to import each wrapper - failures are silent (model just won't be available)
def _register_wrapper(name: str, module_name: str, class_name: str, aliases: List[str] = None):
    """Helper to safely import and register a wrapper."""
    try:
        module = __import__(module_name, fromlist=[class_name])
        wrapper_class = getattr(module, class_name)
        AVAILABLE_MODELS[name] = wrapper_class
        # Register aliases
        if aliases:
            for alias in aliases:
                AVAILABLE_MODELS[alias] = wrapper_class
        return True
    except (ImportError, AttributeError) as e:
        # Silently skip unavailable wrappers
        return False


# Register available wrappers
_register_wrapper("sam3", "sam3_wrapper", "SAM3Wrapper")
_register_wrapper("biomedparse", "biomedparse_wrapper", "BiomedParseV1Wrapper", aliases=["biomedparse_v1"])
_register_wrapper("dualprotoseg", "dualprotoseg_wrapper", "DualProtoSegWrapper")
_register_wrapper("medisee", "medisee_wrapper", "MediSeeWrapper")

# Fallback: try importing from root-level biomedparse_wrapper.py if wrappers/ import failed
if "biomedparse" not in AVAILABLE_MODELS:
    _register_wrapper("biomedparse", "biomedparse_wrapper", "BiomedParseV1Wrapper", aliases=["biomedparse_v1"])
    _register_wrapper("biomedparse", "biomedparse_wrapper", "BiomedParseWrapper")


# =============================================================================
# Model Factory
# =============================================================================

def get_model(
    model_name: str,
    device: str = "cuda",
    **kwargs,
) -> TextSegmentationModel:
    """
    Factory function to get a model wrapper by name.
    
    Args:
        model_name: Model identifier (see --list_models for available options)
        device: Device to run model on ('cuda' or 'cpu')
        **kwargs: Additional arguments for specific model
        
    Returns:
        Model wrapper instance
    """
    if model_name not in AVAILABLE_MODELS:
        available = ", ".join(sorted(AVAILABLE_MODELS.keys()))
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {available}"
        )
    
    model_class = AVAILABLE_MODELS[model_name]
    return model_class(device=device, **kwargs)


def list_available_models():
    """Print list of available models."""
    print("\n" + "=" * 70)
    print("AVAILABLE SEGMENTATION MODELS")
    print("=" * 70)
    
    model_info = {
        "sam3": "SAM3 - Meta's Segment Anything with text prompts",
        "biomedparse": ["BiomedParse v1 - 2D multi-modality medical segmentation (9 modalities)",
                        "First version of biomedparse (includes histopathology images)"],
        "dualprotoseg": "DualProtoSeg - Histopathology prototype learning",
        "medisee": "MediSee - Reasoning-based pixel-level perception in medical images"
    }
    
    for name in sorted(AVAILABLE_MODELS.keys()):
        status = "✓ Available"
        desc = model_info.get(name, "")
        print(f"\n  {name}")
        print(f"    Status: {status}")
        if desc:
            print(f"    {desc}")
    
    # Show models that failed to load
    all_models = ["sam3", "biomedparse", "dualprotoseg", "medisee"]
    unavailable = [m for m in all_models if m not in AVAILABLE_MODELS]
    
    if unavailable:
        print("\n  --- Unavailable (dependencies not installed) ---")
        for name in unavailable:
            print(f"  ✗ {name}")
    
    print("\n" + "=" * 70)


# =============================================================================
# Dataset Loading Utilities
# =============================================================================

def load_simple_dataset(
    images_dir: str,
    masks_dir: str,
    prompts: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Load a simple dataset with images, masks, and prompts.
    
    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing GT masks
        prompts: Dict mapping image names (without extension) to prompts
        
    Returns:
        List of sample dictionaries
    """
    dataset = []
    
    for img_name, prompt in prompts.items():
        # Find image
        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            candidate = os.path.join(images_dir, f"{img_name}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
        
        if img_path is None:
            print(f"Warning: Image not found for '{img_name}'")
            continue
        
        # Find mask
        mask_path = None
        for ext in [".png", ".jpg", ".npy"]:
            candidate = os.path.join(masks_dir, f"{img_name}{ext}")
            if os.path.exists(candidate):
                mask_path = candidate
                break
        
        if mask_path is None:
            print(f"Warning: Mask not found for '{img_name}'")
            continue
        
        dataset.append({
            "id": img_name,
            "image_path": img_path,
            "mask_path": mask_path,
            "prompt": prompt,
        })
    
    print(f"Loaded {len(dataset)} samples")
    return dataset


def _flatten_reasoning_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []


def _resolve_file_path(base_dir: str, base_name: str, extensions: List[str]) -> Optional[str]:
    """Return the first existing path base_dir/base_name+ext for ext in extensions, else None."""
    for ext in extensions:
        candidate = os.path.join(base_dir, f"{base_name}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


# Extensions tried when resolving image/mask paths for region-based datasets (colon, bcss, prostate).
IMAGE_EXTENSIONS = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]
MASK_EXTENSIONS = [".png", ".npy"]


def load_biomedparse_coco_dataset(
    images_dir: str,
    masks_dir: str,
    json_path: str,
) -> List[Dict[str, Any]]:
    """
    Load BiomedParse official dataset (COCO format: test.json with images + annotations).
    Each annotation has mask_file, sentences[{'raw': prompt}], linked to image via image_id.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    images_by_id = {img["id"]: img for img in data.get("images", [])}
    dataset = []
    for ann in data.get("annotations", []):
        img = images_by_id.get(ann["image_id"])
        if not img:
            continue
        img_path = os.path.join(images_dir, img["file_name"])
        mask_path = os.path.join(masks_dir, ann.get("mask_file", img["file_name"]))
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        sents = ann.get("sentences", [])
        prompt = sents[0].get("raw", sents[0].get("sent", "")) if sents else ""
        dataset.append({
            "id": f"{ann.get('id', len(dataset))}__{img['file_name']}",
            "image_path": img_path,
            "mask_path": mask_path,
            "prompt": prompt,
        })
    print(f"Loaded {len(dataset)} samples from BiomedParse COCO {json_path}")
    return dataset


def load_json_dataset(
    images_dir: str,
    masks_dir: str,
    json_path: str,
    image_key: str = "image",
    mask_key: str = "mask",
    prompt_key: str = "prompt",
    id_key: str = "id",
) -> List[Dict[str, Any]]:

    with open(json_path, "r") as f:
        data = json.load(f)

    dataset: List[Dict[str, Any]] = []

    if isinstance(data, list):
        items = [(d.get(id_key, f"sample_{i}"), d) for i, d in enumerate(data)]
    else:
        items = list(data.items())
    
    for sample_id, sample in items:
        # -----------------------------
        # Region-based format (colon GlaS-style with glands, or BCSS/prostate with gpt per region)
        # -----------------------------
        regions = sample.get("regions", {})
        if isinstance(regions, dict) and regions:
            case_dx = (sample.get("diagnosis", "") or "").strip()

            # 1) Try colon-style: one concatenated prompt from ALL regions' gpt.glands
            parts: List[str] = []
            for region_id, region_block in regions.items():
                gpt = (region_block or {}).get("gpt", {})
                if not isinstance(gpt, dict):
                    continue
                glands = gpt.get("glands", {})
                if not isinstance(glands, dict):
                    continue
                for gland_id, gland_block in glands.items():
                    if not isinstance(gland_block, dict):
                        continue
                    for rk in REASONING_KEYS:
                        vals = _flatten_reasoning_list(gland_block.get(rk))
                        parts.extend(vals)

            seen = set()
            unique_parts: List[str] = []
            for p in parts:
                if p not in seen:
                    seen.add(p)
                    unique_parts.append(p)

            case_prompt = "\n".join(unique_parts).strip()
            if not case_prompt:
                case_prompt = case_dx

            # If we got a case-level prompt from glands (colon), use it for all regions
            if case_prompt and unique_parts:
                per_region_prompts = None  # one prompt for all
            else:
                # 2) BCSS/prostate: no glands — build one prompt per region from gpt reasoning keys
                per_region_prompts = {}
                for region_id, region_block in regions.items():
                    gpt = (region_block or {}).get("gpt", {})
                    if not isinstance(gpt, dict):
                        per_region_prompts[region_id] = case_dx or ""
                        continue
                    r_parts: List[str] = []
                    # Reasoning keys may be directly under gpt, or nested under a
                    # class-name key (BCSS format: gpt[class_name][reasoning_key]).
                    for rk in REASONING_KEYS:
                        r_parts.extend(_flatten_reasoning_list(gpt.get(rk)))
                    if not r_parts:
                        for val in gpt.values():
                            if isinstance(val, dict):
                                for rk in REASONING_KEYS:
                                    r_parts.extend(_flatten_reasoning_list(val.get(rk)))
                    fallback = (
                        (gpt.get("region_class") or gpt.get("diagnosis") or case_dx) or ""
                    )
                    if isinstance(fallback, list):
                        fallback = " ".join(str(x) for x in fallback).strip()
                    else:
                        fallback = str(fallback).strip()
                    prompt_text = "\n".join(r_parts).strip() if r_parts else fallback
                    per_region_prompts[region_id] = prompt_text or case_dx

            # 3) Resolve image path (support .bmp, .png, .jpg, etc.)
            img_path = _resolve_file_path(images_dir, sample_id, IMAGE_EXTENSIONS)
            if not img_path:
                continue

            for region_id in regions.keys():
                mask_path = _resolve_file_path(masks_dir, region_id, MASK_EXTENSIONS)
                if not mask_path:
                    continue
                prompt = (
                    case_prompt
                    if per_region_prompts is None
                    else per_region_prompts.get(region_id, case_dx)
                )
                dataset.append({
                    "id": f"{sample_id}__{region_id}",
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "prompt": prompt,
                })

            continue  # avoid also treating this as "standard format"

        # -----------------------------
        # Standard format
        # -----------------------------
        img_file = sample.get(image_key)
        mask_file = sample.get(mask_key)
        prompt = sample.get(prompt_key, "")

        if not img_file or not mask_file:
            continue

        img_path = os.path.join(images_dir, img_file) if not os.path.isabs(img_file) else img_file
        mask_path = os.path.join(masks_dir, mask_file) if not os.path.isabs(mask_file) else mask_file

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        dataset.append({
            "id": sample_id,
            "image_path": img_path,
            "mask_path": mask_path,
            "prompt": prompt,
        })

    print(f"Loaded {len(dataset)} samples from {json_path}")
    return dataset


# =============================================================================
# Main CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate text-prompted segmentation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # List available models
        python text_seg_eval_example.py --list_models

        # Evaluate BiomedParse
        python text_seg_eval_example.py \\
            --model biomedparse --images_dir ./images --masks_dir ./masks \\
            --prompts_json ./prompts.json

        # Evaluate SAM3
        python text_seg_eval_example.py \\
            --model sam3 --images_dir ./images --masks_dir ./masks \\
            --prompts_json ./prompts.json

        # Compare multiple models
        python text_seg_eval_example.py \\
            --compare biomedparse sam3 --images_dir ./images --masks_dir ./masks \\
            --prompts_json ./prompts.json
                """
    )
    
    # Model selection
    parser.add_argument("--model", type=str, default="biomedparse",
                       help="Model to evaluate (use --list_models to see options)")
    parser.add_argument("--compare", type=str, nargs="+", default=None,
                       help="Compare multiple models")
    parser.add_argument("--list_models", action="store_true",
                       help="List all available models and exit")
    
    # Data paths
    parser.add_argument("--images_dir", type=str,
                       help="Directory containing images")
    parser.add_argument("--masks_dir", type=str,
                       help="Directory containing ground truth masks")
    parser.add_argument("--prompts_json", type=str,
                       help="JSON file with prompts")
    
    # Output
    parser.add_argument("--output", type=str, default="eval_results.json",
                       help="Output JSON file")
    parser.add_argument("--save_predictions", type=str, default=None,
                       help="Directory to save predicted masks")
    
    # Model settings
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Detection/segmentation threshold")
    
    # Model-specific paths
    parser.add_argument("--biomedparse_path", type=str, 
                       default="/home/roba/miccai26/BiomedParse",
                       help="Path to BiomedParse repository")
    parser.add_argument("--sam3_path", type=str, default=None,
                       help="Path to SAM3 repository")
    parser.add_argument("--dualprotoseg_path", type=str, default=None,
                       help="Path to DualProtoSeg repository")
    parser.add_argument("--dataset", type=str, default="colon",
                       help="Dataset name for loading class prompts (colon, bcss, liver, prostate)")
    parser.add_argument("--prompts_config", type=str, default=None,
                       help="Path to class prompts config file (default: configs/class_prompts.yaml)")
    parser.add_argument("--medisee_path", type=str,
                       default="/home/roba/miccai26/MediSee",
                       help="Path to MediSee repository")
    parser.add_argument("--medisee_model_dir", type=str,
                       default="/home/roba/miccai26/models/medisee",
                       help="Directory containing MediSee checkpoints")
    parser.add_argument("--sat_path", type=str, default=None,
                       help="Path to SAT repository")
    parser.add_argument("--textdiff_path", type=str, default=None,
                       help="Path to TextDiff repository")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint (for dualprotoseg, medisee, etc.)")
    parser.add_argument("--biomedparse_checkpoint", type=str, default=None,
                       help="Path to BiomedParse fine-tuned checkpoint (dir or model_state_dict.pt)")
    parser.add_argument("--biomedparse_official_data", type=str, default=None,
                       help="Path to BiomedParseData root (e.g. biomedparse_datasets/biomedParse/BiomedParseData). "
                            "When set with --biomedparse_official_datasets, loads COCO-format test splits.")
    parser.add_argument("--biomedparse_official_datasets", type=str, default=None,
                       help="Comma-separated dataset names (e.g. ACDC,DRIVE,ISIC) or 'all' for all registered.")
    
    # Metrics
    parser.add_argument("--metrics", type=str, nargs="+",
                       default=["iou", "dice", "precision", "recall", "accuracy", "giou", "ciou"],
                       help=f"Metrics to compute. Available: {AVAILABLE_METRICS}")
    
    args = parser.parse_args()
    
    # Handle --list_models
    if args.list_models:
        list_available_models()
        return
    
    # Validate required arguments
    # Standard mode (run.sh, eval.sh run_eval): requires images_dir, masks_dir, prompts_json
    # BiomedParse official mode: requires --biomedparse_official_data and --biomedparse_official_datasets
    if not args.biomedparse_official_data or not args.biomedparse_official_datasets:
        # Original evaluation path - unchanged from before
        if not args.images_dir or not args.masks_dir or not args.prompts_json:
            parser.error("--images_dir, --masks_dir, and --prompts_json are required for evaluation "
                         "(or use --biomedparse_official_data and --biomedparse_official_datasets)")
    
    print("=" * 70)
    print("TEXT-PROMPTED SEGMENTATION EVALUATION")
    print("=" * 70)
    
    # Load dataset
    print(f"\nLoading dataset...")
    if args.biomedparse_official_data and args.biomedparse_official_datasets:
        # BiomedParse official COCO format (eval.sh BIOMEDPARSE_OFFICIAL=1)
        _BIOMED_OFFICIAL = [
            "ACDC", "BreastUS", "CDD-CESM", "DRIVE", "G1020", "ISIC", "LGG",
            "OCT-CME", "PanNuke", "UWaterlooSkinCancer",
        ]
        names = _BIOMED_OFFICIAL if args.biomedparse_official_datasets.lower() == "all" else [
            n.strip() for n in args.biomedparse_official_datasets.split(",") if n.strip()
        ]
        dataset = []
        for ds_name in names:
            root = os.path.join(args.biomedparse_official_data, ds_name)
            img_dir = os.path.join(root, "test")
            mask_dir = os.path.join(root, "test_mask")
            json_path = os.path.join(root, "test.json")
            if not os.path.isfile(json_path):
                print(f"  Skipping {ds_name}: test.json not found")
                continue
            dataset.extend(load_biomedparse_coco_dataset(img_dir, mask_dir, json_path))
    else:
        # Original path: run.sh and eval.sh run_eval use load_json_dataset
        dataset = load_json_dataset(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            json_path=args.prompts_json,
        )
    
    if len(dataset) == 0:
        print("ERROR: No samples found! Check your paths.")
        return
    
    # Prepare model kwargs
    model_kwargs = {
        "threshold": args.threshold,
    }
    
    # Add model-specific kwargs based on model name
    model_name = args.model.lower()
    
    if "biomedparse" in model_name:
        model_kwargs["biomedparse_path"] = args.biomedparse_path
        if args.biomedparse_checkpoint:
            model_kwargs["model_checkpoint"] = args.biomedparse_checkpoint
        elif args.checkpoint and args.model.lower() == "biomedparse":
            model_kwargs["model_checkpoint"] = args.checkpoint
    
    if model_name == "sam3" and args.sam3_path:
        model_kwargs["sam3_path"] = args.sam3_path
    
    if model_name == "dualprotoseg":
        if args.dualprotoseg_path:
            model_kwargs["dualprotoseg_path"] = args.dualprotoseg_path
        if args.checkpoint:
            model_kwargs["checkpoint_path"] = args.checkpoint
        if hasattr(args, 'dataset') and args.dataset:
            model_kwargs["dataset"] = args.dataset
        if hasattr(args, 'prompts_config') and args.prompts_config:
            model_kwargs["prompts_config"] = args.prompts_config
    
    if model_name == "medisee":
        if args.medisee_path:
            model_kwargs["medisee_path"] = args.medisee_path
        if args.medisee_model_dir:
            model_kwargs["model_dir"] = args.medisee_model_dir
        if args.checkpoint:
            model_kwargs["model_path"] = args.checkpoint
    
    if model_name == "sat":
        if args.sat_path:
            model_kwargs["sat_path"] = args.sat_path
        if args.checkpoint:
            model_kwargs["model_path"] = args.checkpoint
    
    if model_name == "textdiff":
        if args.textdiff_path:
            model_kwargs["textdiff_path"] = args.textdiff_path
        if args.checkpoint:
            model_kwargs["model_path"] = args.checkpoint
    
    # Generic checkpoint fallback
    if args.checkpoint and "checkpoint_path" not in model_kwargs and "model_path" not in model_kwargs:
        model_kwargs["checkpoint_path"] = args.checkpoint
    
    # Compare multiple models or evaluate single model
    if args.compare:
        print(f"\nComparing models: {args.compare}")
        models = {}
        for model_name in args.compare:
            print(f"  Loading {model_name}...")
            kwargs = {k: v for k, v in model_kwargs.items()
                      if k != "checkpoint_path" and k != "model_checkpoint"}
            if "biomedparse" in model_name and args.biomedparse_checkpoint:
                kwargs["model_checkpoint"] = args.biomedparse_checkpoint
            elif "dualprotoseg" in model_name and args.checkpoint:
                kwargs["checkpoint_path"] = args.checkpoint
            try:
                models[model_name] = get_model(
                    model_name,
                    device=args.device,
                    **kwargs,
                )
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
        
        if not models:
            print("ERROR: No models loaded successfully")
            return
        
        all_results = compare_models(
            models=models,
            dataset=dataset,
            metrics=args.metrics,
            output_path=args.output,
        )
    else:
        # Single model evaluation
        print(f"\nLoading model: {args.model}")
        model = get_model(
            args.model,
            device=args.device,
            **model_kwargs,
        )
        
        print(f"\nEvaluating with metrics: {args.metrics}")
        results = evaluate_model(
            model=model,
            dataset=dataset,
            metrics=args.metrics,
            verbose=True,
            save_predictions=args.save_predictions,
        )
        
        results.save_json(args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
