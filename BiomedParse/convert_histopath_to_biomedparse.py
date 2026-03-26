"""
convert_histopath_to_biomedparse.py
====================================
Converts multi-dimensional histopathology referral-segmentation JSON annotations
(colon, BCSS breast, lung, prostate) into the BiomedParse v1 train/test JSON format.

BiomedParse v1 expects a JSON file where each entry is:
{
    "image_name": "<IMAGE-NAME>_<MODALITY>_<SITE>.png",
    "mask_name":  "<IMAGE-NAME>_<MODALITY>_<SITE>_<TARGET>.png",
    "sentences": [
        {"sent": "<text prompt>"},
        ...          # one entry per reasoning dimension selected
    ]
}

Usage
-----
python convert_histopath_to_biomedparse.py \
    --input  colon_contoured_gpt-5_1_0_original.json \
    --dataset_type  colon \
    --split  train \
    --output_json  biomedparse_datasets/HistoPathColon/train.json \
    --prompt_mode  all          # "all" | "histological" | "spatial" | "hierarchical" | "ambiguity"
    --combine_dimensions        # if set, concatenates all dims into a single sentence

Dataset types: colon | bcss | lung | prostate
"""

import argparse
import json
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Dimension key aliases (case-insensitive matching)
# ---------------------------------------------------------------------------
DIM_ALIASES = {
    "histological": ["Histological Reasoning", "histological reasoning"],
    "spatial":      ["Spatial Reasoning",      "spatial reasoning"],
    "hierarchical": ["Hierarchical Reasoning", "hierarchical reasoning"],
    "ambiguity":    ["Ambiguity Reasoning",    "ambiguity reasoning"],
}

ALL_DIMS = list(DIM_ALIASES.keys())

MODALITY_MAP = {
    "colon":        ("pathology", "colon"),
    "bcss":         ("pathology", "breast"),
    "breast_cells": ("pathology", "breast_cells"),
    "lung":         ("pathology", "lung"),
    "prostate":     ("pathology", "prostate"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_dim_text(gpt_block: dict, dim: str) -> str | None:
    """
    Extract the first sentence for a given reasoning dimension from a GPT block.
    Returns None if the dimension is absent in this block.
    """
    for alias in DIM_ALIASES[dim]:
        if alias in gpt_block:
            sentences = gpt_block[alias]
            if sentences:
                return sentences[0]
    return None


def _extract_gpt_block(region_data: dict, dataset_type: str) -> dict:
    """
    Navigate the (sometimes nested) GPT block to reach the dict that holds
    the reasoning dimension keys directly.

    Structure differences:
        colon        -> region_data["gpt"]["glands"][region_key] (per-gland sub-dict)
        bcss         -> region_data["gpt"][class_label]          (one level of nesting)
        breast_cells -> region_data["gpt"][class_label]          (one level of nesting)
        lung         -> region_data["gpt"][class_label]          (one level of nesting)
        prostate     -> region_data["gpt"]                       (flat, or one level)
    """
    gpt = region_data.get("gpt", {})

    if dataset_type == "colon":
        # gpt["glands"][<region_key>] -> reasoning dims
        glands = gpt.get("glands", {})
        if glands:
            inner = next(iter(glands.values()), {})
            return inner
        return gpt

    else:
        # bcss, breast_cells, lung, prostate:
        # Reasoning keys may sit directly on gpt (old flat format) OR be nested one
        # level deeper under a class-name key (newer format).  Check for the nested
        # case by looking for any dict-valued key that is not a known metadata key.
        skip_keys = {"region_label", "region_class", "diagnosis"}
        # First, see if reasoning keys are already at the top level
        for alias_list in DIM_ALIASES.values():
            for alias in alias_list:
                if alias in gpt:
                    return gpt  # flat format
        # Otherwise look for a single nested class-name dict
        for k, v in gpt.items():
            if k not in skip_keys and isinstance(v, dict):
                return v
        return gpt


def _safe_target_name(region_key: str, dataset_type: str) -> str:
    """
    Derive a safe [TARGET] string (no underscores, no special chars) from the region key.
    BiomedParse mask naming: [IMAGE]_[MODALITY]_[SITE]_[TARGET].png
    Underscores in TARGET must be replaced with '+'; brackets/commas removed.
    """
    # Replace all non-alphanumeric characters with '+'
    name = re.sub(r"[^a-zA-Z0-9]", "+", region_key)
    # Collapse consecutive '+'
    name = re.sub(r"\++", "+", name).strip("+")
    return name


def _find_image_name(img_key: str, images_dir: str | None) -> str:
    """
    Return the filename (basename only) for an image given its key.
    If images_dir is provided, try common extensions to find the actual file.
    Falls back to <img_key>.png.
    """
    if images_dir:
        for ext in (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            candidate = os.path.join(images_dir, f"{img_key}{ext}")
            if os.path.exists(candidate):
                return f"{img_key}{ext}"
    return f"{img_key}.png"


def _build_entries(img_key: str, img_data: dict, dataset_type: str,
                   prompt_mode: str, combine_dimensions: bool,
                   images_dir: str | None = None) -> list[dict]:
    """
    Build a list of BiomedParse JSON entries for one image.

    Image filename  : <img_key>.<ext>   (ext detected from images_dir, default .png)
    Mask filename   : <region_key>.png  (matches the actual mask files on disk)

    prompt_mode:
        "all"          – one entry per region, sentences list has one sent per available dim
        "histological" – one entry per region, only histological dim
        "spatial"      – one entry per region, only spatial dim
        "hierarchical" – one entry per region, only hierarchical dim
        "ambiguity"    – one entry per region, only ambiguity dim

    combine_dimensions (bool):
        If True, all selected dimensions are merged into a single sentence (joined by " ").
        If False, each dimension produces a separate {"sent": ...} entry.
    """
    entries = []
    image_name = _find_image_name(img_key, images_dir)

    for region_key, region_data in img_data.get("regions", {}).items():
        gpt_block = _extract_gpt_block(region_data, dataset_type)

        # Select which dimensions to include
        if prompt_mode == "all":
            dims_to_use = ALL_DIMS
        else:
            dims_to_use = [prompt_mode]

        # Gather sentences for selected dimensions
        sentences = []
        for dim in dims_to_use:
            text = _get_dim_text(gpt_block, dim)
            if text:
                sentences.append(text)

        if not sentences:
            # Fallback: any present reasoning text
            for dim in ALL_DIMS:
                text = _get_dim_text(gpt_block, dim)
                if text:
                    sentences.append(text)

        if not sentences:
            continue  # No usable text → skip

        if combine_dimensions:
            merged = " ".join(sentences)
            sent_list = [{"sent": merged}]
        else:
            sent_list = [{"sent": s} for s in sentences]

        # Mask filename matches what's actually on disk: <region_key>.png
        mask_name = f"{region_key}.png"

        entries.append({
            "image_name": image_name,
            "mask_name":  mask_name,
            "sentences":  sent_list,
        })

    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(input_json: str, dataset_type: str, split: str,
            output_json: str, prompt_mode: str, combine_dimensions: bool,
            images_dir: str | None = None):

    with open(input_json) as f:
        data = json.load(f)

    all_entries = []

    for img_key, img_data in data.items():
        entries = _build_entries(
            img_key, img_data, dataset_type,
            prompt_mode, combine_dimensions,
            images_dir=images_dir,
        )
        all_entries.extend(entries)

    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"[{dataset_type}|{split}|{prompt_mode}] Wrote {len(all_entries)} entries "
          f"-> {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw GPT annotation JSON")
    parser.add_argument("--dataset_type", required=True, choices=["colon", "bcss", "breast_cells", "lung", "prostate"], help="Which histopathology dataset")
    parser.add_argument("--split", required=True, choices=["train", "eval", "test"], help="Data split")
    parser.add_argument("--output_json", required=True, help="Output path for BiomedParse-format JSON")
    parser.add_argument("--prompt_mode", default="all", choices=["all", "histological", "spatial", "hierarchical", "ambiguity"], help="Which reasoning dimension(s) to include in text prompts")
    parser.add_argument("--images_dir", default=None, help="Directory containing images for this split (used to detect the correct file extension).If omitted, .png is assumed.")
    parser.add_argument("--combine_dimensions", action="store_true", help="Merge all selected dimensions into one sentence")
    args = parser.parse_args()

    convert(
        input_json=args.input,
        dataset_type=args.dataset_type,
        split=args.split,
        output_json=args.output_json,
        prompt_mode=args.prompt_mode,
        combine_dimensions=args.combine_dimensions,
        images_dir=args.images_dir,
    )
