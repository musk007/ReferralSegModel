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
    "colon":    ("pathology", "colon"),
    "bcss":     ("pathology", "breast"),
    "lung":     ("pathology", "lung"),
    "prostate": ("pathology", "prostate"),
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
        colon   -> region_data["gpt"]["glands"][region_key] (has per-gland sub-dict)
        bcss    -> region_data["gpt"]                        (flat)
        lung    -> region_data["gpt"][class_label]           (one level of nesting)
        prostate-> region_data["gpt"]                        (flat)
    """
    gpt = region_data.get("gpt", {})

    if dataset_type == "colon":
        # gpt["glands"][<region_key>] -> reasoning dims
        glands = gpt.get("glands", {})
        if glands:
            # Pick the first (and typically only) gland sub-entry
            inner = next(iter(glands.values()), {})
            return inner
        return gpt

    elif dataset_type == "lung":
        # gpt[<class_name>] -> reasoning dims
        # Skip known non-dimension keys
        skip_keys = {"region_label", "region_class", "diagnosis"}
        for k, v in gpt.items():
            if k not in skip_keys and isinstance(v, dict):
                return v
        return gpt

    else:
        # bcss and prostate: gpt itself holds reasoning dims
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


def _build_entries(img_key: str, img_data: dict, dataset_type: str,
                   prompt_mode: str, combine_dimensions: bool) -> list[dict]:
    """
    Build a list of BiomedParse JSON entries for one image.

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
    modality, site = MODALITY_MAP[dataset_type]
    entries = []

    for region_key, region_data in img_data.get("regions", {}).items():
        gpt_block = _extract_gpt_block(region_data, dataset_type)

        # Collect the target class name for mask filename
        region_class = (
            gpt_block.get("region_class")
            or region_data.get("gpt", {}).get("region_class")
            or _safe_target_name(region_key, dataset_type)
        )
        # Sanitise: spaces -> '+', no underscores
        target = region_class.replace(" ", "+").replace("_", "+")
        target = re.sub(r"\++", "+", target).strip("+")

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
            # Fallback: concatenate all present reasoning text
            for dim in ALL_DIMS:
                text = _get_dim_text(gpt_block, dim)
                if text:
                    sentences.append(text)

        if not sentences:
            continue  # No usable text → skip

        if combine_dimensions:
            # Merge into single long prompt
            merged = " ".join(sentences)
            sent_list = [{"sent": merged}]
        else:
            sent_list = [{"sent": s} for s in sentences]

        image_name = f"{img_key}_{modality}_{site}.png"
        mask_name  = f"{img_key}_{modality}_{site}_{target}.png"

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
            output_json: str, prompt_mode: str, combine_dimensions: bool):

    with open(input_json) as f:
        data = json.load(f)

    all_entries = []
    for img_key, img_data in data.items():
        entries = _build_entries(
            img_key, img_data, dataset_type,
            prompt_mode, combine_dimensions
        )
        all_entries.extend(entries)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"[{dataset_type}|{split}|{prompt_mode}] Wrote {len(all_entries)} entries "
          f"-> {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         required=True,
                        help="Path to raw GPT annotation JSON")
    parser.add_argument("--dataset_type",  required=True,
                        choices=["colon", "bcss", "lung", "prostate"],
                        help="Which histopathology dataset")
    parser.add_argument("--split",         required=True,
                        choices=["train", "test"],
                        help="Data split")
    parser.add_argument("--output_json",   required=True,
                        help="Output path for BiomedParse-format JSON")
    parser.add_argument("--prompt_mode",   default="all",
                        choices=["all", "histological", "spatial",
                                 "hierarchical", "ambiguity"],
                        help="Which reasoning dimension(s) to include in text prompts")
    parser.add_argument("--combine_dimensions", action="store_true",
                        help="Merge all selected dimensions into one sentence")
    args = parser.parse_args()

    convert(
        input_json=args.input,
        dataset_type=args.dataset_type,
        split=args.split,
        output_json=args.output_json,
        prompt_mode=args.prompt_mode,
        combine_dimensions=args.combine_dimensions,
    )
