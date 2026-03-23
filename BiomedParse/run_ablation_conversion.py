"""
run_ablation_conversion.py
==========================
Convenience script: generates all dimension-specific JSON files
needed for ablation studies, for one or more datasets.

Run once before training:
    python run_ablation_conversion.py \\
        --colon_json   colon_contoured_gpt-5_1_0_original.json \\
        --bcss_json    bcss_contoured_gpt-5_1_0_original.json \\
        --lung_json    lung_contoured_terminology_gpt-5_1_0_original.json \\
        --prostate_json prostate_contoured_gpt-5_1_0_original.json \\
        --split train \\
        --output_root  biomedparse_datasets

This generates, for example:
    biomedparse_datasets/HistoPathColon/train.json               (all dims)
    biomedparse_datasets/HistoPathColon/train_histological.json
    biomedparse_datasets/HistoPathColon/train_spatial.json
    biomedparse_datasets/HistoPathColon/train_hierarchical.json
    biomedparse_datasets/HistoPathColon/train_ambiguity.json
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

SCRIPT = Path(__file__).parent / "convert_histopath_to_biomedparse.py"
DIMS   = ["all", "histological", "spatial", "hierarchical", "ambiguity"]
DATASET_MAP = {
    "colon_json":    ("colon",    "HistoPathColon"),
    "bcss_json":     ("bcss",     "HistoPathBCSS"),
    "lung_json":     ("lung",     "HistoPathLung"),
    "prostate_json": ("prostate", "HistoPathProstate"),
}


def run(args):
    for arg_name, (dtype, dname) in DATASET_MAP.items():
        json_path = getattr(args, arg_name, None)
        if not json_path:
            continue
        if not os.path.exists(json_path):
            print(f"[SKIP] {json_path} not found, skipping {dname}")
            continue

        for split in args.splits:
            for dim in DIMS:
                suffix = "" if dim == "all" else f"_{dim}"
                out = os.path.join(
                    args.output_root, dname, f"{split}{suffix}.json"
                )
                cmd = [
                    sys.executable, str(SCRIPT),
                    "--input",        json_path,
                    "--dataset_type", dtype,
                    "--split",        split,
                    "--output_json",  out,
                    "--prompt_mode",  dim,
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--colon_json")
    p.add_argument("--bcss_json")
    p.add_argument("--lung_json")
    p.add_argument("--prostate_json")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--output_root", default="biomedparse_datasets")
    run(p.parse_args())
