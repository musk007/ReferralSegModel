"""
Fit Beta distributions for ALL FOUR BiomedParse Pathology targets on your
data, using the pretrained v1 model AND the real GPT-generated text
prompts from your instruction JSONs (not a template).

What this script does:
    1. Reads your instruction JSON (same one used by eval.sh).
    2. For each region in that JSON:
       - resolves image and per-region mask paths (same logic as your eval)
       - extracts the real text prompt from the region's gpt.* fields
       - reads the region's 'region_class' value
       - maps region_class -> BiomedParse target via CLASS_MAP below
    3. Runs BiomedParse with that real prompt on that image.
    4. Records four statistics (prob_mean over GT mask + R/G/B mean over GT mask).
    5. After all regions are processed, fits Beta(alpha, beta) for each of the
       four targets and prints a side-by-side comparison with HF values.

YOU MUST EDIT THE CLASS_MAP DICT BELOW to reflect how your region classes map
to BiomedParse's 4 Pathology targets. Unmapped classes are skipped.

Usage:
    python fit_target_dist_real_prompts.py <images_dir> <masks_dir> \\
        <instructions_json> <biomedparse_repo_path>

Example (matching eval.sh 'breast_cells' entry):
    python fit_target_dist_real_prompts.py \\
        /adialab/usr/roba/biomedparse_datasets/breast_cells/test \\
        /adialab/usr/roba/biomedparse_datasets/breast_cells/test_mask \\
        /home/roba/miccai26/test_data/instructions/test/breast_cells.json \\
        /home/user/ReferralSegModel/BiomedParse
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from scipy import stats
import torch

# ---- MAP YOUR REGION CLASSES TO BIOMEDPARSE TARGETS -------------------
# Edit this to match the region_class values actually present in your
# instruction JSON. Any region with a class NOT listed here is skipped.
# The four BiomedParse Pathology targets are:
#   "neoplastic cells", "inflammatory cells",
#   "connective tissue cells", "epithelial cells"
CLASS_MAP = {
    # ---- breast_cells.json and breast_bcss.json ----
    "invasive tumor":                     "neoplastic cells",
    "in-situ tumor":                      "neoplastic cells",
    "tumor-associated stroma":            "connective tissue cells",
    "inflamed stroma":                    "inflammatory cells",
    "Healthy glands":                     "epithelial cells",
    # "necrosis not in-situ":             <no good BiomedParse target, leave unmapped>

    # ---- colon.json ----
    "moderately differentiated":          "neoplastic cells",
    "poorly differentiated":              "neoplastic cells",
    "moderately-to-poorly differentated": "neoplastic cells",
    "adenomatous":                        "neoplastic cells",
    "healthy":                            "epithelial cells",

    # ---- prostate.json ----
    "Gleason grade 3 tissue":             "neoplastic cells",
    "Gleason grade 4 tissue":             "neoplastic cells",
    "Gleason grade 5 tissue":             "neoplastic cells",
    "Benign":                             "epithelial cells",

    # ---- lung.json ----
    "tumor epithelium":                   "neoplastic cells",
    "tumor stroma":                       "connective tissue cells",
    "normal tissue":                      "epithelial cells",
}

# ---- arguments --------------------------------------------------------
images_dir         = sys.argv[1]
masks_dir          = sys.argv[2]
instructions_json  = sys.argv[3]
biomedparse_path   = sys.argv[4]
hf_json_path       = "target_dist_HF.json"

TARGETS = ["neoplastic cells", "inflammatory cells",
           "connective tissue cells", "epithelial cells"]

# ---- build the sample list using your existing dataset loader ---------
# This gives us the SAME prompts and paths that your eval pipeline uses.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_seg_eval_example import load_json_dataset, REASONING_KEYS

samples = load_json_dataset(images_dir, masks_dir, instructions_json)
print(f"load_json_dataset returned {len(samples)} regions.")

# load_json_dataset doesn't return region_class; re-open the JSON to look it up
raw = json.load(open(instructions_json))
region_class_lookup = {}
for image_id, image_entry in raw.items():
    for region_id, region_block in (image_entry.get("regions") or {}).items():
        gpt = (region_block or {}).get("gpt", {}) or {}
        rc = gpt.get("region_class") or gpt.get("diagnosis") or image_entry.get("diagnosis")
        if isinstance(rc, list):
            rc = " ".join(str(x) for x in rc).strip()
        region_class_lookup[f"{image_id}__{region_id}"] = str(rc).strip() if rc else ""

# ---- load the BiomedParse v1 model (once) -----------------------------
sys.path.insert(0, biomedparse_path)
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image

config_path = os.path.join(biomedparse_path, "configs", "biomed_seg_lang_v1_inference.yaml")
opt = load_opt_from_config_files([config_path])
opt = init_distributed(opt)

model = BaseModel(opt, build_model(opt)).from_pretrained("hf_hub:microsoft/BiomedParse").eval()
if torch.cuda.is_available():
    model = model.cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
        BIOMED_CLASSES + ["background"], is_eval=True
    )
print("Model loaded.")

# ---- collect statistics bucketed by BiomedParse target ----------------
stats_per_target = {t: {"prob": [], "R": [], "G": [], "B": []} for t in TARGETS}
unmapped_classes = {}       # region_class -> count (for the final report)
n_empty_mask = 0
n_empty_prompt = 0

for i, sample in enumerate(samples):
    rc = region_class_lookup.get(sample["id"], "")
    target = CLASS_MAP.get(rc)
    if target is None:
        unmapped_classes[rc] = unmapped_classes.get(rc, 0) + 1
        continue

    pil_img   = Image.open(sample["image_path"]).convert("RGB")
    img_np    = np.array(pil_img)
    mask_np   = np.array(Image.open(sample["mask_path"]).convert("L"))
    mask_bool = mask_np > 127
    if mask_bool.sum() < 10:
        n_empty_mask += 1
        continue

    # KEY STEP: use the real text prompt from the instruction JSON,
    # not a template. load_json_dataset already built it from the gpt.* fields.
    text_prompt = sample["prompt"]
    if not text_prompt:
        n_empty_prompt += 1
        continue

    pred_prob = interactive_infer_image(model, pil_img, [text_prompt])[0]
    pred_prob = np.asarray(pred_prob)
    if pred_prob.ndim > 2:
        pred_prob = np.squeeze(pred_prob)
    if pred_prob.shape != mask_bool.shape:
        # Some inference paths may return logits/probability maps at a different
        # spatial resolution than the original mask. Resize to GT-mask size.
        pred_prob = np.array(
            Image.fromarray(pred_prob.astype(np.float32), mode="F").resize(
                (mask_bool.shape[1], mask_bool.shape[0]),
                Image.BILINEAR,
            ),
            dtype=np.float32,
        )

    stats_per_target[target]["prob"].append(float(pred_prob[mask_bool].mean()))
    stats_per_target[target]["R"].append(img_np[..., 0][mask_bool].mean() / 255.0)
    stats_per_target[target]["G"].append(img_np[..., 1][mask_bool].mean() / 255.0)
    stats_per_target[target]["B"].append(img_np[..., 2][mask_bool].mean() / 255.0)

    if (i + 1) % 50 == 0:
        print(f"  processed {i+1}/{len(samples)} regions")

# ---- diagnostic report on skipped regions -----------------------------
print("\n--- skipped regions ---")
print(f"empty/tiny masks:      {n_empty_mask}")
print(f"empty text prompts:    {n_empty_prompt}")
print(f"unmapped region_class: {sum(unmapped_classes.values())}")
if unmapped_classes:
    print("  classes not in CLASS_MAP (add them if you want to include them):")
    for rc, n in sorted(unmapped_classes.items(), key=lambda x: -x[1]):
        print(f"    {n:>5}  {rc!r}")

# ---- fit Beta distributions and compare to HF -------------------------
hf = None
hf_candidates = [
    hf_json_path,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), hf_json_path),
    os.path.join(biomedparse_path, hf_json_path),
]
for p in hf_candidates:
    if os.path.isfile(p):
        with open(p) as f:
            hf = json.load(f)
        hf_json_path = p
        break
if hf is None:
    print(
        f"Warning: {hf_json_path!r} not found. "
        "Proceeding without HF side-by-side comparison."
    )
else:
    print(f"Loaded HF reference from: {hf_json_path}")

def beta_mean_std(a, b):
    m = a / (a + b)
    v = (a * b) / ((a + b) ** 2 * (a + b + 1))
    return m, np.sqrt(v)

def robust_beta_fit(values, eps=1e-4):
    """
    Fit Beta(alpha, beta) robustly for values in [0,1].
    - Clips values away from exact 0/1 to avoid MLE instability.
    - Falls back to method-of-moments when scipy MLE does not converge.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 1.0, 1.0
    x = np.clip(x, eps, 1.0 - eps)
    if float(np.var(x)) < 1e-12:
        # Nearly constant data -> very concentrated beta around mean.
        m = float(np.mean(x))
        k = 5000.0
        return max(m * k, eps), max((1.0 - m) * k, eps)
    try:
        a, b, _, _ = stats.beta.fit(x, floc=0, fscale=1)
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            return float(a), float(b)
    except Exception:
        pass

    # Method-of-moments fallback.
    m = float(np.mean(x))
    v = float(np.var(x))
    v = min(v, m * (1.0 - m) * 0.99)
    if v <= 0:
        k = 5000.0
        return max(m * k, eps), max((1.0 - m) * k, eps)
    t = (m * (1.0 - m) / v) - 1.0
    if t <= 0:
        return 1.0, 1.0
    a = max(m * t, eps)
    b = max((1.0 - m) * t, eps)
    return a, b

out = {"Pathology": {}}

for target in TARGETS:
    data = stats_per_target[target]
    n = len(data["prob"])
    print(f"\n=== {target} (n={n}) ===")
    if n < 5:
        print("  Too few samples; skipping fit.")
        continue

    a_prob, b_prob = robust_beta_fit(data["prob"])
    a_R,    b_R    = robust_beta_fit(data["R"])
    a_G,    b_G    = robust_beta_fit(data["G"])
    a_B,    b_B    = robust_beta_fit(data["B"])

    out["Pathology"][target] = [[a_prob, b_prob], [a_R, b_R], [a_G, b_G], [a_B, b_B]]

    mine_rows = [("prob", a_prob, b_prob), ("R", a_R, b_R), ("G", a_G, b_G), ("B", a_B, b_B)]
    if hf is not None and "Pathology" in hf and target in hf["Pathology"]:
        hf_entry = hf["Pathology"][target]
        hf_rows = [("prob", *hf_entry[0]), ("R", *hf_entry[1]), ("G", *hf_entry[2]), ("B", *hf_entry[3])]

        print(f"{'stat':<6} {'source':<6} {'alpha':>12} {'beta':>10} {'mean':>8} {'std':>8}")
        print("-" * 60)
        for (name, a_m, b_m), (_, a_h, b_h) in zip(mine_rows, hf_rows):
            m_m, s_m = beta_mean_std(a_m, b_m)
            m_h, s_h = beta_mean_std(a_h, b_h)
            print(f"{name:<6} {'mine':<6} {a_m:>12.3f} {b_m:>10.3f} {m_m:>8.4f} {s_m:>8.4f}")
            print(f"{name:<6} {'HF':<6}   {a_h:>12.3f} {b_h:>10.3f} {m_h:>8.4f} {s_h:>8.4f}")
    else:
        print(f"{'stat':<6} {'source':<6} {'alpha':>12} {'beta':>10} {'mean':>8} {'std':>8}")
        print("-" * 60)
        for name, a_m, b_m in mine_rows:
            m_m, s_m = beta_mean_std(a_m, b_m)
            print(f"{name:<6} {'mine':<6} {a_m:>12.3f} {b_m:>10.3f} {m_m:>8.4f} {s_m:>8.4f}")

# ---- save combined JSON -----------------------------------------------
out_file = "/adialab/usr/roba/lung_target_dist_test.json"
json.dump(out, open(out_file, "w"), indent=2)
print(f"\nSaved: {out_file}")