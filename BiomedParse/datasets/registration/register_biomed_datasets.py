# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Extended to support custom histopathology fine-tuning datasets.
# See README_HISTOPATH.md for setup instructions.
# --------------------------------------------------------
import json
import os
import collections
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Original BiomedParse dataset registration
# ---------------------------------------------------------------------------
# BiomedParseData-Demo: legacy single-demo format (if present under DATASET root)
_PREDEFINED_SPLITS_BIOMED = {}
datasets = ['BiomedParseData-Demo', ]
splits = ['demo']
for name in datasets:
    for split in splits:
        dataname = f'biomed_{name.replace("/", "-")}_{split}'
        image_root = f"{name}/{split}"
        ann_root = f"{name}/{split}.json"
        _PREDEFINED_SPLITS_BIOMED[dataname] = (image_root, ann_root)


def get_metadata():
    meta = {}
    return meta


def load_biomed_json(image_root, annot_json, metadata):
    with PathManager.open(annot_json) as f:
        json_info = json.load(f)

    grd_dict = collections.defaultdict(list)
    for grd_ann in json_info['annotations']:
        image_id = int(grd_ann["image_id"])
        grd_dict[image_id].append(grd_ann)

    mask_root = image_root + '_mask'
    ret = []
    for image in json_info["images"]:
        image_id = int(image["id"])
        image_file = os.path.join(image_root, image['file_name'])
        grounding_anno = grd_dict[image_id]
        for ann in grounding_anno:
            if 'mask_file' not in ann:
                ann['mask_file'] = image['file_name']
            ann['mask_file'] = os.path.join(mask_root, ann['mask_file'])
            ret.append(
                {
                    "file_name": image_file,
                    "image_id": image_id,
                    "grounding_info": [ann],
                }
            )
    assert len(ret), f"No images found in {image_root}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    return ret


def register_biomed(name, metadata, image_root, annot_json):
    DatasetCatalog.register(
        name,
        lambda: load_biomed_json(image_root, annot_json, metadata),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        json_file=annot_json,
        evaluator_type="grounding_refcoco",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_biomed(root):
    for (
        prefix,
        (image_root, annot_root),
    ) in _PREDEFINED_SPLITS_BIOMED.items():
        register_biomed(
            prefix,
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, annot_root),
        )


_root = os.getenv("DATASET", "datasets")
register_all_biomed(_root)


# ---------------------------------------------------------------------------
# Custom histopathology datasets
# ---------------------------------------------------------------------------
# biomedparse_datasets/ lives at  <BiomedParse_root>/../biomedparse_datasets
# This file is at datasets/registration/, so we need three levels up.
# ---------------------------------------------------------------------------

_BIOMEDPARSE_DATASETS_ROOT = os.environ.get(
    "DETECTRON2_DATASETS",
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "biomedparse_datasets")
    ),
)

# Mapping from the registered dataset name to the folder name under
# biomedparse_datasets/.  The config YAML uses the registered name.
_HISTOPATH_DATASETS = {
    "HistoPathColon":    "colon",
    "HistoPathBCSS":     "breast_bcss",
    "HistoPathCells":    "breast_cells",
    "HistoPathLung":     "lung",
    "HistoPathProstate": "prostate",
}

_REASONING_DIMENSIONS = ["histological", "spatial", "hierarchical", "ambiguity"]


def _load_histopath_json(root: str, split: str) -> list[dict]:
    """
    Load a JSON file produced by convert_histopath_to_biomedparse.py and
    return dataset dicts in the format expected by BioMedDatasetMapper:

        {
            "file_name":      <absolute path to image>,
            "image_id":       <int>,
            "grounding_info": [{
                "mask_file":   <absolute path to mask>,
                "sentences":   [{"raw": "...", "sent": "...", "sent_id": N}, ...],
                "category_id": 1,
                "id":          <int>,
            }],
        }
    """
    json_path = os.path.join(root, f"{split}.json")
    img_dir   = os.path.join(root, split)
    mask_dir  = os.path.join(root, f"{split}_mask")

    if not os.path.exists(json_path):
        logger.warning(f"Histopath JSON not found (run convert script first): {json_path}")
        return []

    with open(json_path) as f:
        entries = json.load(f)

    _EXT_FALLBACKS = {".png": ".jpg", ".jpg": ".png", ".jpeg": ".png"}

    def _resolve_path(directory: str, filename: str) -> str:
        """Return the path to `filename` under `directory`, trying alternate
        extensions if the referenced extension is not found on disk."""
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
        stem, ext = os.path.splitext(filename)
        alt_ext = _EXT_FALLBACKS.get(ext.lower())
        if alt_ext:
            alt_path = os.path.join(directory, stem + alt_ext)
            if os.path.exists(alt_path):
                return alt_path
        return path  # return original even if missing (will be caught by mapper)

    dataset_dicts = []
    for i, e in enumerate(entries):
        img_path  = _resolve_path(img_dir,  e["image_name"])
        mask_path = _resolve_path(mask_dir, e["mask_name"])

        # Normalise sentences: add "raw" key if missing (mapper reads ['raw'])
        sentences = []
        for j, s in enumerate(e.get("sentences", [])):
            text = s.get("raw") or s.get("sent", "")
            sentences.append({"raw": text, "sent": text, "sent_id": j})

        dataset_dicts.append({
            "file_name":      img_path,
            "image_id":       i,
            "grounding_info": [{
                "mask_file":   mask_path,
                "sentences":   sentences,
                "category_id": 1,
                "id":          i,
            }],
        })

    return dataset_dicts


def _register_histopath_dataset(registered_name: str, folder_name: str, split: str):
    if registered_name in DatasetCatalog:
        return
    root = os.path.join(_BIOMEDPARSE_DATASETS_ROOT, folder_name)
    DatasetCatalog.register(
        registered_name,
        lambda r=root, s=split: _load_histopath_json(r, s),
    )
    MetadataCatalog.get(registered_name).set(
        image_root=os.path.join(root, split),
        json_file=os.path.join(root, f"{split}.json"),
        evaluator_type="grounding_refcoco",
        ignore_label=255,
        label_divisor=1000,
        dataset_family="histopath",
    )
    logger.info(f"Registered histopath dataset: {registered_name}")


def _register_combined_histopath(split: str):
    """Register HistoPathAll as the union of all individual datasets."""
    combined_name = f"biomed_HistoPathAll_{split}"
    if combined_name in DatasetCatalog:
        return

    component_folders = list(_HISTOPATH_DATASETS.values())

    def _loader(folders=component_folders, s=split):
        all_dicts = []
        for folder in folders:
            root = os.path.join(_BIOMEDPARSE_DATASETS_ROOT, folder)
            all_dicts.extend(_load_histopath_json(root, s))
        return all_dicts

    DatasetCatalog.register(combined_name, _loader)
    MetadataCatalog.get(combined_name).set(
        dataset_family="histopath",
        components=component_folders,
        evaluator_type="grounding_refcoco",
        ignore_label=255,
        label_divisor=1000,
    )
    logger.info(f"Registered combined histopath dataset: {combined_name}")


def _register_all_histopath_datasets():
    for dataset_name, folder_name in _HISTOPATH_DATASETS.items():
        for split in ("train", "eval", "test"):
            registered_name = f"biomed_{dataset_name}_{split}"
            _register_histopath_dataset(registered_name, folder_name, split)

    for split in ("train", "eval", "test"):
        _register_combined_histopath(split)


_register_all_histopath_datasets()


# ---------------------------------------------------------------------------
# BiomedParse official datasets (COCO format: images + annotations)
# Path: biomedparse_datasets/biomedParse/BiomedParseData/<DatasetName>/{train,test}
# ---------------------------------------------------------------------------

_BIOMEDPARSE_OFFICIAL_ROOT = os.path.join(
    os.environ.get(
        "DETECTRON2_DATASETS",
        os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "biomedparse_datasets")
        ),
    ),
    "biomedParse", "BiomedParseData",
)

# Unzipped BiomedParse datasets (each has train/, train.json, train_mask/, test/, test.json, test_mask/)
_BIOMEDPARSE_OFFICIAL_DATASETS = [
    "ACDC", "BreastUS", "CDD-CESM", "DRIVE", "G1020", "ISIC", "LGG",
    "OCT-CME", "PanNuke", "UWaterlooSkinCancer",
]


def _register_biomedparse_official_dataset(dataset_name: str, split: str):
    """Register a single BiomedParse official dataset split (COCO format)."""
    registered_name = f"biomed_{dataset_name}_{split}"
    if registered_name in DatasetCatalog:
        return
    root = os.path.join(_BIOMEDPARSE_OFFICIAL_ROOT, dataset_name)
    image_root = os.path.join(root, split)
    annot_json = os.path.join(root, f"{split}.json")
    if not os.path.isdir(image_root) or not os.path.isfile(annot_json):
        logger.warning(f"BiomedParse {dataset_name}/{split} not found at {root}, skipping")
        return
    register_biomed(
        registered_name,
        get_metadata(),
        image_root,
        annot_json,
    )
    logger.info(f"Registered BiomedParse dataset: {registered_name}")


def _register_all_biomedparse_official():
    """Register all available BiomedParse official datasets."""
    for ds_name in _BIOMEDPARSE_OFFICIAL_DATASETS:
        for split in ("train", "test"):
            _register_biomedparse_official_dataset(ds_name, split)


def _register_combined_biomedparse_official(split: str):
    """Register biomed_BiomedParseAll_{split} = union of all official BiomedParse datasets."""
    combined_name = f"biomed_BiomedParseAll_{split}"
    if combined_name in DatasetCatalog:
        return

    def _loader(s=split):
        all_dicts = []
        for ds_name in _BIOMEDPARSE_OFFICIAL_DATASETS:
            reg_name = f"biomed_{ds_name}_{s}"
            if reg_name in DatasetCatalog:
                try:
                    all_dicts.extend(DatasetCatalog.get(reg_name))
                except Exception as e:
                    logger.warning(f"Could not load {reg_name}: {e}")
        return all_dicts

    DatasetCatalog.register(combined_name, _loader)
    MetadataCatalog.get(combined_name).set(
        evaluator_type="grounding_refcoco",
        ignore_label=255,
        label_divisor=1000,
        dataset_family="biomedparse_official",
    )
    logger.info(f"Registered combined BiomedParse official: {combined_name}")


def _register_combined_full(split: str):
    """Register biomed_CombinedFull_{split} = BiomedParseAll + HistoPathAll (both your data + official)."""
    combined_name = f"biomed_CombinedFull_{split}"
    if combined_name in DatasetCatalog:
        return

    def _loader(s=split):
        all_dicts = []
        # BiomedParse official datasets
        for ds_name in _BIOMEDPARSE_OFFICIAL_DATASETS:
            reg_name = f"biomed_{ds_name}_{s}"
            if reg_name in DatasetCatalog:
                try:
                    all_dicts.extend(DatasetCatalog.get(reg_name))
                except Exception as e:
                    logger.warning(f"Could not load {reg_name}: {e}")
        # Your histopathology datasets
        for folder in _HISTOPATH_DATASETS.values():
            root = os.path.join(_BIOMEDPARSE_DATASETS_ROOT, folder)
            all_dicts.extend(_load_histopath_json(root, s))
        return all_dicts

    DatasetCatalog.register(combined_name, _loader)
    MetadataCatalog.get(combined_name).set(
        evaluator_type="grounding_refcoco",
        ignore_label=255,
        label_divisor=1000,
        dataset_family="combined_full",
    )
    logger.info(f"Registered combined full (BiomedParse + HistoPath): {combined_name}")


_register_all_biomedparse_official()
for s in ("train", "test"):
    _register_combined_biomedparse_official(s)
    _register_combined_full(s)


# ---------------------------------------------------------------------------
# Ablation study helpers
# ---------------------------------------------------------------------------

def register_ablation_json_datasets(dataset_name: str, ablation_json_dir: str):
    """
    Register dimension-specific JSON variants for ablation studies.
    Expects files: <ablation_json_dir>/<split>_<dim>.json
    Generated by: convert_histopath_to_biomedparse.py --prompt_mode <dim>
    """
    folder_name = _HISTOPATH_DATASETS.get(dataset_name, dataset_name)
    root = os.path.join(_BIOMEDPARSE_DATASETS_ROOT, folder_name)

    for dim in _REASONING_DIMENSIONS:
        for split in ("train", "test"):
            registered_name = f"biomed_{dataset_name}_{dim}_{split}"
            if registered_name in DatasetCatalog:
                continue

            json_path = os.path.join(ablation_json_dir, f"{split}_{dim}.json")
            img_dir   = os.path.join(root, split)
            mask_dir  = os.path.join(root, f"{split}_mask")

            def _loader(jp=json_path, id_=img_dir, md=mask_dir):
                if not os.path.exists(jp):
                    logger.warning(f"Ablation JSON not found: {jp}")
                    return []
                with open(jp) as f:
                    entries = json.load(f)
                dicts = []
                for i, e in enumerate(entries):
                    sentences = []
                    for j, s in enumerate(e.get("sentences", [])):
                        text = s.get("raw") or s.get("sent", "")
                        sentences.append({"raw": text, "sent": text, "sent_id": j})
                    dicts.append({
                        "file_name":      os.path.join(id_, e["image_name"]),
                        "image_id":       i,
                        "grounding_info": [{
                            "mask_file":   os.path.join(md, e["mask_name"]),
                            "sentences":   sentences,
                            "category_id": 1,
                            "id":          i,
                        }],
                    })
                return dicts

            DatasetCatalog.register(registered_name, _loader)
            MetadataCatalog.get(registered_name).set(
                dataset_family="histopath",
                prompt_dimension=dim,
            )
            logger.info(f"Registered ablation dataset: {registered_name}")
