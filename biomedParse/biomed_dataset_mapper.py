"""
datasets/dataset_mappers/biomed_dataset_mapper.py  (MODIFIED)
=============================================================
Drop-in replacement for the original BiomedParse v1 dataset mapper.

Changes vs original:
  1. Reads multi-sentence entries (one sentence per reasoning dimension).
  2. Supports `prompt_dimension` config key to select a single dimension
     at training time (for ablation studies):
         cfg.DATASETS.PROMPT_DIMENSION = "histological"  # or spatial/hierarchical/ambiguity/all
  3. When PROMPT_DIMENSION == "all" (default), randomly samples ONE sentence
     from the available sentences per entry (same behaviour as original single-
     sentence datasets, but leverages multi-dimensional diversity).
  4. When PROMPT_DIMENSION is a specific dimension name, only sentences tagged
     with that dimension are used.  If none are available, falls back to the
     first sentence.
  5. All original image/mask loading and augmentation logic is preserved.

Config additions (biomed_seg_lang_v1.yaml):
    DATASETS:
      PROMPT_DIMENSION: "all"   # "all"|"histological"|"spatial"|"hierarchical"|"ambiguity"
      PROMPT_COMBINE: False     # if True, join all sentences into one long prompt
"""

import copy
import logging
import random
import numpy as np
import torch
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances

logger = logging.getLogger(__name__)

# Reasoning dimension keywords used to match sentences when the JSON was
# generated with prompt_mode="all" and sentences are stored un-tagged.
# If you generated dimension-specific JSONs, this matching is not needed.
DIM_KEYWORDS = {
    "histological": [
        "segment", "cells", "nuclei", "lumen", "gland", "crypt", "epithelial",
        "mucin", "stroma", "goblet", "cytolog", "atyp", "polarity", "architecture",
    ],
    "spatial": [
        "located", "position", "adjacent", "edge", "border", "corner",
        "inferior", "superior", "left", "right", "central", "peripheral",
        "along", "within", "surround",
    ],
    "hierarchical": [
        "largest", "longest", "smallest", "most", "dominant", "main",
        "complex", "extent", "grade", "differentiat",
    ],
    "ambiguity": [
        "distinguish", "differentiate", "unlike", "compared", "identify",
        "unlike", "exclude", "not", "rather than",
    ],
}


def _select_sentence(sentences: list[dict], prompt_dimension: str,
                     combine: bool, rng: random.Random) -> str:
    """
    Select / compose the text prompt from the multi-sentence list.

    sentences: [{"sent": "..."}, ...]  (one per available dimension)

    prompt_dimension:
        "all"   -> random sample from all available sentences
        <dim>   -> heuristically select the sentence most relevant to <dim>;
                   falls back to random if no match.

    combine (bool):
        If True, join all sentences regardless of dimension selection.
    """
    texts = [s["sent"] for s in sentences if s.get("sent")]

    if not texts:
        return ""

    if combine:
        return " ".join(texts)

    if prompt_dimension == "all":
        return rng.choice(texts)

    # Try to match by keyword heuristic
    keywords = DIM_KEYWORDS.get(prompt_dimension, [])
    scored = []
    for t in texts:
        t_lower = t.lower()
        score = sum(1 for kw in keywords if kw in t_lower)
        scored.append((score, t))

    scored.sort(key=lambda x: -x[0])
    best_score, best_text = scored[0]

    if best_score == 0:
        # No keyword match → random fallback
        return rng.choice(texts)

    return best_text


class BiomedDatasetMapper:
    """
    Dataset mapper for BiomedParse v1 with multi-dimensional prompt support.

    Parameters
    ----------
    cfg : CfgNode
        Full Detectron2 config.
    is_train : bool
    augmentations : list of Augmentation
        Geometric augmentations applied to image and mask.
    image_format : str
        "RGB" or "BGR" (default "RGB" for pathology).
    prompt_dimension : str
        One of "all"|"histological"|"spatial"|"hierarchical"|"ambiguity".
        Overrides cfg.DATASETS.PROMPT_DIMENSION if provided directly.
    prompt_combine : bool
        If True, concatenate all dimensions into one prompt.
        Overrides cfg.DATASETS.PROMPT_COMBINE if provided directly.
    """

    def __init__(
        self,
        cfg,
        is_train: bool = True,
        augmentations=None,
        image_format: str = "RGB",
        prompt_dimension: str | None = None,
        prompt_combine: bool | None = None,
    ):
        self.is_train      = is_train
        self.image_format  = image_format
        self.augmentations = T.AugmentationList(augmentations or [])
        self._rng          = random.Random()

        # Prompt dimension from config or explicit argument
        self.prompt_dimension = (
            prompt_dimension
            if prompt_dimension is not None
            else getattr(cfg.DATASETS, "PROMPT_DIMENSION", "all")
        )
        self.prompt_combine = (
            prompt_combine
            if prompt_combine is not None
            else getattr(cfg.DATASETS, "PROMPT_COMBINE", False)
        )

        logger.info(
            f"BiomedDatasetMapper initialised: "
            f"prompt_dimension={self.prompt_dimension}, "
            f"prompt_combine={self.prompt_combine}, "
            f"is_train={is_train}"
        )

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        return cls(
            cfg,
            is_train=is_train,
            augmentations=augs,
            image_format=cfg.INPUT.FORMAT,
        )

    def __call__(self, dataset_dict: dict) -> dict | None:
        """
        Map a single dataset entry to a model-ready dict.

        Returns None if the sample should be skipped (e.g. missing files).
        The DataLoader will discard None entries via a collate filter.
        """
        dataset_dict = copy.deepcopy(dataset_dict)

        # ---- Load image ----
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            logger.warning(f"Could not read image {dataset_dict['file_name']}: {e}")
            return None

        utils.check_image_size(dataset_dict, image)

        # ---- Load mask ----
        mask_path = dataset_dict.get("mask_file_name")
        try:
            mask_img = Image.open(mask_path).convert("L")
            mask = np.array(mask_img)
            mask = (mask > 0).astype(np.uint8)
        except Exception as e:
            logger.warning(f"Could not read mask {mask_path}: {e}")
            return None

        # ---- Geometric augmentations ----
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        # Apply same transforms to mask
        mask = transforms.apply_segmentation(mask)

        # ---- Compose text prompt ----
        sentences = dataset_dict.get("sentences", [{"sent": ""}])
        text = _select_sentence(
            sentences,
            self.prompt_dimension,
            self.prompt_combine,
            self._rng,
        )

        # ---- Build output dict ----
        image_shape = image.shape[:2]  # H, W
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        dataset_dict["text"]  = text
        dataset_dict["prompt_dimension"] = self.prompt_dimension

        # Pack mask into Instances (Detectron2 convention)
        instances = Instances(image_shape)
        gt_masks  = BitMasks(
            torch.stack([torch.from_numpy(mask.copy())], dim=0)
        )
        instances.gt_masks  = gt_masks
        instances.gt_classes = torch.tensor([0], dtype=torch.int64)
        dataset_dict["instances"] = instances

        return dataset_dict
