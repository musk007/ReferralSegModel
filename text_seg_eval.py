"""
Text-Prompted Segmentation Evaluation Framework
Minimal version with core classes and functions needed by text_seg_eval_example.py
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from PIL import Image


# Available metrics
AVAILABLE_METRICS = [
    "iou", "dice", "precision", "recall", "specificity",
    "accuracy", "boundary_iou", "hausdorff",
    "giou",   # mean of per-sample IoU  (= mIoU, standard in referring seg)
    "ciou",   # cumulative IoU = Σ intersection / Σ union  (= overall IoU)
]

DEFAULT_METRICS = ["iou", "dice", "precision", "recall", "accuracy", "giou", "ciou"]


@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    metrics: Dict[str, float] = field(default_factory=dict)
    per_sample_metrics: Dict[str, List[float]] = field(default_factory=dict)
    num_samples: int = 0
    model_name: str = ""
    
    def __str__(self):
        lines = [f"Model: {self.model_name}", f"Samples: {self.num_samples}", "Metrics:"]
        for metric, value in self.metrics.items():
            std = np.std(self.per_sample_metrics.get(metric, [0]))
            lines.append(f"  {metric:20s}: {value:.4f} ± {std:.4f}")
        return "\n".join(lines)
    
    def save_json(self, filepath: str):
        """Save evaluation results to JSON file"""
        import json
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        # Prepare data for JSON serialization
        data = {
            "model_name": self.model_name,
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "per_sample_metrics": self.per_sample_metrics,
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'EvaluationResults':
        """Load evaluation results from JSON file"""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            model_name=data.get("model_name", ""),
            num_samples=data.get("num_samples", 0),
            metrics=data.get("metrics", {}),
            per_sample_metrics=data.get("per_sample_metrics", {}),
        )


class TextSegmentationModel(ABC):
    """Abstract base class for text-prompted segmentation models"""
    
    @abstractmethod
    def segment(
        self, 
        image: Image.Image, 
        text_prompt: str
    ) -> Tuple[np.ndarray, float]:
        """
        Segment an image based on a text prompt.
        
        Args:
            image: PIL Image
            text_prompt: Text description of what to segment
            
        Returns:
            binary_mask: Binary segmentation mask (H, W) with values {0, 1}
            confidence: Confidence score (0-1)
        """
        pass


def compute_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    metrics: List[str] = None
) -> Dict[str, float]:
    """
    Compute segmentation metrics using library functions.
    
    IMPORTANT: Metrics focus on FOREGROUND only (excluding background).
    This is achieved by using pos_label=1 in sklearn metrics, which only 
    considers the positive (foreground) class performance.
    
    Uses sklearn for confusion matrix-based metrics and scipy/scikit-image for boundary metrics.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS
    
    from sklearn.metrics import (
        jaccard_score, 
        precision_score, 
        recall_score, 
        f1_score,
        accuracy_score
    )
    
    results = {}

    # Ensure 2D masks (squeeze channels/batch dims)

    pred_mask = np.squeeze(pred_mask)
    gt_mask = np.squeeze(gt_mask)

    if pred_mask.ndim != 2 or gt_mask.ndim != 2:
        gt_mask = gt_mask[:,:,0]
        # raise ValueError(f"Masks must be 2D after squeeze. Got pred {pred_mask.shape}, gt {gt_mask.shape}")


    # Resize pred_mask to match gt_mask shape if different (e.g. model outputs 1024x1024, GT is 2048x2048)
    if pred_mask.shape != gt_mask.shape:
        from scipy.ndimage import zoom
        zoom_factors = (gt_mask.shape[0] / pred_mask.shape[0], gt_mask.shape[1] / pred_mask.shape[1])
        pred_mask = zoom(pred_mask.astype(np.float64), zoom_factors, order=0)
        pred_mask = pred_mask.astype(gt_mask.dtype)

    # Ensure binary masks
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)

    # Flatten arrays for sklearn
    pred_flat = pred_binary.flatten()
    gt_flat = gt_binary.flatten()
    
    for metric in metrics:
        if metric == "iou":
            # Using sklearn's jaccard_score with pos_label=1 (foreground only)
            # This computes IoU for the foreground class only
            
            results["iou"] = jaccard_score(gt_flat, pred_flat, pos_label=1, zero_division=0)
                
        elif metric == "dice":
            # Using sklearn's f1_score with pos_label=1 (foreground only)
            # Dice coefficient for foreground class only
            results["dice"] = f1_score(gt_flat, pred_flat, pos_label=1, zero_division=0)
                
        elif metric == "precision":
            # Precision for foreground: TP / (TP + FP)
            # What fraction of predicted foreground pixels are correct?
            results["precision"] = precision_score(gt_flat, pred_flat, pos_label=1, zero_division=0)
                
        elif metric == "recall":
            # Recall for foreground: TP / (TP + FN)
            # What fraction of ground truth foreground pixels are detected?
            results["recall"] = recall_score(gt_flat, pred_flat, pos_label=1, zero_division=0)
                
        elif metric == "accuracy":
            # For foreground-focused accuracy, we compute it only on union of pred and GT foreground
            # This excludes the background from consideration
            union_mask = np.logical_or(pred_binary, gt_binary)
            if union_mask.sum() > 0:
                # Accuracy computed only on pixels where either pred or GT has foreground
                pred_union = pred_flat[union_mask.flatten()]
                gt_union = gt_flat[union_mask.flatten()]
                results["accuracy"] = accuracy_score(gt_union, pred_union)
            else:
                # Both masks are empty - perfect agreement
                results["accuracy"] = 1.0
            
        elif metric == "specificity":
            # Specificity for foreground segmentation doesn't make much sense
            # as it measures background classification performance (TN / (TN + FP))
            # We'll compute it but note this includes background consideration
            from sklearn.metrics import confusion_matrix
            
            # Handle edge case where both masks are empty or all positive
            if len(np.unique(gt_flat)) == 1:
                # If GT is all negative (0) or all positive (1)
                if gt_flat[0] == 0 and pred_flat.sum() == 0:
                    results["specificity"] = 1.0  # All true negatives
                elif gt_flat[0] == 1:
                    results["specificity"] = 0.0  # No true negatives possible
                else:
                    results["specificity"] = 0.0  # All false positives
            else:
                cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
                tn = cm[0, 0]
                fp = cm[0, 1]
                if tn + fp > 0:
                    results["specificity"] = tn / (tn + fp)
                else:
                    results["specificity"] = 0.0
        
        elif metric == "boundary_iou":
            # Compute boundary IoU using morphological operations
            # This is inherently foreground-focused as boundaries are on foreground objects
            from scipy import ndimage
            
            # Get boundaries using binary erosion
            pred_boundary = pred_binary - ndimage.binary_erosion(pred_binary).astype(np.uint8)
            gt_boundary = gt_binary - ndimage.binary_erosion(gt_binary).astype(np.uint8)
            
            # Compute IoU of boundaries (foreground boundaries only)
            boundary_intersection = np.logical_and(pred_boundary, gt_boundary).sum()
            boundary_union = np.logical_or(pred_boundary, gt_boundary).sum()
            
            if boundary_union > 0:
                results["boundary_iou"] = boundary_intersection / boundary_union
            else:
                results["boundary_iou"] = 0.0
                
        elif metric == "hausdorff":
            # Hausdorff distance computed on foreground pixels only
            from scipy.spatial.distance import directed_hausdorff
            
            # Get coordinates of foreground pixels only
            pred_coords = np.argwhere(pred_binary > 0)
            gt_coords = np.argwhere(gt_binary > 0)
            
            # Handle empty masks
            if len(pred_coords) == 0 or len(gt_coords) == 0:
                if len(pred_coords) == len(gt_coords):
                    results["hausdorff"] = 0.0
                else:
                    results["hausdorff"] = np.inf
            else:
                # Compute bidirectional Hausdorff distance on foreground pixels
                forward_hd = directed_hausdorff(pred_coords, gt_coords)[0]
                backward_hd = directed_hausdorff(gt_coords, pred_coords)[0]
                results["hausdorff"] = max(forward_hd, backward_hd)

        elif metric == "giou":
            # Generalized IoU (gIoU): per-sample intersection / union.
            # The final gIoU score is the mean of these values across all samples,
            # computed in evaluate_model() via np.mean — equivalent to mIoU.
            intersection = float(np.logical_and(pred_binary, gt_binary).sum())
            union = float(np.logical_or(pred_binary, gt_binary).sum())
            results["giou"] = intersection / union if union > 0 else 0.0

        elif metric == "ciou":
            # Cumulative IoU (cIoU): per-sample raw intersection and union are
            # stored as hidden keys so evaluate_model() can sum them globally.
            # Final cIoU = Σ intersection_i / Σ union_i  (overall / global IoU).
            intersection = float(np.logical_and(pred_binary, gt_binary).sum())
            union = float(np.logical_or(pred_binary, gt_binary).sum())
            # Per-sample ratio stored for per_sample_metrics display / JSON export
            results["ciou"] = intersection / union if union > 0 else 0.0
            # Raw components for cumulative aggregation (filtered out of per_sample)
            results["_ciou_I"] = intersection
            results["_ciou_U"] = union

    return results


def evaluate_model(
    model: TextSegmentationModel,
    dataset: List[Dict],
    metrics: List[str] = None,
    verbose: bool = True,
    save_predictions: Optional[str] = None
) -> EvaluationResults:
    """Evaluate a model on a dataset"""
    if metrics is None:
        metrics = DEFAULT_METRICS

    # Hidden internal keys used for cumulative IoU aggregation; never stored in
    # per_sample_metrics or exposed in the final results dict.
    _CIOU_I = "_ciou_I"
    _CIOU_U = "_ciou_U"
    _HIDDEN = {_CIOU_I, _CIOU_U}

    results = EvaluationResults(model_name=model.__class__.__name__)
    per_sample = {m: [] for m in metrics}

    # Accumulators for cumulative IoU
    ciou_I_total = 0.0
    ciou_U_total = 0.0

    for i, sample in enumerate(dataset):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} samples...")

        # Load image and mask from paths if needed
        if "image" in sample:
            image = sample["image"]
        elif "image_path" in sample:
            image = Image.open(sample["image_path"]).convert("RGB")
        else:
            raise KeyError("Sample must contain either 'image' or 'image_path'")

        if "mask" in sample:
            gt_mask = sample["mask"]
        elif "mask_path" in sample:
            gt_mask = np.array(Image.open(sample["mask_path"]))
        else:
            raise KeyError("Sample must contain either 'mask' or 'mask_path'")

        # Get text prompt
        text_prompt = sample.get("prompt") or sample.get("text_prompt", "")

        # Get prediction
        pred_mask, confidence = model.segment(image, text_prompt)

        # Compute metrics
        sample_metrics = compute_metrics(pred_mask, gt_mask, metrics)

        for key, value in sample_metrics.items():
            if key == _CIOU_I:
                ciou_I_total += value
            elif key == _CIOU_U:
                ciou_U_total += value
            elif key in per_sample:
                per_sample[key].append(value)

        # Save prediction if requested
        if save_predictions:
            import os
            os.makedirs(save_predictions, exist_ok=True)
            pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
            sample_id = sample.get("id", f"sample_{i:04d}")
            pred_img.save(os.path.join(save_predictions, f"{sample_id}_pred.png"))

    # Aggregate results
    results.num_samples = len(dataset)
    results.per_sample_metrics = per_sample
    for metric in metrics:
        if metric == "ciou":
            # Global ratio: total intersection over total union across all samples
            results.metrics["ciou"] = ciou_I_total / ciou_U_total if ciou_U_total > 0 else 0.0
        else:
            results.metrics[metric] = np.mean(per_sample[metric]) if per_sample[metric] else 0.0

    return results


def compare_models(
    models: Dict[str, TextSegmentationModel],
    dataset: List[Dict],
    metrics: List[str] = None
) -> Dict[str, EvaluationResults]:
    """Compare multiple models on the same dataset"""
    if metrics is None:
        metrics = DEFAULT_METRICS
    
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_model(model, dataset, metrics)
    
    return results
