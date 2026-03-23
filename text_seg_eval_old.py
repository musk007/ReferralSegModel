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
    "accuracy", "boundary_iou", "hausdorff"
]

DEFAULT_METRICS = ["iou", "dice", "precision", "recall", "accuracy"]


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
    """Compute segmentation metrics"""
    if metrics is None:
        metrics = DEFAULT_METRICS
    
    results = {}
    
    # Ensure binary masks
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # Compute intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_and(pred_binary, gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    
    for metric in metrics:
        if metric == "iou":
            if union > 0:
                results["iou"] = intersection / union
            else:
                results["iou"] = 0.0
                
        elif metric == "dice":
            if pred_sum + gt_sum > 0:
                results["dice"] = 2 * intersection / (pred_sum + gt_sum)
            else:
                results["dice"] = 0.0
                
        elif metric == "precision":
            if pred_sum > 0:
                results["precision"] = intersection / pred_sum
            else:
                results["precision"] = 0.0
                
        elif metric == "recall":
            if gt_sum > 0:
                results["recall"] = intersection / gt_sum
            else:
                results["recall"] = 0.0
                
        elif metric == "accuracy":
            correct = np.logical_or(
                np.logical_and(pred_binary, gt_binary),
                np.logical_and(1 - pred_binary, 1 - gt_binary)
            ).sum()
            results["accuracy"] = correct / pred_binary.size
            
        elif metric == "specificity":
            tn = np.logical_and(1 - pred_binary, 1 - gt_binary).sum()
            fp = np.logical_and(pred_binary, 1 - gt_binary).sum()
            if tn + fp > 0:
                results["specificity"] = tn / (tn + fp)
            else:
                results["specificity"] = 0.0
    
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
    
    results = EvaluationResults(model_name=model.__class__.__name__)
    per_sample = {m: [] for m in metrics}
    
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
        
        for metric, value in sample_metrics.items():
            per_sample[metric].append(value)
        
        # Save prediction if requested
        if save_predictions:
            import os
            os.makedirs(save_predictions, exist_ok=True)
            pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
            
            # Use sample ID if available, otherwise use index
            sample_id = sample.get("id", f"sample_{i:04d}")
            pred_img.save(os.path.join(save_predictions, f"{sample_id}_pred.png"))
    
    # Aggregate results
    results.num_samples = len(dataset)
    results.per_sample_metrics = per_sample
    for metric in metrics:
        results.metrics[metric] = np.mean(per_sample[metric])
    
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
