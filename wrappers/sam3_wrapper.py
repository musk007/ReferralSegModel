"""
SAM3 (Segment Anything Model 3) Wrapper

Meta's SAM3 is a unified foundation model for promptable segmentation that can
detect, segment, and track objects using text or visual prompts.

Requirements:
    pip install torch torchvision
    git clone https://github.com/facebookresearch/sam3.git
    cd sam3 && pip install -e .
    
    # Request access to checkpoints on HuggingFace:
    # https://huggingface.co/facebook/sam3

Usage:
    from sam3_wrapper import SAM3Wrapper
    
    model = SAM3Wrapper(device="cuda")
    mask, confidence = model.segment(image, "liver")
"""

import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional
import sys

from text_seg_eval import TextSegmentationModel


class SAM3Wrapper(TextSegmentationModel):
    """
    Wrapper for SAM3 (Segment Anything Model 3) for text-prompted segmentation.
    
    SAM3 supports:
    - Text prompts (e.g., "a dog", "kidney", "tumor")
    - Visual prompts (points, boxes, masks)
    - Open-vocabulary segmentation
    - Video tracking (not used in this wrapper)
    
    Args:
        device: Device to run on ('cuda' or 'cpu')
        threshold: Segmentation threshold (default: 0.5)
        sam3_path: Path to SAM3 repository (optional)
    """
    
    def __init__(
        self,
        device: str = "cuda",
        threshold: float = 0.5,
        sam3_path: Optional[str] = None,
    ):
        self.device = device
        self.threshold = threshold
        
        # Add SAM3 to path if specified
        if sam3_path:
            sys.path.insert(0, sam3_path)
        
        # Import SAM3 modules
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            self._imports = {
                "build_sam3_image_model": build_sam3_image_model,
                "Sam3Processor": Sam3Processor,
            }
        except ImportError as e:
            raise ImportError(
                f"Failed to import SAM3 modules: {e}\n"
                "Please install SAM3:\n"
                "  git clone https://github.com/facebookresearch/sam3.git\n"
                "  cd sam3 && pip install -e .\n"
                "Also request checkpoint access: https://huggingface.co/facebook/sam3"
            )
        
        # Load model
        print(f"Loading SAM3 model on {device}...")
        self._load_model()
        print("SAM3 model loaded successfully!")
    
    def _load_model(self):
        """Load the SAM3 image model and processor."""
        build_sam3_image_model = self._imports["build_sam3_image_model"]
        Sam3Processor = self._imports["Sam3Processor"]
        
        # Build model (downloads from HuggingFace if needed)
        self.model = build_sam3_image_model()
        
        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model = self.model.eval()
        
        # Create processor
        self.processor = Sam3Processor(self.model)
    
    @torch.inference_mode()
    def segment(
        self, 
        image: Image.Image, 
        text_prompt: str
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Segment the image based on text prompt.
        
        Args:
            image: PIL Image (RGB)
            text_prompt: Text description of object to segment
            
        Returns:
            mask: Binary segmentation mask (H, W), values in {0, 1}
            confidence: Confidence score (mean of scores)
        """
        # Ensure RGB
        image = image.convert("RGB")
        
        # Set image in processor
        inference_state = self.processor.set_image(image)
        
        # Prompt with text
        output = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt
        )
        
        # Extract results
        masks = output["masks"]  # Shape: (N, H, W) for N detected objects
        scores = output["scores"]  # Shape: (N,)
        
        # Handle no detections
        if masks is None or len(masks) == 0:
            h, w = image.size[1], image.size[0]
            return np.zeros((h, w), dtype=np.uint8), 0.0
        
        # Convert to numpy if needed
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        
        # Select best mask (highest score)
        if len(scores) > 0:
            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])
        else:
            best_mask = masks[0]
            best_score = 0.5
        
        # Apply threshold
        binary_mask = (best_mask > self.threshold).astype(np.uint8)
        
        # Ensure correct shape
        if binary_mask.shape != (image.size[1], image.size[0]):
            # Resize if needed
            mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(image.size[::-1], Image.NEAREST)
            binary_mask = (np.array(mask_pil) > 127).astype(np.uint8)
        
        return binary_mask, best_score


class SAM3ImageWrapper(TextSegmentationModel):
    """
    Alternative SAM3 wrapper that works directly with the Sam3Processor API.
    
    This is a simpler interface that may be easier to use in some cases.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        threshold: float = 0.5,
        sam3_path: Optional[str] = None,
    ):
        self.device = device
        self.threshold = threshold
        
        if sam3_path:
            sys.path.insert(0, sam3_path)
        
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            # Build and load model
            print(f"Loading SAM3 model on {device}...")
            model = build_sam3_image_model()
            if device == "cuda" and torch.cuda.is_available():
                model = model.cuda()
            model = model.eval()
            
            # Create processor
            self.processor = Sam3Processor(model)
            print("SAM3 model loaded successfully!")
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import SAM3: {e}\n"
                "Install: git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e ."
            )
    
    @torch.inference_mode()
    def segment(
        self, 
        image: Image.Image, 
        text_prompt: str
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Segment image with text prompt."""
        image = image.convert("RGB")
        
        # Process image and prompt
        state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(state=state, prompt=text_prompt)
        
        # Get best mask
        masks = output.get("masks")
        scores = output.get("scores")
        
        if masks is None or len(masks) == 0:
            return np.zeros((image.size[1], image.size[0]), dtype=np.uint8), 0.0
        
        # Convert to numpy
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        
        # Select best
        best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
        mask = (masks[best_idx] > self.threshold).astype(np.uint8)
        score = float(scores[best_idx]) if len(scores) > 0 else 0.5
        
        # Resize if needed
        if mask.shape != (image.size[1], image.size[0]):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize(image.size[::-1], Image.NEAREST)
            mask = (np.array(mask_img) > 127).astype(np.uint8)
        
        return mask, score


# Example usage
if __name__ == "__main__":
    print("Testing SAM3 wrapper...")
    
    # Create dummy image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    )
    
    try:
        # Test wrapper
        model = SAM3Wrapper(device="cuda" if torch.cuda.is_available() else "cpu")
        mask, conf = model.segment(dummy_image, "object")
        print(f"Mask shape: {mask.shape}, Confidence: {conf:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")
