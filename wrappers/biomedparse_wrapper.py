"""
BiomedParse Model Wrapper for text_seg_eval

This wrapper integrates the BiomedParse v2 model for text-prompted biomedical
image segmentation into the text_seg_eval evaluation framework.

BiomedParse v2 supports 3D volumetric segmentation across multiple modalities
including CT, MRI, Ultrasound, PET, and 3D Microscopy.

Requirements:
    - BiomedParse repository cloned
    - Model weights downloaded from HuggingFace
    - Dependencies installed (see BiomedParse README)
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Optional
import sys
import os

# Mock MPI to prevent initialization errors in single-GPU inference
# BiomedParse imports mpi4py in trainer modules, but we don't need MPI for inference
import unittest.mock as mock

class MockMPI:
    """Mock MPI class to prevent real MPI initialization"""
    COMM_WORLD = mock.Mock()
    def __init__(self):
        self.COMM_WORLD.Get_rank = lambda: 0
        self.COMM_WORLD.Get_size = lambda: 1
        self.COMM_WORLD.barrier = lambda: None
        
# Mock mpi4py before BiomedParse modules import it
sys.modules['mpi4py'] = mock.Mock()
sys.modules['mpi4py.MPI'] = MockMPI()

# Add BiomedParse to path if needed
# sys.path.insert(0, "/path/to/BiomedParse")

from text_seg_eval import TextSegmentationModel


class BiomedParseWrapper(TextSegmentationModel):
    """
    Wrapper for BiomedParse v2 model (3D volumetric segmentation).
    
    BiomedParse v2 is designed for 3D medical imaging modalities and processes
    volumes in a slice-by-slice manner with neighboring 3D context.
    
    Supported modalities:
        - CT, MRI, Ultrasound, PET
        - 3D Microscopy (EM, lightsheet)
    
    Requirements:
        pip install huggingface_hub
        git clone https://github.com/microsoft/BiomedParse.git
        # Follow installation instructions in BiomedParse README
    
    Args:
        model_checkpoint: Path to model checkpoint or HuggingFace repo ID
        device: Device to run inference on ('cuda' or 'cpu')
        threshold: Segmentation threshold (default: 0.5)
        image_size: Input image size for preprocessing (default: 512)
        slice_batch_size: Number of slices to process in batch (default: 4)
    """
    
    def __init__(
        self,
        model_checkpoint: str = "microsoft/BiomedParse",
        device: str = "cuda",
        threshold: float = 0.5,
        image_size: int = 512,
        slice_batch_size: int = 4,
        biomedparse_path: Optional[str] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.image_size = image_size
        self.slice_batch_size = slice_batch_size
        
        # Add BiomedParse to path if specified
        if biomedparse_path:
            sys.path.insert(0, biomedparse_path)
        
        # Import BiomedParse modules
        try:
            import hydra
            from hydra import compose
            from hydra.core.global_hydra import GlobalHydra
            from huggingface_hub import hf_hub_download
            
            # Store imports for later use
            self._imports = {
                "hydra": hydra,
                "compose": compose,
                "GlobalHydra": GlobalHydra,
                "hf_hub_download": hf_hub_download,
            }
            
            # Import processing utilities
            from utils import process_input, process_output
            from inference import postprocess, merge_multiclass_masks
            
            self._processing = {
                "process_input": process_input,
                "process_output": process_output,
                "postprocess": postprocess,
                "merge_multiclass_masks": merge_multiclass_masks,
            }
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import BiomedParse modules: {e}\n"
                "Make sure BiomedParse is installed and in your Python path.\n"
                "See: https://github.com/microsoft/BiomedParse"
            )
        
        # Initialize model
        print(f"Loading BiomedParse model on {self.device}...")
        self._load_model(model_checkpoint)
        print("Model loaded successfully!")
    
    def _load_model(self, model_checkpoint: str):
        """Load the BiomedParse v2 model."""
        hydra = self._imports["hydra"]
        compose = self._imports["compose"]
        GlobalHydra = self._imports["GlobalHydra"]
        hf_hub_download = self._imports["hf_hub_download"]
        
        # Clear any existing hydra instance
        GlobalHydra.instance().clear()
        
        # Initialize hydra with BiomedParse config
        # Note: Assumes configs are in the BiomedParse directory
        hydra.initialize(config_path="configs/model", job_name="biomedparse_inference")
        cfg = compose(config_name="biomedparse_3D")
        
        # Build model
        self.model = hydra.utils.instantiate(cfg, _convert_="object")
        
        # Load pretrained weights
        if model_checkpoint.startswith("microsoft/") or "/" in model_checkpoint:
            # Download from HuggingFace
            checkpoint_path = hf_hub_download(
                repo_id=model_checkpoint.split(":")[0] if ":" in model_checkpoint else model_checkpoint,
                filename="biomedparse_v2.ckpt"
            )
        else:
            # Local checkpoint
            checkpoint_path = model_checkpoint
        
        self.model.load_pretrained(checkpoint_path)
        self.model = self.model.to(self.device).eval()
    
    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL image to numpy array suitable for BiomedParse.
        
        For 2D images, we need to create a pseudo-3D volume by stacking
        the image as a single slice with neighboring context.
        """
        # Convert to numpy array
        img_array = np.array(image.convert("RGB"))
        
        # For 2D images, create a single-slice volume
        # Shape: (H, W, 3) -> (1, H, W, 3)
        if img_array.ndim == 3:
            img_array = img_array[np.newaxis, ...]
        
        return img_array
    
    @torch.inference_mode()
    def segment(
        self, 
        image: Image.Image, 
        text_prompt: str
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Segment the image based on the text prompt.
        
        Args:
            image: PIL Image (RGB)
            text_prompt: Text description of the object to segment
            
        Returns:
            mask: Binary segmentation mask (H, W), values in {0, 1}
            confidence: Confidence score (or None)
        """
        process_input = self._processing["process_input"]
        process_output = self._processing["process_output"]
        postprocess = self._processing["postprocess"]
        merge_multiclass_masks = self._processing["merge_multiclass_masks"]
        
        # Prepare image
        imgs = self._prepare_image(image)
        
        # Process input (handles padding and resizing)
        imgs_processed, pad_width, padded_size, valid_axis = process_input(
            imgs, self.image_size
        )
        
        # Move to device
        imgs_tensor = torch.from_numpy(imgs_processed).to(self.device).int()
        
        # Prepare input dict
        input_tensor = {
            "image": imgs_tensor.unsqueeze(0),  # Add batch dimension
            "text": [text_prompt],
        }
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor, mode="eval", slice_batch_size=self.slice_batch_size)
        
        # Extract predictions
        mask_preds = output["predictions"]["pred_gmasks"]
        
        # Resize to target size
        mask_preds = F.interpolate(
            mask_preds, 
            size=(self.image_size, self.image_size), 
            mode="bicubic", 
            align_corners=False, 
            antialias=True
        )
        
        # Postprocess (handles object existence detection)
        mask_preds = postprocess(
            mask_preds, 
            output["predictions"]["object_existence"]
        )
        
        # For single class, merge is identity operation
        ids = [0]  # Single object
        mask_preds = merge_multiclass_masks(mask_preds, ids)
        
        # Process output (undo padding)
        mask_preds = process_output(mask_preds, pad_width, padded_size, valid_axis)
        
        # Convert to binary mask
        # For 2D input, we have a single slice
        if mask_preds.ndim == 3:
            mask = mask_preds[0]  # Take first slice
        else:
            mask = mask_preds
        
        # Apply threshold
        binary_mask = (mask > self.threshold).astype(np.uint8)
        
        # Calculate confidence (mean of positive predictions)
        if binary_mask.sum() > 0:
            confidence = float(mask[binary_mask > 0].mean())
        else:
            confidence = 0.0
        
        # Resize to original image size if needed
        if binary_mask.shape != (image.size[1], image.size[0]):
            mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(image.size[::-1], Image.NEAREST)
            binary_mask = (np.array(mask_pil) > 127).astype(np.uint8)
        
        return binary_mask, confidence


class BiomedParseV1Wrapper(TextSegmentationModel):
    """
    Wrapper for BiomedParse v1 model (2D multi-modality segmentation).
    
    BiomedParse v1 supports 2D imaging across 9 modalities:
        - CT, MRI, Ultrasound, X-Ray
        - Pathology, Endoscopy, Dermoscopy
        - Fundus, OCT
    
    This wrapper is for the original BiomedParse model (v1 branch).
    
    Requirements:
        git clone https://github.com/microsoft/BiomedParse.git
        git checkout v1  # Switch to v1 branch
        # Follow v1 installation instructions
    
    Args:
        model_checkpoint: Path to model checkpoint or 'hf_hub:microsoft/BiomedParse'
        device: Device to run inference on ('cuda' or 'cpu')
        threshold: Segmentation threshold (default: 0.5)
        biomedparse_path: Path to BiomedParse repository (optional)
    """
    
    def __init__(
        self,
        model_checkpoint: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        threshold: float = 0.5,
        biomedparse_path: Optional[str] = None,
    ):
        self.device = device
        self.threshold = threshold
        self._biomedparse_path = biomedparse_path
        # Resolve checkpoint: explicit model_checkpoint, else checkpoint_path, else pretrained
        ckpt = model_checkpoint or checkpoint_path or "hf_hub:microsoft/BiomedParse"
        self._model_checkpoint = self._resolve_checkpoint_path(ckpt)

        # Add BiomedParse to path if specified
        if biomedparse_path:
            sys.path.insert(0, biomedparse_path)
        
        # Import BiomedParse v1 modules
        try:
            from modeling.BaseModel import BaseModel
            from modeling import build_model
            from utilities.distributed import init_distributed
            from utilities.arguments import load_opt_from_config_files
            from inference_utils.inference import interactive_infer_image
            
            self._imports = {
                "BaseModel": BaseModel,
                "build_model": build_model,
                "init_distributed": init_distributed,
                "load_opt_from_config_files": load_opt_from_config_files,
                "interactive_infer_image": interactive_infer_image,
            }
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import BiomedParse v1 modules: {e}\n"
                "Make sure BiomedParse v1 is installed and in your Python path.\n"
                "git clone https://github.com/microsoft/BiomedParse.git\n"
                "git checkout v1"
            )
        
        # Load model
        print(f"Loading BiomedParse v1 model on {device}...")
        self._load_model(self._model_checkpoint)
        print("Model loaded successfully!")

    def _resolve_checkpoint_path(self, path: str) -> str:
        """If path is a directory, resolve to model_state_dict.pt inside default/ or direct."""
        if path.startswith("hf_hub:"):
            return path
        p = os.path.abspath(path)
        if os.path.isfile(p):
            return p
        if os.path.isdir(p):
            for sub in ("default", "."):
                candidate = os.path.join(p, sub, "model_state_dict.pt") if sub != "." else os.path.join(p, "model_state_dict.pt")
                if os.path.isfile(candidate):
                    return candidate
        return p

    def _load_model(self, model_checkpoint: str):
        """Load the BiomedParse v1 model."""
        load_opt_from_config_files = self._imports["load_opt_from_config_files"]
        init_distributed = self._imports["init_distributed"]
        BaseModel = self._imports["BaseModel"]
        build_model = self._imports["build_model"]
        
        # Import BIOMED_CLASSES for text embedding initialization
        from utilities.constants import BIOMED_CLASSES
        
        # Fine-tuned checkpoints use seem_model_v1 (768-dim text); biomedparse_inference
        # builds seem_model_demo (512-dim) which causes shape mismatch. Use v1 config
        # for local/fine-tuned checkpoints. For hf_hub pretrained, use demo if v1 fails.
        use_v1_config = not model_checkpoint.startswith("hf_hub:")
        config_name = "biomed_seg_lang_v1_inference.yaml" if use_v1_config else "biomedparse_inference.yaml"
        config_path = None
        if self._biomedparse_path:
            config_path = os.path.join(self._biomedparse_path, "configs", config_name)
        else:
            search_paths = [
                os.path.join("configs", config_name),
                os.path.join("/home/roba/miccai26/BiomedParse/configs", config_name),
            ]
            for path in search_paths:
                if os.path.exists(path):
                    config_path = path
                    break

        if config_path is None or not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Could not find biomedparse_inference.yaml config file. "
                f"Please set biomedparse_path parameter to the BiomedParse repository path."
            )
        
        # Load config
        opt = load_opt_from_config_files([config_path])
        opt = init_distributed(opt)
        
        # Build and load model
        self.model = BaseModel(opt, build_model(opt)).from_pretrained(
            model_checkpoint
        ).eval()
        
        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Initialize text embeddings for all biomedical classes (required for v1)
        with torch.no_grad():
            self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                BIOMED_CLASSES + ["background"], is_eval=True
            )
    
    @torch.inference_mode()
    def segment(
        self, 
        image: Image.Image, 
        text_prompt: str
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Segment the image based on the text prompt.
        
        Args:
            image: PIL Image (RGB)
            text_prompt: Text description of the object to segment
            
        Returns:
            mask: Binary segmentation mask (H, W), values in {0, 1}
            confidence: Confidence score (mean prediction value)
        """
        interactive_infer_image = self._imports["interactive_infer_image"]
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        
        # Run inference
        # interactive_infer_image returns a list of masks (one per prompt)
        
        pred_masks = interactive_infer_image(self.model, image, [text_prompt])
        
        # Get the first (and only) mask
        pred_mask = pred_masks[0]
        
        # Apply threshold to get binary mask
        binary_mask = (pred_mask > self.threshold).astype(np.uint8)
        
        # Calculate confidence
        if binary_mask.sum() > 0:
            confidence = float(pred_mask[binary_mask > 0].mean())
        else:
            confidence = 0.0
        
        return binary_mask, confidence


# Convenience function to get the appropriate wrapper
def get_biomedparse_wrapper(
    version: str = "v2",
    model_checkpoint: Optional[str] = None,
    device: str = "cuda",
    threshold: float = 0.5,
    **kwargs
) -> TextSegmentationModel:
    """
    Get the appropriate BiomedParse wrapper based on version.
    
    Args:
        version: Model version ('v1' or 'v2')
        model_checkpoint: Path to checkpoint (uses default if None)
        device: Device to run on
        threshold: Segmentation threshold
        **kwargs: Additional arguments for the wrapper
        
    Returns:
        BiomedParse wrapper instance
    """
    if version == "v1":
        if model_checkpoint is None:
            model_checkpoint = "hf_hub:microsoft/BiomedParse"
        return BiomedParseV1Wrapper(
            model_checkpoint=model_checkpoint,
            device=device,
            threshold=threshold,
            **kwargs
        )
    elif version == "v2":
        if model_checkpoint is None:
            model_checkpoint = "microsoft/BiomedParse"
        return BiomedParseWrapper(
            model_checkpoint=model_checkpoint,
            device=device,
            threshold=threshold,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown version: {version}. Choose 'v1' or 'v2'")


# # Example usage
# if __name__ == "__main__":
#     import sys
    
#     # Example: Test wrapper with dummy image
#     print("Testing BiomedParse wrapper...")
    
#     # Create dummy image
#     dummy_image = Image.fromarray(
#         np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
#     )
    
#     # Test v2 wrapper (comment out if v2 not installed)
#     try:
#         print("\nTesting v2 wrapper...")
#         model_v2 = get_biomedparse_wrapper(
#             version="v2",
#             device="cuda" if torch.cuda.is_available() else "cpu"
#         )
#         mask, conf = model_v2.segment(dummy_image, "liver")
#         print(f"V2 - Mask shape: {mask.shape}, Confidence: {conf:.4f}")
#     except Exception as e:
#         print(f"V2 test failed: {e}")
    
#     # Test v1 wrapper (comment out if v1 not installed)
#     try:
#         print("\nTesting v1 wrapper...")
#         model_v1 = get_biomedparse_wrapper(
#             version="v1",
#             device="cuda" if torch.cuda.is_available() else "cpu"
#         )
#         mask, conf = model_v1.segment(dummy_image, "liver")
#         print(f"V1 - Mask shape: {mask.shape}, Confidence: {conf:.4f}")
#     except Exception as e:
#         print(f"V1 test failed: {e}")
