"""
DualProtoSeg Wrapper

DualProtoSeg is a weakly supervised semantic segmentation method for histopathology
that combines text-guided and image-guided prototype learning using CONCH.

Paper: "DualProtoSeg: Simple and Efficient Design with Text- and Image-Guided
       Prototype Learning for Weakly Supervised Histopathology Image Segmentation"
GitHub: https://github.com/maianhpuco/DualProtoSeg

Requirements:
    git clone https://github.com/maianhpuco/DualProtoSeg.git
    cd DualProtoSeg && pip install -r requirements.txt
    
    Also requires CONCH model:
    pip install conch

Usage:
    from dualprotoseg_wrapper import DualProtoSegWrapper
    
    model = DualProtoSegWrapper(
        dualprotoseg_path="/path/to/DualProtoSeg",
        checkpoint_path="path/to/checkpoint.pth",
        device="cuda"
    )
    mask, confidence = model.segment(image, "tumor")
"""

import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional, Dict, List
import sys
import os
import warnings

from text_seg_eval import TextSegmentationModel


# Default class names and prompts (BCSS). Override with class_names/class_prompts.
DEFAULT_CLASS_NAMES = [
    "tumor",
    "stroma",
    "inflammatory",
    "necrosis",
]

DEFAULT_CLASS_PROMPTS = {
    0: [  # tumor
        "invasive carcinoma",
        "malignant epithelium",
        "pleomorphic nuclei",
    ],
    1: [  # stroma
        "fibrous stroma",
        "collagen bundles",
        "spindle cells",
    ],
    2: [  # inflammatory
        "lymphocytic infiltrate",
        "TILs",
        "plasma cells",
    ],
    3: [  # necrosis
        "tumor necrosis",
        "necrotic debris",
        "ghost cells",
    ],
}


class DualProtoSegWrapper(TextSegmentationModel):
    """
    Wrapper for DualProtoSeg - text and image-guided prototype learning
    for weakly supervised histopathology segmentation.
    
    DualProtoSeg combines:
    - Text-based prototypes from CONCH (pathology vision-language model)
    - Learnable image-based prototypes
    - Multi-scale feature pyramid
    - CAM-based weakly supervised segmentation
    
    Supports multiple datasets by configuring class names and prompts.
    If class_names/class_prompts are not provided, BCSS defaults are used.
    
    Args:
        dualprotoseg_path: Path to DualProtoSeg repository (required)
        checkpoint_path: Path to trained model checkpoint
        config_path: Path to config YAML file
        device: Device to run on ('cuda' or 'cpu')
        threshold: Segmentation threshold (default: 0.5)
        num_classes: Number of segmentation classes (default: len(class_names))
        class_names: List of class names (override BCSS defaults)
        class_prompts: Dict mapping class idx/name -> list of prompts
        default_class: Default class (name or index) if no match
        force_class_idx: If set, always use this class index (for per-image prompts)
        force_class_idx: If set, always use this class index
        use_text_prompt_for_adapter: Use text_prompt to update CONCH prompts
        text_prompt_mode: "replace" or "append" for prompt updates
        prototypes_per_class: Number of prototypes per class (default: 6)
        image_size: Input size for preprocessing (default: 224)
        image_mean: Normalization mean (default: BCSS mean)
        image_std: Normalization std (default: BCSS std)
    """
    
    def __init__(
        self,
        dualprotoseg_path: str = None,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cuda",
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        class_prompts: Optional[Dict] = None,
        default_class: Optional[object] = None,
        force_class_idx: Optional[int] = None,
        use_text_prompt_for_adapter: bool = True,
        text_prompt_mode: str = "replace",
        prototypes_per_class: int = 6,
        image_size: int = 224,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        dataset: Optional[str] = None,
        prompts_config: Optional[str] = None,
        **kwargs,
    ):
        self.device = device
        self.threshold = threshold
        self._dualprotoseg_path = dualprotoseg_path
        self.use_text_prompt_for_adapter = use_text_prompt_for_adapter
        self.text_prompt_mode = text_prompt_mode
        self.prototypes_per_class = int(prototypes_per_class)
        self.force_class_idx = force_class_idx
        self.image_size = int(image_size)
        self.image_mean = (
            np.array(image_mean, dtype=np.float32)
            if image_mean is not None
            else np.array([0.66791496, 0.47791372, 0.70623304], dtype=np.float32)
        )
        self.image_std = (
            np.array(image_std, dtype=np.float32)
            if image_std is not None
            else np.array([0.1736589, 0.22564577, 0.19820057], dtype=np.float32)
        )

        if text_prompt_mode not in {"replace", "append"}:
            raise ValueError("text_prompt_mode must be 'replace' or 'append'")

        # Load prompts from config file if dataset is specified
        if dataset is not None:
            loaded_config = self._load_prompts_config(dataset, prompts_config)
            if loaded_config:
                if class_names is None:
                    class_names = loaded_config.get('class_names')
                if class_prompts is None:
                    class_prompts = loaded_config.get('class_prompts')
                if num_classes is None:
                    num_classes = loaded_config.get('num_classes')

        # Configure classes and prompts
        self.class_names = self._normalize_class_names(class_names, num_classes)
        self.num_classes = len(self.class_names)
        self.class_name_to_idx = {n.lower(): i for i, n in enumerate(self.class_names)}
        self.class_prompts = self._normalize_class_prompts(class_prompts)

        # Default class index if no match
        self.default_class_idx = self._resolve_default_class(default_class)
        
        if dualprotoseg_path is None:
            raise ValueError(
                "dualprotoseg_path is required. Please provide the path to DualProtoSeg repository.\n"
                "Example: DualProtoSegWrapper(dualprotoseg_path='/path/to/DualProtoSeg', ...)"
            )
        
        # Add DualProtoSeg to path
        sys.path.insert(0, dualprotoseg_path)
        
        # Import DualProtoSeg modules
        try:
            from src.model import ClsNetwork
            from src.conch_adapter import ConchAdapter
            from omegaconf import OmegaConf
            
            self._imports = {
                "ClsNetwork": ClsNetwork,
                "ConchAdapter": ConchAdapter,
                "OmegaConf": OmegaConf,
            }
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import DualProtoSeg modules: {e}\n"
                "Please install DualProtoSeg:\n"
                "  git clone https://github.com/maianhpuco/DualProtoSeg.git\n"
                "  cd DualProtoSeg && pip install -r requirements.txt\n"
                "Also install CONCH: pip install conch"
            )
        
        # Load model
        print(f"Loading DualProtoSeg model on {device}...")
        self._load_model(checkpoint_path, config_path)
        print("DualProtoSeg model loaded successfully!")
    
    def _load_prompts_config(
        self, dataset: str, prompts_config: Optional[str] = None
    ) -> Optional[Dict]:
        """Load class prompts from configuration file for the specified dataset."""
        import yaml
        import os
        
        # Default config path
        if prompts_config is None:
            # Try relative path from wrapper directory
            wrapper_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_config = os.path.join(
                os.path.dirname(wrapper_dir), "configs", "class_prompts.yaml"
            )
        
        if not os.path.exists(prompts_config):
            warnings.warn(
                f"Class prompts config file not found at {prompts_config}. "
                f"Using default or provided class_prompts."
            )
            return None
        
        try:
            with open(prompts_config, 'r') as f:
                all_configs = yaml.safe_load(f)
            
            dataset_lower = dataset.lower()
            if dataset_lower not in all_configs:
                available = ', '.join(all_configs.keys())
                warnings.warn(
                    f"Dataset '{dataset}' not found in prompts config. "
                    f"Available: {available}. Using default or provided class_prompts."
                )
                return None
            
            config = all_configs[dataset_lower]
            print(f"Loaded class prompts for dataset: {dataset}")
            print(f"  - Classes: {config['class_names']}")
            print(f"  - Num classes: {config['num_classes']}")
            
            return config
            
        except Exception as e:
            warnings.warn(
                f"Failed to load prompts config from {prompts_config}: {e}. "
                f"Using default or provided class_prompts."
            )
            return None
    
    def _normalize_class_names(
        self, class_names: Optional[List[str]], num_classes: Optional[int]
    ) -> List[str]:
        """Normalize class names; defaults to BCSS if not provided."""
        if class_names is None:
            class_names = DEFAULT_CLASS_NAMES
        elif isinstance(class_names, dict):
            # Dict of {idx: name}
            class_names = [class_names[i] for i in sorted(class_names.keys())]

        if num_classes is not None and num_classes != len(class_names):
            warnings.warn(
                f"num_classes ({num_classes}) does not match class_names length "
                f"({len(class_names)}). Using class_names length."
            )
        return list(class_names)

    def _normalize_class_prompts(self, class_prompts: Optional[Dict]) -> Dict[int, List[str]]:
        """Normalize prompts to {class_idx: [prompt, ...]}."""
        if class_prompts is None:
            if self.class_names == DEFAULT_CLASS_NAMES:
                return {k: list(v) for k, v in DEFAULT_CLASS_PROMPTS.items()}
            return {i: [name] for i, name in enumerate(self.class_names)}

        prompts_out: Dict[int, List[str]] = {i: [] for i in range(self.num_classes)}
        for key, prompts in class_prompts.items():
            if isinstance(key, int):
                idx = key
            else:
                idx = self.class_name_to_idx.get(str(key).lower())
            if idx is None or idx < 0 or idx >= self.num_classes:
                continue
            if isinstance(prompts, str):
                prompts = [prompts]
            prompts_out[idx].extend([str(p) for p in prompts if str(p).strip()])

        # Fallback to class name if no prompts
        for i, name in enumerate(self.class_names):
            if not prompts_out[i]:
                prompts_out[i] = [name]
        return prompts_out

    def _resolve_default_class(self, default_class: Optional[object]) -> int:
        """Resolve default class index when no prompt matches."""
        if default_class is None:
            return 0
        if isinstance(default_class, int) and 0 <= default_class < self.num_classes:
            return default_class
        if isinstance(default_class, str):
            idx = self.class_name_to_idx.get(default_class.lower())
            if idx is not None:
                return idx
        return 0
    
    def _load_model(
        self, 
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """Load the DualProtoSeg model."""
        ClsNetwork = self._imports["ClsNetwork"]
        ConchAdapter = self._imports["ConchAdapter"]
        OmegaConf = self._imports["OmegaConf"]
        
        # Load or create config
        if config_path and os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
        else:
            # Use default config
            default_config = os.path.join(self._dualprotoseg_path, "config.yaml")
            if os.path.exists(default_config):
                cfg = OmegaConf.load(default_config)
            else:
                # Create minimal config
                cfg = OmegaConf.create({
                    "dataset": {"cls_num_classes": self.num_classes},
                    "clip": {"model_name": "conch_ViT-B-16"},
                })
        
        # Initialize CONCH adapter with prompts
        try:
            from conch.open_clip_custom import create_model_from_pretrained
            
            # Determine CONCH checkpoint path
            conch_checkpoint = None
            if hasattr(cfg, 'clip') and hasattr(cfg.clip, 'checkpoint_path'):
                conch_checkpoint = cfg.clip.checkpoint_path
            else:
                # Try default location
                default_conch_path = os.path.join(self._dualprotoseg_path, "conch", "model", "pytorch_model.bin")
                if os.path.exists(default_conch_path):
                    conch_checkpoint = default_conch_path
            
            # Load CONCH model
            model_conch, preprocess = create_model_from_pretrained(
                model_cfg="conch_ViT-B-16",
                checkpoint_path=conch_checkpoint,
                device=torch.device(self.device),
                force_image_size=224,
                cache_dir="",
                hf_auth_token=None,
            )
            model_conch.eval()
            self._conch_model = model_conch
            self._conch_preprocess = preprocess
            
            # Create CONCH adapter
            prompts = self.class_prompts
            conch_checkpoint = "/home/roba/miccai26/DualProtoSeg/conch/model/pytorch_model.bin"
            self.clip_adapter = ConchAdapter(
                checkpoint_path=conch_checkpoint,
                device=self.device,
                class_prompts=prompts,
            )
            
        except Exception as e:
            print(f"Warning: Could not load CONCH model: {e}")
            print("DualProtoSeg requires CONCH for text-guided segmentation.")
            self.clip_adapter = None
            self.model = None
            return
        
        # Create model
        try:
            self.model = ClsNetwork(
                backbone='mit_b1',
                cls_num_classes=self.num_classes,
                clip_adapter=self.clip_adapter,
                stride=[4, 2, 2, 1],
                pretrained=True,
                enable_text_fusion=False,
            )
        except Exception as e:
            print(f"Warning: Could not create ClsNetwork: {e}")
            self.model = None
            return
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
        else:
            warnings.warn(
                "No checkpoint provided. DualProtoSeg model initialized without trained weights.\n"
                "The model will use random weights and may not produce meaningful results."
            )
        
        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model = self.model.eval()
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for DualProtoSeg."""
        # DualProtoSeg uses CONCH-style preprocessing (default 224x224)
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Convert to tensor and normalize (CONCH normalization)
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Dataset-specific normalization
        img_array = (img_array - self.image_mean) / self.image_std
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0)
        
        if self.device == "cuda" and torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        return img_tensor
    
    def _get_class_index(self, text_prompt: str) -> int:
        """Map text prompt to class index using class names and prompts."""
        if self.force_class_idx is not None:
            return int(self.force_class_idx)
        if not text_prompt:
            return self.default_class_idx

        text_lower = text_prompt.lower().strip()

        # Direct class name match
        if text_lower in self.class_name_to_idx:
            return self.class_name_to_idx[text_lower]

        # Substring match against class names
        for name, idx in self.class_name_to_idx.items():
            if name in text_lower:
                return idx

        # Match against provided prompts
        for idx, prompts in self.class_prompts.items():
            for p in prompts:
                p_lower = p.lower()
                if p_lower in text_lower or text_lower in p_lower:
                    return idx

        return self.default_class_idx

    def _build_prompts_for_call(self, text_prompt: str, class_idx: int) -> Dict[int, List[str]]:
        """Build prompts per class for the current text prompt."""
        prompts = {i: list(v) for i, v in self.class_prompts.items()}
        if not text_prompt:
            return prompts

        if self.text_prompt_mode == "replace":
            prompts[class_idx] = [text_prompt]
        else:
            # append mode, keep existing prompts and add the new text
            merged = [text_prompt] + prompts.get(class_idx, [])
            # de-dup while preserving order
            seen = set()
            deduped = []
            for p in merged:
                key = p.lower().strip()
                if key and key not in seen:
                    seen.add(key)
                    deduped.append(p)
            prompts[class_idx] = deduped
        return prompts

    def _maybe_update_clip_adapter(self, text_prompt: str, class_idx: int):
        """Update CONCH prompts to use the provided text prompt."""
        if not self.use_text_prompt_for_adapter or self.clip_adapter is None:
            return

        prompts = self._build_prompts_for_call(text_prompt, class_idx)
        updated = False

        # Try common adapter update methods first
        if hasattr(self.clip_adapter, "update_prompts"):
            try:
                self.clip_adapter.update_prompts(prompts)
                updated = True
            except Exception:
                updated = False
        elif hasattr(self.clip_adapter, "set_prompts"):
            try:
                self.clip_adapter.set_prompts(prompts)
                updated = True
            except Exception:
                updated = False

        # Fallback: reinitialize adapter (DISABLED - causes dimension mismatch)
        # When we recreate CONCH with different class counts, it breaks the model
        # The checkpoint was trained with specific dimensions that must be preserved
        if not updated:
            warnings.warn(
                f"Cannot dynamically update CONCH prompts during inference.\n"
                f"The model was trained with specific class dimensions and "
                f"recreating CONCH would cause dimension mismatches.\n"
                f"Using the original CONCH adapter from checkpoint."
            )
    
    @torch.inference_mode()
    def segment(
        self, 
        image: Image.Image, 
        text_prompt: str
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Segment histopathology image with text prompt using DualProtoSeg.
        
        Args:
            image: PIL Image of histopathology slide
            text_prompt: Text description (e.g., "tumor", "stroma", "necrosis")
            
        Returns:
            mask: Binary segmentation mask (H, W)
            confidence: Average confidence score
        """
        if self.model is None:
            warnings.warn("DualProtoSeg model not loaded. Returning empty mask.")
            return np.zeros((image.size[1], image.size[0]), dtype=np.uint8), 0.0
        
        orig_size = image.size  # (W, H)
        
        # Get class index for the prompt
        class_idx = self._get_class_index(text_prompt)
        self._maybe_update_clip_adapter(text_prompt, class_idx)
        
        # Preprocess image
        img_tensor = self._preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            try:
                outputs = self.model(img_tensor)
                
                # DualProtoSeg returns multiple outputs:
                # (cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, ...)
                # cam4 is at index 7, at 224x224 resolution
                cam4 = outputs[7]  # [B, P, 224, 224] where P = prototypes total
                
            except Exception as e:
                print(f"DualProtoSeg inference error: {e}")
                return np.zeros((orig_size[1], orig_size[0]), dtype=np.uint8), 0.0
        
        # Merge prototype CAMs to class CAMs
        # Get the actual number of prototypes from the model output
        total_prototypes = cam4.shape[1]  # [B, P, 224, 224]
        
        # Default: equal prototypes per class
        proto_per_class = self.prototypes_per_class
        start_idx = class_idx * proto_per_class
        end_idx = start_idx + proto_per_class

        # If model returns per-class prototype counts, use them
        if isinstance(outputs, (list, tuple)) and len(outputs) > 9:
            try:
                k_list = outputs[9]
                if torch.is_tensor(k_list):
                    k_list = k_list.detach().cpu().tolist()
                if isinstance(k_list, (list, tuple)) and len(k_list) > 0:
                    # Use actual model's class count
                    model_num_classes = len(k_list)
                    if class_idx < model_num_classes:
                        proto_per_class = int(k_list[class_idx])
                        start_idx = int(sum(k_list[:class_idx]))
                        end_idx = start_idx + proto_per_class
                    else:
                        # Class index exceeds model's classes - use last class
                        print(f"Warning: class_idx {class_idx} exceeds model classes {model_num_classes}, using last class")
                        class_idx = model_num_classes - 1
                        proto_per_class = int(k_list[class_idx])
                        start_idx = int(sum(k_list[:class_idx]))
                        end_idx = start_idx + proto_per_class
            except Exception as e:
                print(f"Warning: Failed to get prototype counts from model: {e}")
                pass
        
        # Bounds checking
        if end_idx > total_prototypes:
            print(f"Warning: end_idx {end_idx} exceeds total_prototypes {total_prototypes}, clipping")
            end_idx = total_prototypes
        if start_idx >= total_prototypes:
            print(f"Warning: start_idx {start_idx} >= total_prototypes {total_prototypes}, using all prototypes")
            start_idx = 0
            end_idx = total_prototypes
        
        # Average the prototypes for the target class
        class_cam = cam4[0, start_idx:end_idx].mean(dim=0)  # [224, 224]
        
        # Normalize CAM to [0, 1]
        cam_np = class_cam.cpu().numpy()
        cam_np = np.maximum(cam_np, 0)
        cam_max = cam_np.max()
        cam_min = cam_np.min()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
        
        # Apply threshold
        binary_mask = (cam_np > self.threshold).astype(np.uint8)
        
        # Calculate confidence from CAM values BEFORE resizing
        # (cam_np and binary_mask are same size here: 224x224)
        confidence = float(cam_np[binary_mask > 0].mean()) if binary_mask.sum() > 0 else 0.0
        
        # Resize to original size
        if binary_mask.shape != (orig_size[1], orig_size[0]):
            mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(orig_size, Image.NEAREST)
            binary_mask = (np.array(mask_pil) > 127).astype(np.uint8)
        
        return binary_mask, confidence
    
    def get_supported_classes(self) -> List[str]:
        """Get list of supported tissue classes."""
        return list(self.class_names)


class DualProtoSegSimpleWrapper(TextSegmentationModel):
    """
    Simplified DualProtoSeg wrapper that loads from checkpoint directly.
    
    Use this if the full wrapper has dependency issues.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
        class_prompts: Optional[Dict] = None,
        default_class: Optional[object] = None,
        force_class_idx: Optional[int] = None,
        prototypes_per_class: int = 6,
        image_size: int = 224,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        **kwargs,
    ):
        self.device = device
        self.threshold = threshold
        self.prototypes_per_class = int(prototypes_per_class)
        self.force_class_idx = force_class_idx
        self.image_size = int(image_size)
        self.image_mean = (
            np.array(image_mean, dtype=np.float32)
            if image_mean is not None
            else np.array([0.66791496, 0.47791372, 0.70623304], dtype=np.float32)
        )
        self.image_std = (
            np.array(image_std, dtype=np.float32)
            if image_std is not None
            else np.array([0.1736589, 0.22564577, 0.19820057], dtype=np.float32)
        )

        self.class_names = (
            list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
        )
        self.class_name_to_idx = {n.lower(): i for i, n in enumerate(self.class_names)}
        self.class_prompts = (
            self._normalize_class_prompts(class_prompts)
            if class_prompts is not None
            else {i: [name] for i, name in enumerate(self.class_names)}
        )
        self.default_class_idx = self._resolve_default_class(default_class)
        
        print("Loading DualProtoSeg (simple mode)...")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("Checkpoint contains state_dict only. Cannot load in simple mode.")
                print("Use DualProtoSegWrapper with dualprotoseg_path instead.")
                self.model = None
            else:
                self.model = checkpoint
            
            if self.model is not None and hasattr(self.model, 'eval'):
                self.model = self.model.eval()
                if device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.cuda()
            
            print("DualProtoSeg model loaded!")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
    
    @torch.inference_mode()
    def segment(
        self, 
        image: Image.Image, 
        text_prompt: str
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Simple segmentation interface."""
        if self.model is None:
            return np.zeros((image.size[1], image.size[0]), dtype=np.uint8), 0.0
        
        orig_size = image.size
        
        # Simple preprocessing
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = (img_array - self.image_mean) / self.image_std
        
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0)
        
        if self.device == "cuda" and torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        # Determine class index for the text prompt
        class_idx = self._get_class_index(text_prompt)

        try:
            outputs = self.model(img_tensor)
            cam4 = outputs[7]  # [B, 24, 224, 224]
            
            # Merge prototypes for the target class
            proto_per_class = self.prototypes_per_class
            start_idx = class_idx * proto_per_class
            end_idx = start_idx + proto_per_class

            # If model returns per-class prototype counts, use them
            if isinstance(outputs, (list, tuple)) and len(outputs) > 9:
                try:
                    k_list = outputs[9]
                    if torch.is_tensor(k_list):
                        k_list = k_list.detach().cpu().tolist()
                    if isinstance(k_list, (list, tuple)) and len(k_list) >= len(self.class_names):
                        proto_per_class = int(k_list[class_idx])
                        start_idx = int(sum(k_list[:class_idx]))
                        end_idx = start_idx + proto_per_class
                except Exception:
                    pass

            class_cam = cam4[0, start_idx:end_idx].mean(dim=0).cpu().numpy()
            
            # Normalize
            class_cam = np.maximum(class_cam, 0)
            if class_cam.max() > class_cam.min():
                class_cam = (class_cam - class_cam.min()) / (class_cam.max() - class_cam.min())
            
            binary_mask = (class_cam > self.threshold).astype(np.uint8)
            
            # Resize
            if binary_mask.shape != (orig_size[1], orig_size[0]):
                mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize(orig_size, Image.NEAREST)
                binary_mask = (np.array(mask_pil) > 127).astype(np.uint8)
            
            confidence = float(class_cam.mean()) if binary_mask.sum() > 0 else 0.0
            
            return binary_mask, confidence
            
        except Exception as e:
            print(f"Inference error: {e}")
            return np.zeros((orig_size[1], orig_size[0]), dtype=np.uint8), 0.0

    def _normalize_class_prompts(self, class_prompts: Optional[Dict]) -> Dict[int, List[str]]:
        prompts_out: Dict[int, List[str]] = {i: [] for i in range(len(self.class_names))}
        for key, prompts in class_prompts.items():
            if isinstance(key, int):
                idx = key
            else:
                idx = self.class_name_to_idx.get(str(key).lower())
            if idx is None or idx < 0 or idx >= len(self.class_names):
                continue
            if isinstance(prompts, str):
                prompts = [prompts]
            prompts_out[idx].extend([str(p) for p in prompts if str(p).strip()])
        for i, name in enumerate(self.class_names):
            if not prompts_out[i]:
                prompts_out[i] = [name]
        return prompts_out

    def _resolve_default_class(self, default_class: Optional[object]) -> int:
        if default_class is None:
            return 0
        if isinstance(default_class, int) and 0 <= default_class < len(self.class_names):
            return default_class
        if isinstance(default_class, str):
            idx = self.class_name_to_idx.get(default_class.lower())
            if idx is not None:
                return idx
        return 0

    def _get_class_index(self, text_prompt: str) -> int:
        if self.force_class_idx is not None:
            return int(self.force_class_idx)
        if not text_prompt:
            return self.default_class_idx

        text_lower = text_prompt.lower().strip()

        if text_lower in self.class_name_to_idx:
            return self.class_name_to_idx[text_lower]

        for name, idx in self.class_name_to_idx.items():
            if name in text_lower:
                return idx

        for idx, prompts in self.class_prompts.items():
            for p in prompts:
                p_lower = p.lower()
                if p_lower in text_lower or text_lower in p_lower:
                    return idx

        return self.default_class_idx


# Example usage
if __name__ == "__main__":
    print("DualProtoSeg Wrapper")
    print("=" * 60)
    print("Designed for histopathology image segmentation")
    print("Supported classes (default):", DEFAULT_CLASS_NAMES)
    print("=" * 60)
