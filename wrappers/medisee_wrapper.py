"""
MediSee Wrapper

MediSee is a reasoning-based pixel-level perception model for medical images.
It combines visual understanding with medical reasoning for segmentation.

Paper: "MediSee: Reasoning-based Pixel-level Perception in Medical Images"
GitHub: https://github.com/Edisonhimself/MediSee

Requirements:
    git clone https://github.com/Edisonhimself/MediSee.git
    cd MediSee && pip install -r requirements.txt

Usage:
    from medisee_wrapper import MediSeeWrapper
    
    model = MediSeeWrapper(device="cuda")
    mask, confidence = model.segment(image, "tumor region")
"""

import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional, List
import sys
import os
import warnings

from text_seg_eval import TextSegmentationModel


DEFAULT_MEDISEE_MODEL_DIR = "/home/roba/miccai26/models/medisee"


class MediSeeWrapper(TextSegmentationModel):
    """
    Wrapper for MediSee - reasoning-based pixel-level perception for medical images.
    
    MediSee is designed to:
    - Understand complex medical images through reasoning
    - Segment structures based on textual descriptions
    - Support multiple medical imaging modalities
    
    Args:
        model_path: Path to MediSee model checkpoint
        device: Device to run on ('cuda' or 'cpu')
        threshold: Segmentation threshold (default: 0.5)
        medisee_path: Path to MediSee repository
        model_dir: Directory containing MediSee checkpoints (default: /home/roba/miccai26/models/medisee)
        image_size: Input image size (default: 512)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        threshold: float = 0.5,
        medisee_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        version: Optional[str] = None,
        vision_tower: Optional[str] = None,
        vision_pretrained: Optional[str] = None,
        use_mm_start_end: bool = False,
        conv_type: str = "llava_v1",
        image_size: int = 512,
    ):
        self.device = device
        self.threshold = threshold
        self.image_size = image_size
        self.model_dir = model_dir
        self.version = version
        self.vision_tower = vision_tower
        self.vision_pretrained = vision_pretrained
        self.use_mm_start_end = use_mm_start_end
        self.conv_type = conv_type
        
        # Add MediSee to path if specified
        if medisee_path:
            sys.path.insert(0, medisee_path)
            self._medisee_path = medisee_path
        else:
            self._medisee_path = None
        
        # Import MediSee modules
        try:
            from model.MediSee import MediSeeForCausalLM
            from model.llava import conversation as conversation_lib
            from model.llava.mm_utils import tokenizer_image_token
            from model.segment_anything.utils.transforms import ResizeLongestSide
            from utils.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            from transformers import AutoTokenizer, AutoConfig, MistralConfig, CLIPImageProcessor

            self._imports = {
                "MediSeeForCausalLM": MediSeeForCausalLM,
                "conversation_lib": conversation_lib,
                "tokenizer_image_token": tokenizer_image_token,
                "ResizeLongestSide": ResizeLongestSide,
                "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
                "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
                "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
                "AutoTokenizer": AutoTokenizer,
                "AutoConfig": AutoConfig,
                "MistralConfig": MistralConfig,
                "CLIPImageProcessor": CLIPImageProcessor,
            }
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import MediSee modules: {e}\n"
                "Please install MediSee dependencies:\n"
                "  cd /home/roba/miccai26/MediSee && conda env update -f environment.yml\n"
            )
        
        # Resolve model checkpoint path
        model_path = self._resolve_model_path(model_path, model_dir)

        # Load model
        print(f"Loading MediSee model on {device}...")
        self._load_model(model_path)
        print("MediSee model loaded successfully!")
    
    def _resolve_model_path(self, model_path: Optional[str], model_dir: Optional[str]) -> Optional[str]:
        """Resolve model checkpoint path from provided path or default directory."""
        if model_path:
            if os.path.isdir(model_path):
                return self._find_checkpoint_in_dir(model_path)
            return model_path

        search_dir = model_dir or os.environ.get("MEDISEE_MODEL_DIR") or DEFAULT_MEDISEE_MODEL_DIR
        if search_dir and os.path.isdir(search_dir):
            resolved = self._find_checkpoint_in_dir(search_dir)
            if resolved:
                print(f"Using MediSee checkpoint: {resolved}")
            else:
                warnings.warn(f"No MediSee checkpoint found in {search_dir}")
            return resolved
        return None

    def _find_checkpoint_in_dir(self, directory: str) -> Optional[str]:
        """Find the most recently modified checkpoint in a directory."""
        # Prefer sharded HF-style checkpoints if present
        index_path = os.path.join(directory, "pytorch_model.bin.index.json")
        if os.path.isfile(index_path):
            return directory

        exts = (".pth", ".pt", ".ckpt", ".safetensors")
        candidates = []
        for name in os.listdir(directory):
            if name.lower().endswith(exts):
                path = os.path.join(directory, name)
                if os.path.isfile(path):
                    candidates.append(path)
        if not candidates:
            return None
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    def _load_model(self, model_path: Optional[str] = None):
        """Load the MediSee model."""
        MediSeeForCausalLM = self._imports["MediSeeForCausalLM"]
        AutoTokenizer = self._imports["AutoTokenizer"]
        AutoConfig = self._imports["AutoConfig"]
        MistralConfig = self._imports["MistralConfig"]

        # Resolve base LLM and vision tower paths
        default_version = "/home/roba/miccai26/models/llavamed"
        default_vision_tower = "/home/roba/miccai26/models/clip"
        default_vision_pretrained = "/home/roba/miccai26/models/medsam/medsam_vit_b.pth"

        self.version = self.version or os.environ.get("MEDISEE_LLM_DIR") or default_version
        self.vision_tower = (
            self.vision_tower or os.environ.get("MEDISEE_CLIP_DIR") or default_vision_tower
        )
        self.vision_pretrained = (
            self.vision_pretrained
            or os.environ.get("MEDISEE_SAM_WEIGHTS")
            or default_vision_pretrained
        )

        if not os.path.exists(self.version):
            warnings.warn(f"MediSee base model not found at {self.version}")
        if not os.path.exists(self.vision_tower):
            warnings.warn(f"Vision tower not found at {self.vision_tower}")
        if not os.path.exists(self.vision_pretrained):
            warnings.warn(f"SAM weights not found at {self.vision_pretrained}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.version,
            cache_dir=None,
            model_max_length=1024,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens("[SEG]")
        self.tokenizer.add_tokens("[BBOX]")
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.box_token_idx = self.tokenizer("[BBOX]", add_special_tokens=False).input_ids[0]

        # Model args (match MediSee defaults)
        model_args = {
            "train_mask_decoder": True,
            "out_dim": 256,
            "ce_loss_weight": 1.0,
            "dice_loss_weight": 0.5,
            "bce_loss_weight": 2.0,
            "seg_token_idx": self.seg_token_idx,
            "box_token_idx": self.box_token_idx,
            "vision_pretrained": self.vision_pretrained,
            "vision_tower": self.vision_tower,
            "use_mm_start_end": self.use_mm_start_end,
            "bbox_loss_weight": 2.0,
        }

        # Load config and ensure required fields exist
        cfg = AutoConfig.from_pretrained(self.version)
        if not hasattr(cfg, "hidden_size") or not hasattr(cfg, "vocab_size"):
            cfg = MistralConfig()
            cfg.vocab_size = len(self.tokenizer)
        if not hasattr(cfg, "hidden_size"):
            cfg.hidden_size = 4096
        if not hasattr(cfg, "intermediate_size"):
            cfg.intermediate_size = 14336
        if not hasattr(cfg, "num_attention_heads"):
            cfg.num_attention_heads = 32
        if not hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = 32
        if not hasattr(cfg, "num_key_value_heads"):
            cfg.num_key_value_heads = 8
        if not hasattr(cfg, "mm_vision_select_layer"):
            cfg.mm_vision_select_layer = -2
        if not hasattr(cfg, "mm_vision_select_feature"):
            cfg.mm_vision_select_feature = "patch"
        if not hasattr(cfg, "mm_vision_tower"):
            cfg.mm_vision_tower = self.vision_tower
        if not hasattr(cfg, "mm_hidden_size"):
            cfg.mm_hidden_size = 1024
        if not hasattr(cfg, "mm_projector_type"):
            cfg.mm_projector_type = "linear"
        if not hasattr(cfg, "pad_token_id"):
            cfg.pad_token_id = self.tokenizer.pad_token_id
        if not hasattr(cfg, "bos_token_id"):
            cfg.bos_token_id = self.tokenizer.bos_token_id
        if not hasattr(cfg, "eos_token_id"):
            cfg.eos_token_id = self.tokenizer.eos_token_id
        if hasattr(cfg, "vocab_size") and cfg.vocab_size != len(self.tokenizer):
            cfg.vocab_size = len(self.tokenizer)

        # Load model
        self.model = MediSeeForCausalLM.from_pretrained(
            self.version,
            config=cfg,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
            **model_args,
        )
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Initialize modules
        self.model.get_model().initialize_vision_modules(self.model.get_model().config)
        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch.float32, device=self.device)
        self.model.get_model().initialize_MediSee_modules(self.model.get_model().config)

        # Resize embeddings after adding special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Load checkpoint weights if provided
        if model_path and os.path.isdir(model_path):
            self._load_weight_dir(model_path)
        elif model_path and os.path.isfile(model_path):
            self._load_weight_file(model_path)

        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model = self.model.eval()
        self.model.conversation_lib = self._imports["conversation_lib"]
        self.model.conversation_lib.default_conversation = self.model.conversation_lib.conv_templates[
            self.conv_type
        ]

    def _load_weight_dir(self, weight_dir: str):
        """Load MediSee weights from a directory of sharded or single .bin files."""
        index_path = os.path.join(weight_dir, "pytorch_model.bin.index.json")
        if os.path.isfile(index_path):
            if os.path.getsize(index_path) == 0:
                raise RuntimeError(
                    "MediSee index file is empty. The weights download is incomplete or corrupted. "
                    "Please re-download the weights (e.g., re-extract or run git-lfs pull)."
                )
            try:
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(self.model, weight_dir, strict=False)
                return
            except Exception as e:
                warnings.warn(f"Failed to load sharded checkpoint from {weight_dir}: {e}")

        all_state_dict = {}
        bin_files = [f for f in os.listdir(weight_dir) if f.endswith(".bin")]
        if not bin_files:
            warnings.warn(f"No .bin weights found in {weight_dir}")
            return

        for filename in bin_files:
            path = os.path.join(weight_dir, filename)
            if self._is_lfs_pointer(path):
                warnings.warn(
                    f"Weight file looks like a git-lfs pointer: {path}. "
                    "Run `git lfs pull` in the weights directory."
                )
                continue
            try:
                state_dict = torch.load(path, map_location="cpu")
            except RuntimeError as e:
                msg = str(e)
                if "failed finding central directory" in msg:
                    raise RuntimeError(
                        f"Checkpoint shard is corrupted: {path}. "
                        "Re-download the MediSee weights and ensure all shards are complete."
                    ) from e
                raise
            for key, value in state_dict.items():
                new_key = key.replace(".base_layer", "")
                all_state_dict[new_key] = value
        if all_state_dict:
            self.model.load_state_dict(all_state_dict, strict=False)
        else:
            warnings.warn(f"No usable .bin weights found in {weight_dir}")

    def _load_weight_file(self, weight_file: str):
        """Load MediSee weights from a single checkpoint file."""
        state_dict = torch.load(weight_file, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)

    def _is_lfs_pointer(self, path: str) -> bool:
        """Detect git-lfs pointer files masquerading as weights."""
        try:
            with open(path, "rb") as f:
                head = f.read(200)
            return b"git-lfs" in head or head.startswith(b"version https://git-lfs")
        except Exception:
            return False

    def _build_input_dict(
        self,
        img_tensor: torch.Tensor,
        image_clip: torch.Tensor,
        resize_hw: Tuple[int, int],
        prompt: str,
        dummy_mask: torch.Tensor,
        dummy_label: torch.Tensor,
        dummy_bbox: torch.Tensor,
    ) -> dict:
        """Build minimal input dict for MediSee inference."""
        tokenizer_image_token = self._imports["tokenizer_image_token"]
        DEFAULT_IMAGE_TOKEN = self._imports["DEFAULT_IMAGE_TOKEN"]
        DEFAULT_IM_START_TOKEN = self._imports["DEFAULT_IM_START_TOKEN"]
        DEFAULT_IM_END_TOKEN = self._imports["DEFAULT_IM_END_TOKEN"]

        prompt_text = prompt
        if self.use_mm_start_end:
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            prompt_text = prompt_text.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        input_ids = tokenizer_image_token(prompt_text, self.tokenizer, return_tensors="pt")
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id)
        labels = input_ids.clone()

        input_dict = {
            "image_paths": ["inference_image"],
            "images": img_tensor,
            "images_clip": image_clip.unsqueeze(0),
            "input_ids": input_ids,
            "labels": labels,
            "attention_masks": attention_masks,
            "masks_list": [dummy_mask],
            "label_list": [dummy_label],
            "resize_list": [resize_hw],
            "offset": torch.LongTensor([0, 1]),
            "questions_list": [[prompt]],
            "sampled_classes_list": [[prompt]],
            "bboxes_list": [dummy_bbox],
            "inference": True,
            "conversation_list": [prompt_text],
        }

        if self.device == "cuda" and torch.cuda.is_available():
            input_dict = {k: v.cuda() if torch.is_tensor(v) else v for k, v in input_dict.items()}
        return input_dict
    
    def _preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess image for MediSee input (SAM branch)."""
        # Preprocess image for SAM (ResizeLongestSide + normalize + pad)
        ResizeLongestSide = self._imports["ResizeLongestSide"]
        transform = ResizeLongestSide(1024)
        image_np = np.array(image.convert("RGB"))
        image_resized = transform.apply_image(image_np)
        resize_hw = image_resized.shape[:2]

        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        image_t = torch.from_numpy(image_resized).permute(2, 0, 1).contiguous()
        image_t = (image_t - pixel_mean) / pixel_std
        h, w = image_t.shape[-2:]
        pad_h = 1024 - h
        pad_w = 1024 - w
        image_t = torch.nn.functional.pad(image_t, (0, pad_w, 0, pad_h))
        image_t = image_t.unsqueeze(0)

        if self.device == "cuda" and torch.cuda.is_available():
            image_t = image_t.cuda()
        return image_t, resize_hw
    
    @torch.inference_mode()
    def segment(
        self, 
        image: Image.Image, 
        text_prompt: str
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Segment medical image with text prompt using MediSee.
        
        Args:
            image: PIL Image (RGB)
            text_prompt: Text description of the structure to segment
            
        Returns:
            mask: Binary segmentation mask (H, W)
            confidence: Confidence score
        """
        # Store original size
        orig_size = image.size  # (W, H)
        
        # Preprocess image for SAM
        img_tensor, resize_hw = self._preprocess_image(image)

        # Preprocess image for CLIP
        CLIPImageProcessor = self._imports["CLIPImageProcessor"]
        clip_processor = CLIPImageProcessor.from_pretrained(self.vision_tower)
        image_clip = clip_processor.preprocess(np.array(image.convert("RGB")), return_tensors="pt")[
            "pixel_values"
        ][0]
        if self.device == "cuda" and torch.cuda.is_available():
            image_clip = image_clip.cuda()

        # Build conversation prompt
        conversation_lib = self._imports["conversation_lib"]
        DEFAULT_IMAGE_TOKEN = self._imports["DEFAULT_IMAGE_TOKEN"]
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        question = (
            DEFAULT_IMAGE_TOKEN
            + "\n"
            + f"Please segment and locate the {text_prompt} in this image."
        )
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], "[SEG] and [BBOX].")
        conversations = [conv.get_prompt()]

        # Dummy masks/labels for inference
        h, w = orig_size[1], orig_size[0]
        dummy_mask = torch.zeros((1, h, w), dtype=torch.float32)
        dummy_label = torch.zeros((h, w), dtype=torch.float32)
        dummy_bbox = torch.zeros((1, 4), dtype=torch.float32)
        if self.device == "cuda" and torch.cuda.is_available():
            dummy_mask = dummy_mask.cuda()
            dummy_label = dummy_label.cuda()
            dummy_bbox = dummy_bbox.cuda()

        # Build minimal input dict for inference (avoid heavy dataset imports)
        input_dict = self._build_input_dict(
            img_tensor=img_tensor,
            image_clip=image_clip,
            resize_hw=resize_hw,
            prompt=conversations[0],
            dummy_mask=dummy_mask,
            dummy_label=dummy_label,
            dummy_bbox=dummy_bbox,
        )

        with torch.no_grad():
            output_dict = self.model(**input_dict)

        pred_masks = output_dict["pred_masks"]
        # Take first predicted mask
        mask = pred_masks[0][0]
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Convert to numpy
        if torch.is_tensor(mask):
            mask = mask.squeeze().cpu().numpy()
        
        # Resize to original size
        if binary_mask.shape != (orig_size[1], orig_size[0]):
            mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(orig_size, Image.NEAREST)
            binary_mask = (np.array(mask_pil) > 127).astype(np.uint8)
        
        # Calculate confidence if not provided
        if binary_mask.sum() > 0:
            confidence = float(mask[binary_mask > 0].mean())
        else:
            confidence = 0.0
        
        return binary_mask, confidence


class MediSeeSimpleWrapper(TextSegmentationModel):
    """
    Simplified MediSee wrapper that loads the model directly from checkpoint.
    
    Use this if the full wrapper has dependency issues or for quick testing.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        threshold: float = 0.5,
    ):
        self.device = device
        self.threshold = threshold
        
        print("Loading MediSee model (simple mode)...")
        
        try:
            # Load entire model from checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model = checkpoint['model']
            else:
                self.model = checkpoint
            
            if hasattr(self.model, 'eval'):
                self.model = self.model.eval()
            
            if device == "cuda" and torch.cuda.is_available():
                if hasattr(self.model, 'cuda'):
                    self.model = self.model.cuda()
            
            print("MediSee model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MediSee model: {e}")
    
    @torch.inference_mode()
    def segment(
        self, 
        image: Image.Image, 
        text_prompt: str
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Simple segmentation interface."""
        # Convert to tensor
        image_array = np.array(image.convert("RGB"))
        orig_size = image.size
        
        # Resize
        image_resized = image.resize((512, 512), Image.BILINEAR)
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        if self.device == "cuda" and torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        # Run model
        try:
            output = self.model(img_tensor, text=text_prompt)
            
            # Extract mask
            if isinstance(output, dict):
                mask = output.get('mask') or output.get('masks') or list(output.values())[0]
            elif isinstance(output, tuple):
                mask = output[0]
            else:
                mask = output
            
            # Convert to numpy
            if torch.is_tensor(mask):
                mask = mask.squeeze().cpu().numpy()
            
            # Threshold
            binary_mask = (mask > self.threshold).astype(np.uint8)
            
            # Resize to original
            if binary_mask.shape != (orig_size[1], orig_size[0]):
                mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize(orig_size, Image.NEAREST)
                binary_mask = (np.array(mask_pil) > 127).astype(np.uint8)
            
            return binary_mask, 0.7
            
        except Exception as e:
            print(f"MediSee inference error: {e}")
            return np.zeros((orig_size[1], orig_size[0]), dtype=np.uint8), 0.0


# Example usage
if __name__ == "__main__":
    print("MediSee Wrapper")
    print("Note: Requires MediSee repository to be installed")
    
    # Test with dummy
    try:
        from PIL import Image
        import numpy as np
        
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )
        
        # This will fail without actual MediSee installation
        # model = MediSeeWrapper(device="cpu")
        # mask, conf = model.segment(dummy_image, "tumor")
        # print(f"Mask shape: {mask.shape}, Confidence: {conf}")
        
        print("MediSee wrapper ready for use once repository is installed")
        
    except Exception as e:
        print(f"Setup required: {e}")
