import os
import logging

import torch
import torch.nn as nn

from utilities.model import align_and_update_state_dicts

from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files

import huggingface_hub

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, opt, module: nn.Module):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.model = module

    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        return outputs

    def save_pretrained(self, save_dir):
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model_state_dict.pt"))

    @staticmethod
    def _checkpoint_has_lora(state_dict: dict) -> bool:
        """Return True if the state dict contains LoRA adapter parameters."""
        return any(k.endswith(".lora_A") or k.endswith(".lora_B") for k in state_dict)

    @staticmethod
    def _infer_lora_config_from_state_dict(state_dict: dict) -> dict:
        """Reconstruct a minimal LoRA config from checkpoint keys.

        Reads the rank (R) from the shape of any lora_A tensor and identifies
        which top-level module names were LoRA-wrapped.
        """
        r = 8
        target_modules = set()
        for key in state_dict:
            if key.endswith(".lora_A"):
                r = state_dict[key].shape[0]
                parts = key.split(".")
                for part in parts:
                    if part in ("lang_encoder", "predictor", "backbone", "pixel_decoder"):
                        target_modules.add(part)
                        break
        return {
            "ENABLED": True,
            "R": r,
            "ALPHA": r * 2,
            "DROPOUT": 0.0,
            "TARGET_MODULES": list(target_modules) if target_modules else ["lang_encoder", "predictor"],
        }

    def from_pretrained(self, pretrained, filename: str = "biomedparse_v1.pt",
                        local_dir: str = "./pretrained", config_dir: str = "./configs"):
        if pretrained.startswith("hf_hub:"):
            hub_name = pretrained.split(":")[1]
            huggingface_hub.hf_hub_download(hub_name, filename=filename, 
                                            local_dir=local_dir)
            huggingface_hub.hf_hub_download(hub_name, filename="config.yaml", 
                                            local_dir=config_dir)
            load_dir = os.path.join(local_dir, filename)
        else:
            load_dir = pretrained

        if os.path.isdir(load_dir):
            load_dir = os.path.join(load_dir, "model_state_dict.pt")

        state_dict = torch.load(load_dir, map_location=self.opt['device'])

        if self._checkpoint_has_lora(state_dict):
            from modeling.utils.lora import apply_lora_to_model
            lora_cfg = self._infer_lora_config_from_state_dict(state_dict)
            n = apply_lora_to_model(self.model, lora_cfg)
            logger.info("Checkpoint contains LoRA weights — applied %d LoRA wrappers "
                        "(r=%d, targets=%s) before loading.", n, lora_cfg["R"],
                        lora_cfg["TARGET_MODULES"])

        state_dict = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        return self