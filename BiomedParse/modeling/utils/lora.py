# --------------------------------------------------------
# LoRA (Low-Rank Adaptation) for BiomedParse fine-tuning
# Wraps nn.Linear layers with low-rank adapters.
# --------------------------------------------------------

import logging
import torch
import torch.nn as nn
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class LinearLoRA(nn.Module):
    """LoRA wrapper for nn.Linear. Output = W @ x + (alpha/r) * (B @ A) @ x."""

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        in_features = linear.in_features
        out_features = linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        # Freeze original linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    # nn.Linear compatibility: MultiheadAttention and other code access .weight / .bias
    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.linear.bias

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * lora_out


def _module_name_matches(module_name: str, targets: List[str]) -> bool:
    """Check if module_name contains any of the target substrings."""
    for t in targets:
        if t in module_name:
            return True
    return False


def apply_lora_to_model(
    model: nn.Module,
    lora_config: dict,
    prefix: str = "",
) -> int:
    """
    Replace nn.Linear layers in target modules with LinearLoRA.

    Args:
        model: The model to modify (in-place).
        lora_config: Dict with ENABLED, R, ALPHA, DROPOUT, TARGET_MODULES.
        prefix: Prefix for module names (e.g. "model." when model is inside BaseModel).

    Returns:
        Number of Linear layers replaced with LoRA.
    """
    if not lora_config.get("ENABLED", False):
        return 0

    r = lora_config.get("R", 8)
    alpha = lora_config.get("ALPHA", 16.0)
    dropout = lora_config.get("DROPOUT", 0.0)
    target_modules = lora_config.get("TARGET_MODULES", ["lang_encoder", "predictor"])
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    replaced = 0
    for name, module in list(model.named_modules()):
        full_name = f"{prefix}{name}" if prefix else name
        if not _module_name_matches(full_name, target_modules):
            continue
        if isinstance(module, nn.Linear):
            # Avoid double-wrap; LinearLoRA.linear is the original
            if hasattr(module, "linear") and isinstance(getattr(module, "linear", None), nn.Linear):
                continue
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            child_name = name.rsplit(".", 1)[-1] if "." in name else name
            try:
                parent = model.get_submodule(parent_name) if parent_name else model
                lora_linear = LinearLoRA(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_linear)
                replaced += 1
                if logger.isEnabledFor(logging.INFO):
                    # logger.info(f"LoRA: replaced {full_name} (in={module.in_features}, out={module.out_features}, r={r})")
                    pass
            except Exception as e:
                logger.warning(f"LoRA: failed to replace {full_name}: {e}")
    return replaced
