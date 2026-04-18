import logging
import os
import time
import pickle
import torch
import torch.nn as nn

from utilities.distributed import is_main_process

logger = logging.getLogger(__name__)


NORM_MODULES = [
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
]

def register_norm_module(cls):
    NORM_MODULES.append(cls)
    return cls

def align_and_update_state_dicts(model_state_dict, ckpt_state_dict):
    model_keys = sorted(model_state_dict.keys())
    ckpt_keys = sorted(ckpt_state_dict.keys())
    result_dicts = {}
    matched_log = []
    unmatched_log = []
    unloaded_log = []
    for model_key in model_keys:
        model_weight = model_state_dict[model_key]
        ckpt_key = model_key
        if model_key not in ckpt_keys and ".linear." in model_key:
            ckpt_key = model_key.replace(".linear.", ".")
        if ckpt_key in ckpt_keys:
            ckpt_weight = ckpt_state_dict[ckpt_key]
            if model_weight.shape == ckpt_weight.shape:
                result_dicts[model_key] = ckpt_weight
                if ckpt_key in ckpt_keys:
                    ckpt_keys.pop(ckpt_keys.index(ckpt_key))
                matched_log.append("Loaded {}, Model Shape: {} <-> Ckpt Shape: {}".format(model_key, model_weight.shape, ckpt_weight.shape))
        #     else:
        #         unmatched_log.append("*UNMATCHED* {}, Model Shape: {} <-> Ckpt Shape: {}".format(model_key, model_weight.shape, ckpt_weight.shape))
        # else:
        #     unloaded_log.append("*UNLOADED* {}, Model Shape: {}".format(model_key, model_weight.shape))
            
    if is_main_process():
        n_unused = len(ckpt_keys)
        n_unloaded = len(unloaded_log)
        n_unmatched = len(unmatched_log)
        logger.info("Loaded {}/{} weights from checkpoint. "
                     "(unused_ckpt={}, unloaded_model={}, shape_mismatch={})".format(
                         len(matched_log), len(matched_log) + n_unloaded,
                         n_unused, n_unloaded, n_unmatched))
        for info in unloaded_log:
            logger.debug(info)
        for key in ckpt_keys:
            logger.debug("$UNUSED$ {}, Ckpt Shape: {}".format(key, ckpt_state_dict[key].shape))
        for info in unmatched_log:
            logger.debug(info)
    return result_dicts