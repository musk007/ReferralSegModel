"""
Model Wrappers for Text-Prompted Segmentation

This package contains wrappers for various segmentation models that implement
the TextSegmentationModel interface for the text_seg_eval framework.

Available wrappers:
- SAM3Wrapper: Meta's Segment Anything 3
- BiomedParseV1Wrapper: BiomedParse v1 (2D multi-modality)
- BiomedParseWrapper: BiomedParse v2 (3D volumetric)
- DualProtoSegWrapper: Histopathology prototype learning
####### new segmentation models #######
- SAM 2
- LLaVA-Med (+ SAM decoder)

Usage:
    from wrappers import SAM3Wrapper, BiomedParseV1Wrapper
    
    model = BiomedParseV1Wrapper(device="cuda")
    mask, conf = model.segment(image, "liver tumor")
"""

# Import available wrappers
_available_wrappers = {}

try:
    from .sam3_wrapper import SAM3Wrapper, SAM3ImageWrapper
    _available_wrappers["SAM3Wrapper"] = SAM3Wrapper
    _available_wrappers["SAM3ImageWrapper"] = SAM3ImageWrapper
except ImportError:
    pass

try:
    from .biomedparse_wrapper import (
        BiomedParseWrapper,
        BiomedParseV1Wrapper,
        get_biomedparse_wrapper,
    )
    _available_wrappers["BiomedParseWrapper"] = BiomedParseWrapper
    _available_wrappers["BiomedParseV1Wrapper"] = BiomedParseV1Wrapper
    _available_wrappers["get_biomedparse_wrapper"] = get_biomedparse_wrapper
except ImportError:
    pass

try:
    from .dualprotoseg_wrapper import DualProtoSegWrapper, DualProtoSegSimpleWrapper
    _available_wrappers["DualProtoSegWrapper"] = DualProtoSegWrapper
    _available_wrappers["DualProtoSegSimpleWrapper"] = DualProtoSegSimpleWrapper
except ImportError:
    pass

# Export available wrappers
__all__ = list(_available_wrappers.keys())

# Make wrappers available at package level
globals().update(_available_wrappers)
