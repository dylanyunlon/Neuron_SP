import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_VIT_ATTN_CLASSNAMES = {
    "ViTAttention",
    "CLIPAttention",
    "SiglipAttention",
    "InternVisionAttention",
    "Qwen2VLVisionAttention",
    "Idefics2VisionAttention",
    "PaliGemmaVisionAttention",
}

_VIT_HAS_CLS_TOKEN = {
    "ViTAttention": True,
    "CLIPAttention": True,
    "SiglipAttention": False,
    "InternVisionAttention": False,
    "Qwen2VLVisionAttention": False,
    "Idefics2VisionAttention": False,
    "PaliGemmaVisionAttention": False,
}

_LLM_ATTN_CLASSNAMES = {
    "LlamaAttention",
    "MistralAttention",
    "Qwen2Attention",
    "InternLM2Attention",
    "GemmaAttention",
    "GroupedQueryAttention",
    "Phi3Attention",
    "GPTNeoXAttention",
    "FalconAttention",
    "MptAttention",
}

_VISION_PROJ_KEYWORDS = (
    "visual_projection",
    "mm_projector",
    "vision_proj",
    "multi_modal_projector",
    "img_projection",
)

@dataclass
class SPModelInfo:

    vit_attn_modules: List[Tuple[str, nn.Module]] = field(default_factory=list)
    llm_attn_modules: List[Tuple[str, nn.Module]] = field(default_factory=list)
    vision_projection_module: Optional[Tuple[str, nn.Module]] = None

def detect_model_sp_info(model: nn.Module) -> SPModelInfo:
    info = SPModelInfo()
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name in _VIT_ATTN_CLASSNAMES:
            info.vit_attn_modules.append((name, module))
        elif cls_name in _LLM_ATTN_CLASSNAMES:
            info.llm_attn_modules.append((name, module))

        if info.vision_projection_module is None:
            if any(kw in name for kw in _VISION_PROJ_KEYWORDS):
                info.vision_projection_module = (name, module)

    return info
