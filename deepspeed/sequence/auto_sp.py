import logging

import torch.nn as nn

from deepspeed.sequence.autosp_detector import detect_model_sp_info, _VIT_HAS_CLS_TOKEN
from deepspeed.sequence.autosp_vit import UlyssesSPViTAttention
from deepspeed.sequence.ulysses_llm_attention import UlyssesSPLLMAttention

logger = logging.getLogger(__name__)

def auto_wrap_model_for_sp(model: nn.Module, process_group) -> nn.Module:
    info = detect_model_sp_info(model)

    if not info.vit_attn_modules and not info.llm_attn_modules:
        raise ValueError("auto_wrap_model_for_sp: no recognisable attention modules found. "
                         "Add the model's attention class name(s) to "
                         "_VIT_ATTN_CLASSNAMES or _LLM_ATTN_CLASSNAMES in "
                         "deepspeed/sequence/autosp_detector.py and retry.")

    for name, module in info.vit_attn_modules:
        cls_name = type(module).__name__
        has_cls = _VIT_HAS_CLS_TOKEN.get(cls_name, True)
        wrapped = UlyssesSPViTAttention(module, process_group, has_cls_token=has_cls)
        _set_module_by_name(model, name, wrapped)
        logger.debug("AutoSP: wrapped ViT attention '%s' with UlyssesSPViTAttention (has_cls_token=%s)", name, has_cls)

    logger.info("AutoSP: wrapped %d ViT attention layer(s).", len(info.vit_attn_modules))

    for name, module in info.llm_attn_modules:
        wrapped = UlyssesSPLLMAttention(module, process_group)
        _set_module_by_name(model, name, wrapped)
        logger.info("AutoSP: wrapped LLM attention '%s' with UlyssesSPLLMAttention", name)

    logger.info("AutoSP: wrapped %d LLM attention layer(s).", len(info.llm_attn_modules))

    if info.vision_projection_module is not None:
        proj_name, _ = info.vision_projection_module
        logger.warning(
            "AutoSP detected vision projection layer '%s'.  "
            "ModalityFusionSPAdapter (Phase 2) is not yet automated.  "
            "Wrap this layer manually with ModalityFusionSPAdapter if you "
            "need correct cross-modal sequence gather/scatter.", proj_name)

    return model

def _set_module_by_name(model: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
