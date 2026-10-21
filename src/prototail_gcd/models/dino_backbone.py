"""DINO/DINOv2 loading and fine-tuning helpers shared across adapters."""

import re
from typing import Any

import torch


BACKBONE_SPECS = {
    "dino_vitb16": {"feature_dim": 768, "patch_size": 16, "image_size": 224},
    "dinov2_vitb14": {"feature_dim": 768, "patch_size": 14, "image_size": 224},
    "dinov2_vitb14_reg": {
        "feature_dim": 768,
        "patch_size": 14,
        "image_size": 224,
    },
}

HUB_SOURCES = {
    "dino_vitb16": ("facebookresearch/dino:main", "dino_vitb16"),
    "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14"),
    "dinov2_vitb14_reg": ("facebookresearch/dinov2", "dinov2_vitb14_reg"),
}


def load_backbone(name: str, pretrained: bool = True) -> Any:
    if name not in HUB_SOURCES:
        available = tuple(HUB_SOURCES)
        raise ValueError(
            f"Unknown backbone {name!r}; choose from {available}"
        )

    repository, entrypoint = HUB_SOURCES[name]
    if name.startswith("dinov2_"):
        return torch.hub.load(repository, entrypoint, pretrained=pretrained)
    return torch.hub.load(repository, entrypoint)


def backbone_spec(name: str) -> dict[str, int]:
    try:
        return dict(BACKBONE_SPECS[name])
    except KeyError as error:
        raise ValueError(f"Unknown backbone {name!r}") from error


def _prepare_tokens(backbone: Any, images: torch.Tensor) -> torch.Tensor:
    if hasattr(backbone, "prepare_tokens_with_masks"):
        return backbone.prepare_tokens_with_masks(images)
    return backbone.prepare_tokens(images)


def _iter_blocks(backbone: Any):
    for block_group in backbone.blocks:
        if isinstance(block_group, torch.nn.ModuleList):
            for block in block_group:
                if not isinstance(block, torch.nn.Identity):
                    yield block
        else:
            yield block_group


def forward_global_and_patches(
    backbone: Any,
    images: torch.Tensor,
    *,
    finetune_from_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = _prepare_tokens(backbone, images)
    for block_index, block in enumerate(_iter_blocks(backbone)):
        if block_index < finetune_from_block:
            with torch.no_grad():
                tokens = block(tokens)
        else:
            tokens = block(tokens)

    tokens = backbone.norm(tokens)
    register_count = int(getattr(backbone, "num_register_tokens", 0) or 0)
    global_feature = tokens[:, 0]
    patch_tokens = tokens[:, 1 + register_count:]
    return global_feature, patch_tokens


def parse_block_index(parameter_name: str) -> int | None:
    if "block" not in parameter_name:
        return None
    indices = re.findall(r"\.(\d+)(?=\.|$)", parameter_name)
    return int(indices[-1]) if indices else None


def set_finetune_blocks(backbone: Any, start_block: int) -> None:
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    for name, parameter in backbone.named_parameters():
        block_index = parse_block_index(name)
        if block_index is not None and block_index >= start_block:
            parameter.requires_grad = True
