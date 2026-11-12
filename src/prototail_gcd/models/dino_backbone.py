"""DINO/DINOv2 loading and fine-tuning helpers shared across adapters."""

import os
import re
import shutil
import ssl
import urllib.error
import urllib.request
from pathlib import Path
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

# File names under ``<torch.hub.get_dir()>/checkpoints/`` expected by each
# model's hubconf (derived from the dl.fbaipublicfiles.com URL basename).
_CHECKPOINT_FILES = {
    "dino_vitb16": "dino_vitb16_pretrain.pth",
    "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
    "dinov2_vitb14_reg": "dinov2_vitb14_reg4_pretrain.pth",
}

# Canonical checkpoint URLs. Pre-fetching them ourselves works around CDN
# certificate problems that crash torch.hub's internal download (e.g. the
# expired dl.fbaipublicfiles.com certificates seen on Kaggle).
_WEIGHT_URLS = {
    "dinov2_vitb14": (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
        "dinov2_vitb14_pretrain.pth"
    ),
    "dinov2_vitb14_reg": (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
        "dinov2_vitb14_reg4_pretrain.pth"
    ),
    "dino_vitb16": (
        "https://dl.fbaipublicfiles.com/dino/dino_vitb16/dino_vitb16_pretrain/"
        "dino_vitb16_pretrain.pth"
    ),
}


def _open_url_certificate_fallback(url: str, timeout: float = 60.0):
    """Open a URL, retrying without certificate verification on SSL errors.

    Some checkpoint CDNs (dl.fbaipublicfiles.com) ship expired certificates,
    which makes torch.hub's verified download fail permanently. Retrying with
    verification disabled is acceptable for a single public checkpoint file
    whose integrity is enforced by ``load_state_dict(strict=True)`` anyway.
    """
    try:
        return urllib.request.urlopen(url, timeout=timeout)
    except (urllib.error.URLError, ssl.SSLError) as error:
        reason = getattr(error, "reason", error)
        if not isinstance(reason, ssl.SSLError):
            raise
        print(
            "[DINO Backbone Loader] Certificate verification failed for the "
            f"checkpoint host ({reason}); retrying WITHOUT certificate "
            "verification for this file only.",
            flush=True,
        )
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return urllib.request.urlopen(url, timeout=timeout, context=context)


def prefetch_checkpoint(
    name: str,
    pretrained: bool = True,
    *,
    hub_dir: Path | None = None,
    timeout: float = 60.0,
) -> Path | None:
    """Pre-fetch the pretrained checkpoint into the torch hub cache.

    Placing the file under ``<hub_dir>/checkpoints/<expected name>`` makes
    ``torch.hub.load_state_dict_from_url`` hit the cache, so the hubconf
    never performs its own (verified, brittle) download. Returns the path on
    success, or None when the download is not applicable or fails.
    """
    if not pretrained:
        return None
    expected = _CHECKPOINT_FILES.get(name)
    url = _WEIGHT_URLS.get(name)
    if expected is None or url is None:
        return None
    hub_dir = hub_dir if hub_dir is not None else Path(torch.hub.get_dir())
    destination = hub_dir / "checkpoints" / expected
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    staging = destination.with_name(destination.name + ".part")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _open_url_certificate_fallback(url, timeout=timeout) as response, \
                staging.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        os.replace(staging, destination)
        size_mb = destination.stat().st_size / 1e6
        print(
            f"[DINO Backbone Loader] Fetched checkpoint {expected} "
            f"({size_mb:.0f} MB) into the hub cache.",
            flush=True,
        )
        return destination
    except Exception as error:
        try:
            if staging.exists():
                staging.unlink()
        except OSError:
            pass
        print(
            f"[DINO Backbone Loader] Checkpoint download failed: {error}",
            flush=True,
        )
        return None


def load_backbone(name: str, pretrained: bool = True) -> Any:
    if name not in HUB_SOURCES:
        available = tuple(HUB_SOURCES)
        raise ValueError(
            f"Unknown backbone {name!r}; choose from {available}"
        )

    repository, entrypoint = HUB_SOURCES[name]
    # Pre-fetch the checkpoint ourselves so torch.hub's internal verified
    # download never hits expired CDN certificates.
    prefetch_checkpoint(name, pretrained=pretrained)
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
