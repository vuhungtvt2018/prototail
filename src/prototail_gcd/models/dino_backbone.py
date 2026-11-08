"""DINO/DINOv2 loading and fine-tuning helpers shared across adapters."""

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


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


def _resolve_repo_cache_dir(repository: str) -> tuple[str, str, str, Path]:
    if ":" in repository:
        repo_owner_name, ref = repository.split(":", 1)
    else:
        repo_owner_name, ref = repository, "main"
    owner, repo_name = repo_owner_name.split("/", 1)
    norm_br = ref.replace("/", "_")
    folder_name = f"{owner}_{repo_name}_{norm_br}"
    hub_dir = Path(torch.hub.get_dir())
    return owner, repo_name, ref, hub_dir / folder_name


def _ensure_hub_repo(repository: str) -> Path | None:
    """Ensure repository is downloaded to torch hub directory via git clone.

    This bypasses GitHub's zipball generation endpoint (which frequently triggers
    HTTP 504 Gateway Time-out on cloud VMs).
    """
    owner, repo_name, ref, target_dir = _resolve_repo_cache_dir(repository)
    if (target_dir / "hubconf.py").is_file():
        return target_dir

    git_url = f"https://github.com/{owner}/{repo_name}.git"
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    for clone_cmd in (
        ["git", "clone", "--depth", "1", "--branch", ref, git_url, str(target_dir)],
        ["git", "clone", "--depth", "1", git_url, str(target_dir)],
    ):
        try:
            res = subprocess.run(
                clone_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
            if res.returncode == 0 and (target_dir / "hubconf.py").is_file():
                return target_dir
        except Exception:
            pass
        if target_dir.exists() and not (target_dir / "hubconf.py").is_file():
            shutil.rmtree(target_dir, ignore_errors=True)

    return target_dir if (target_dir / "hubconf.py").is_file() else None


def load_backbone(name: str, pretrained: bool = True, max_retries: int = 5) -> Any:
    if name not in HUB_SOURCES:
        available = tuple(HUB_SOURCES)
        raise ValueError(
            f"Unknown backbone {name!r}; choose from {available}"
        )

    repository, entrypoint = HUB_SOURCES[name]
    repo_cache_dir = _ensure_hub_repo(repository)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            if repo_cache_dir and (repo_cache_dir / "hubconf.py").is_file():
                if name.startswith("dinov2_"):
                    return torch.hub.load(
                        str(repo_cache_dir),
                        entrypoint,
                        source="local",
                        pretrained=pretrained,
                    )
                return torch.hub.load(
                    str(repo_cache_dir), entrypoint, source="local"
                )
            else:
                if name.startswith("dinov2_"):
                    return torch.hub.load(
                        repository, entrypoint, pretrained=pretrained
                    )
                return torch.hub.load(repository, entrypoint)
        except Exception as exc:
            last_error = exc
            wait_time = 2**attempt
            logger.warning(
                "Attempt %d/%d to load backbone %s failed (%s). Retrying in %ds...",
                attempt,
                max_retries,
                name,
                exc,
                wait_time,
            )
            print(
                f"[Backbone Loader] Attempt {attempt}/{max_retries} failed ({exc}). Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)
            if repo_cache_dir is None or not (repo_cache_dir / "hubconf.py").is_file():
                repo_cache_dir = _ensure_hub_repo(repository)

    raise RuntimeError(
        f"Failed to load backbone {name!r} after {max_retries} attempts."
    ) from last_error


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
