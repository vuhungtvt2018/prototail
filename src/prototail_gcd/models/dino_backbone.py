"""DINO/DINOv2 loading and fine-tuning helpers shared across adapters.

``load_backbone`` is offline-first: it resolves the model definition and the
pretrained checkpoint from local sources (env overrides, the torch hub cache,
or Kaggle dataset mounts under ``/kaggle/input``) before ever touching the
network. This is required for Kaggle notebooks running with Internet OFF,
where ``torch.hub.load("facebookresearch/dinov2:main")`` cannot resolve DNS.
"""

import os
import re
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    "dinov2_vitb14": ("facebookresearch/dinov2:main", "dinov2_vitb14"),
    "dinov2_vitb14_reg": ("facebookresearch/dinov2:main", "dinov2_vitb14_reg"),
}

_REPO_ENV_VAR = "PROTOTAIL_DINO_REPO"
_WEIGHTS_ENV_VAR = "PROTOTAIL_DINO_WEIGHTS"

# File names under ``<torch.hub.get_dir()>/checkpoints/`` expected by each
# model's hubconf (derived from the dl.fbaipublicfiles.com URL basename).
_CHECKPOINT_FILES = {
    "dino_vitb16": "dino_vitb16_pretrain.pth",
    "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
    "dinov2_vitb14_reg": "dinov2_vitb14_reg4_pretrain.pth",
}

# Kaggle mounts scanned for an offline copy of the repo / checkpoint.
_KAGGLE_SCAN_ROOTS: tuple[Path, ...] = (
    Path("/kaggle/input"),
    Path("/kaggle/working"),
)
_SCAN_MAX_DEPTH = 3
_SCAN_DIR_LIMIT = 4000

_OFFLINE_FIX_INSTRUCTIONS = """No Internet connection and no offline DINO backbone assets were found.
Fix (pick one):
  1. Kaggle notebook: Settings -> Internet -> ON (requires a phone-verified account).
  2. Attach an offline dataset with the DINOv2 repo + weights:
       python scripts/prepare_dinov2_offline.py --output dinov2_offline
     then upload the `dinov2_offline` folder as a Kaggle dataset and add it
     to the notebook (it is mounted under /kaggle/input and found automatically).
  3. Point env overrides at local copies:
       PROTOTAIL_DINO_REPO=/path/to/dinov2-repo   (dir with hubconf.py)
       PROTOTAIL_DINO_WEIGHTS=/path/to/dinov2_vitb14_pretrain.pth"""


def _hubconf_mentions(hubconf_path: Path, entrypoint: str) -> bool:
    try:
        text = hubconf_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return entrypoint in text[:16384]


def _walk_limited(root: Path, max_depth: int = _SCAN_MAX_DEPTH) -> Iterable[tuple[Path, list[str], list[str]]]:
    """os.walk capped by depth (below ``root``) and visited-directory count."""
    base_depth = len(root.parts)
    visited = 0
    for current, dirnames, filenames in os.walk(root):
        depth = len(Path(current).parts) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        visited += 1
        if visited > _SCAN_DIR_LIMIT:
            return
        yield Path(current), dirnames, filenames


def find_local_repo(
    name: str,
    *,
    hub_dir: Path | None = None,
    extra_roots: Sequence[Path] = (),
) -> Path | None:
    """Locate a local checkout of the backbone's hub repo, if one exists.

    Search order: ``PROTOTAIL_DINO_REPO`` env override, the torch hub cache
    (populated by a previous online session), then bounded scans of Kaggle
    dataset mounts so any dataset layout works.
    """
    if name not in HUB_SOURCES:
        return None
    repository, entrypoint = HUB_SOURCES[name]

    explicit: list[Path] = []
    env_repo = os.environ.get(_REPO_ENV_VAR)
    if env_repo:
        explicit.append(Path(env_repo))
    hub_dir = hub_dir if hub_dir is not None else Path(torch.hub.get_dir())
    explicit.append(hub_dir / repository.replace("/", "_").replace(":", "_"))

    for candidate in explicit:
        hubconf = candidate / "hubconf.py"
        if hubconf.is_file() and _hubconf_mentions(hubconf, entrypoint):
            return candidate

    for root in (*extra_roots, *_KAGGLE_SCAN_ROOTS):
        if not root.is_dir():
            continue
        for current, _dirnames, filenames in _walk_limited(root):
            if "hubconf.py" not in filenames:
                continue
            hubconf = current / "hubconf.py"
            if _hubconf_mentions(hubconf, entrypoint):
                return current
    return None


def _compatible_weights_file(filename: str, name: str) -> bool:
    stem = Path(filename).stem
    is_reg_variant = "_reg" in stem
    if name.endswith("_reg") != is_reg_variant:
        return False
    return stem.startswith(name)


def _find_weights_in_root(root: Path, name: str, expected: str) -> Path | None:
    exact: Path | None = None
    fuzzy: Path | None = None
    for current, _dirnames, filenames in _walk_limited(root):
        for filename in filenames:
            if not filename.endswith((".pth", ".pt", ".bin")):
                continue
            if filename == expected:
                return current / filename
            if fuzzy is None and _compatible_weights_file(filename, name):
                fuzzy = current / filename
    return fuzzy


def ensure_offline_checkpoint(
    name: str,
    pretrained: bool = True,
    *,
    hub_dir: Path | None = None,
    repo_dir: Path | None = None,
    extra_roots: Sequence[Path] = (),
) -> Path | None:
    """Make the pretrained checkpoint reachable offline via the hub cache.

    ``torch.hub.load_state_dict_from_url`` checks ``<hub_dir>/checkpoints/<url
    basename>`` first, so copying a dataset-mounted copy there (under the file
    name the hubconf expects) makes ``pretrained=True`` work without network.
    """
    if not pretrained:
        return None
    expected = _CHECKPOINT_FILES.get(name)
    if expected is None:
        return None

    hub_dir = hub_dir if hub_dir is not None else Path(torch.hub.get_dir())
    destination = hub_dir / "checkpoints" / expected
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    source: Path | None = None
    env_weights = os.environ.get(_WEIGHTS_ENV_VAR)
    if env_weights and Path(env_weights).is_file():
        source = Path(env_weights)
    if source is None and repo_dir is not None:
        for candidate in (
            repo_dir / "checkpoints" / expected,
            repo_dir / expected,
            repo_dir.parent / "checkpoints" / expected,
            repo_dir.parent / expected,
        ):
            if candidate.is_file():
                source = candidate
                break
    if source is None:
        for root in (*extra_roots, *_KAGGLE_SCAN_ROOTS):
            if not root.is_dir():
                continue
            source = _find_weights_in_root(root, name, expected)
            if source is not None:
                break
    if source is None:
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (destination.is_file() and destination.stat().st_size == source.stat().st_size):
        staging = destination.with_name(destination.name + ".part")
        shutil.copyfile(source, staging)
        os.replace(staging, destination)
    return destination


def locate_offline_assets(
    name: str = "dinov2_vitb14",
    pretrained: bool = True,
) -> tuple[Path | None, Path | None]:
    """Convenience pair (repo dir, cached checkpoint path) for pre-flight checks."""
    repo_dir = find_local_repo(name)
    checkpoint = ensure_offline_checkpoint(
        name, pretrained=pretrained, repo_dir=repo_dir
    )
    return repo_dir, checkpoint


def probe_internet(timeout: float = 4.0) -> bool:
    try:
        socket.create_connection(("github.com", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def load_backbone(
    name: str,
    pretrained: bool = True,
    *,
    extra_roots: Sequence[Path] = (),
) -> Any:
    if name not in HUB_SOURCES:
        available = tuple(HUB_SOURCES)
        raise ValueError(
            f"Unknown backbone {name!r}; choose from {available}"
        )

    repository, entrypoint = HUB_SOURCES[name]

    # 1. Offline-first: repo from env override / hub cache / Kaggle dataset mounts.
    repo_dir = find_local_repo(name, extra_roots=extra_roots)
    checkpoint = ensure_offline_checkpoint(
        name, pretrained=pretrained, repo_dir=repo_dir, extra_roots=extra_roots
    )
    if repo_dir is not None:
        if pretrained and checkpoint is None and not probe_internet():
            raise RuntimeError(
                f"Offline checkpoint for {name!r} not found next to {repo_dir}. "
                + _OFFLINE_FIX_INSTRUCTIONS
            )
        try:
            kwargs = {"pretrained": pretrained} if name.startswith("dinov2_") else {}
            source_note = checkpoint if checkpoint else "weights not pre-cached"
            print(
                f"[DINO Backbone Loader] Loading {name} offline from {repo_dir} "
                f"(checkpoint: {source_note})",
                flush=True,
            )
            return torch.hub.load(
                str(repo_dir), entrypoint, source="local", **kwargs
            )
        except Exception as error:
            if not probe_internet():
                raise RuntimeError(
                    f"Failed to load {name!r} offline from {repo_dir}: {error}\n"
                    + _OFFLINE_FIX_INSTRUCTIONS
                ) from error
            print(
                f"[DINO Backbone Loader] Offline repo load failed ({error}); "
                "falling back to hub download.",
                flush=True,
            )

    # 2. No local repo and no Internet: fail fast with instructions instead of
    #    burning five DNS-failing retries.
    if repo_dir is None and not probe_internet():
        raise RuntimeError(
            f"Cannot load backbone {name!r} from {repository!r}: "
            "no local copy found and Internet is unreachable.\n"
            + _OFFLINE_FIX_INSTRUCTIONS
        )

    # 3. Remote load with retry logic and skip_validation=True
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            kwargs = {
                "skip_validation": True,
                "trust_repo": True,
            }
            if name.startswith("dinov2_"):
                kwargs["pretrained"] = pretrained
            return torch.hub.load(repository, entrypoint, **kwargs)
        except Exception as error:
            last_error = error
            delay = attempt * 3
            print(
                f"[DINO Backbone Loader] Hub load attempt {attempt}/5 failed: {error}. "
                f"Retrying in {delay}s...",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to load backbone {name!r} from {repository!r} after 5 attempts.\n"
        + _OFFLINE_FIX_INSTRUCTIONS
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
