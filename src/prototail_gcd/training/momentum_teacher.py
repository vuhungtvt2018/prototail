"""EMA teacher cho đường global logits, không EMA prototype buffers."""
from __future__ import annotations

import copy
import io
import math
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn


def _clone_module(module: nn.Module) -> nn.Module:
    """Clone a PyTorch module safely without copy.deepcopy, avoiding weight_norm errors."""
    try:
        return copy.deepcopy(module)
    except Exception:
        buf = io.BytesIO()
        torch.save(module, buf)
        buf.seek(0)
        return torch.load(buf, weights_only=False)


class GlobalEMATeacher:
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        *,
        m0: float = 0.996,
    ) -> None:
        self.backbone = _clone_module(backbone).eval()
        self.head = _clone_module(head).eval()
        self.m0 = float(m0)
        self.steps = 0
        for p in tuple(self.backbone.parameters()) + tuple(self.head.parameters()):
            p.requires_grad_(False)

    def to(self, device: torch.device) -> GlobalEMATeacher:
        self.backbone = self.backbone.to(device)
        self.head = self.head.to(device)
        return self

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        *,
        forward_features: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """forward_features là callable global CLS có cùng preprocessing với student."""
        z = forward_features(self.backbone, images)
        head_out = self.head(z)
        if isinstance(head_out, tuple):
            return head_out[-1]  # SimGCD DINOHead -> (projection, logits)
        return head_out

    @torch.no_grad()
    def update(
        self,
        source_backbone: nn.Module,
        source_head: nn.Module,
        *,
        epoch: int,
        total_epochs: int,
    ) -> None:
        total = max(1, int(total_epochs))
        m = 1.0 - (1.0 - self.m0) * 0.5 * (1.0 + math.cos(math.pi * epoch / total))
        for dst, src in zip(self.backbone.parameters(), source_backbone.parameters()):
            dst.mul_(m).add_(src.detach(), alpha=1.0 - m)
        for dst, src in zip(self.head.parameters(), source_head.parameters()):
            dst.mul_(m).add_(src.detach(), alpha=1.0 - m)
        for dst_b, src_b in zip(self.backbone.buffers(), source_backbone.buffers()):
            dst_b.copy_(src_b)
        for dst_b, src_b in zip(self.head.buffers(), source_head.buffers()):
            dst_b.copy_(src_b)
        self.steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone.state_dict(),
            "head": self.head.state_dict(),
            "steps": self.steps,
            "m0": self.m0,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.backbone.load_state_dict(state["backbone"])
        self.head.load_state_dict(state["head"])
        self.steps = int(state["steps"])
        self.m0 = float(state.get("m0", self.m0))
