"""SimGCD head, representation losses, and self-distillation criterion."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        use_bn: bool,
        norm_last_layer: bool,
        num_layers: int,
        hidden_dim: int,
        bottleneck_dim: int,
    ) -> None:
        super().__init__()
        layer_count = max(num_layers, 1)
        if layer_count == 1:
            layers: list[nn.Module] = [nn.Linear(in_dim, bottleneck_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(layer_count - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(in_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projection = self.mlp(features)
        logits = self.last_layer(F.normalize(features, dim=-1, p=2))
        return projection, logits


def parameter_groups(model: nn.Module) -> list[dict[str, Any]]:
    regularized = []
    unregularized = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or parameter.ndim == 1:
            unregularized.append(parameter)
        else:
            regularized.append(parameter)
    return [
        {"params": regularized},
        {"params": unregularized, "weight_decay": 0.0},
    ]


def info_nce_logits(
    features: torch.Tensor,
    *,
    num_views: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.shape[0] % num_views:
        raise ValueError("Feature count must be divisible by num_views")

    batch_size = features.shape[0] // num_views
    sample_ids = torch.arange(
        batch_size,
        device=features.device,
    ).repeat(num_views)
    positive_mask = sample_ids[:, None].eq(sample_ids[None, :])
    normalized = F.normalize(features, dim=1)
    similarities = normalized @ normalized.T

    self_mask = torch.eye(
        features.shape[0],
        dtype=torch.bool,
        device=features.device,
    )
    positive_mask = positive_mask[~self_mask].view(features.shape[0], -1)
    similarities = similarities[~self_mask].view(features.shape[0], -1)
    positives = similarities[positive_mask].view(features.shape[0], -1)
    negatives = similarities[~positive_mask].view(features.shape[0], -1)
    logits = torch.cat((positives, negatives), dim=1) / temperature
    targets = torch.zeros(
        features.shape[0],
        dtype=torch.long,
        device=features.device,
    )
    return logits, targets


class SupConLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                "features must have shape [batch, views, dimensions]"
            )

        batch_size, view_count = features.shape[:2]
        labels = labels.reshape(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Label count does not match feature batch size")

        positive_mask = labels.eq(labels.T).to(features.dtype)
        contrast_features = torch.cat(features.unbind(dim=1), dim=0)
        anchor_features = contrast_features
        anchor_count = view_count

        logits = anchor_features @ contrast_features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        positive_mask = positive_mask.repeat(anchor_count, view_count)
        logits_mask = torch.ones_like(positive_mask)
        diagonal = torch.arange(
            batch_size * anchor_count,
            device=features.device,
        )
        logits_mask.scatter_(1, diagonal[:, None], 0)
        positive_mask = positive_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_count
        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        return loss.view(anchor_count, batch_size).mean()


class DistillLoss(nn.Module):
    def __init__(
        self,
        *,
        warmup_epochs: int,
        total_epochs: int,
        num_views: int,
        warmup_temperature: float,
        teacher_temperature: float,
        student_temperature: float,
    ) -> None:
        super().__init__()
        warmup_count = min(max(int(warmup_epochs), 0), int(total_epochs))
        if warmup_count:
            warmup = np.linspace(
                warmup_temperature,
                teacher_temperature,
                warmup_count,
            )
        else:
            warmup = np.empty(0)
        remaining = max(int(total_epochs) - warmup_count, 0)
        self.teacher_temperature_schedule = np.concatenate(
            (warmup, np.full(remaining, teacher_temperature))
        )
        self.num_views = num_views
        self.student_temperature = student_temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        if epoch >= len(self.teacher_temperature_schedule):
            raise ValueError(
                "epoch exceeds the configured distillation schedule"
            )

        student_views = (student_logits / self.student_temperature).chunk(
            self.num_views
        )
        teacher_temperature = float(self.teacher_temperature_schedule[epoch])
        teacher_views = F.softmax(
            teacher_logits / teacher_temperature,
            dim=-1,
        ).detach().chunk(self.num_views)

        total_loss = student_logits.new_zeros(())
        term_count = 0
        for teacher_view_index, teacher_view in enumerate(teacher_views):
            for student_view_index, student_view in enumerate(student_views):
                if student_view_index == teacher_view_index:
                    continue
                cross_entropy = -teacher_view * F.log_softmax(
                    student_view,
                    dim=-1,
                )
                total_loss = total_loss + cross_entropy.sum(dim=-1).mean()
                term_count += 1

        if term_count == 0:
            raise ValueError("distillation requires at least two views")
        return total_loss / term_count
