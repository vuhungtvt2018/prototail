"""Two-branch BaCon model and its branch-specific heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototail_gcd.contracts import BranchOutput, ModelOutput
from prototail_gcd.models.dino_backbone import forward_global_and_patches


class CosineClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        *,
        norm_last_layer: bool,
    ) -> None:
        super().__init__()
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(feature_dim, num_classes, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.last_layer(F.normalize(features, dim=-1, p=2))


class BaConProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_layers: int,
        hidden_dim: int,
        bottleneck_dim: int,
        use_bn: bool,
        norm_last_layer: bool,
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
        self.mlp.apply(self._initialize)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = F.normalize(self.mlp(features), dim=-1, p=2)
        return self.last_layer(projected)


class BaConNetwork(nn.Module):
    def __init__(
        self,
        classification_backbone: nn.Module,
        classification_head: CosineClassifier,
        contrastive_backbone: nn.Module,
        contrastive_head: BaConProjectionHead,
        *,
        finetune_from_block: int,
    ) -> None:
        super().__init__()
        self.classification_backbone = classification_backbone
        self.classification_head = classification_head
        self.contrastive_backbone = contrastive_backbone
        self.contrastive_head = contrastive_head
        self.finetune_from_block = finetune_from_block

    def forward(self, images: torch.Tensor) -> ModelOutput:
        ce_features, _ = forward_global_and_patches(
            self.classification_backbone,
            images,
            finetune_from_block=self.finetune_from_block,
        )
        cl_features, patch_tokens = forward_global_and_patches(
            self.contrastive_backbone,
            images,
            finetune_from_block=self.finetune_from_block,
        )
        class_logits = self.classification_head(ce_features)
        contrastive_features = self.contrastive_head(cl_features)
        return ModelOutput(
            branches={
                "classification": BranchOutput(
                    global_features=ce_features,
                    logits=class_logits,
                ),
                "contrastive": BranchOutput(
                    global_features=cl_features,
                    patch_tokens=patch_tokens,
                    projection=contrastive_features,
                ),
            }
        )
