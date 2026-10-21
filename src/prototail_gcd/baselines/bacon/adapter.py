"""BaCon adapter preserving the two-branch baseline objectives."""

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototail_gcd.baselines.bacon.losses import (
    distribution_logit_adjustment,
    distribution_regularizer,
    estimate_class_frequencies,
    soft_pseudo_label_contrastive_loss,
)
from prototail_gcd.baselines.bacon.model import (
    BaConNetwork,
    BaConProjectionHead,
    CosineClassifier,
)
from prototail_gcd.baselines.simgcd.components import (
    DistillLoss,
    SupConLoss,
    info_nce_logits,
    parameter_groups,
)
from prototail_gcd.contracts import (
    ModelOutput,
    PredictionOutput,
    TrainingBatch,
)
from prototail_gcd.models.dino_backbone import (
    backbone_spec,
    load_backbone,
    set_finetune_blocks,
)


class BaConAdapter:
    name = "bacon"
    capabilities = frozenset(
        {
            "global_features",
            "patch_tokens",
            "projection",
            "parametric_logits",
            "two_branches",
            "distribution_estimator",
            "cluster_inference",
        }
    )

    def build_model(
        self,
        model_config: Mapping[str, Any],
        baseline_config: Mapping[str, Any],
        num_classes: int,
    ) -> BaConNetwork:
        backbone_name = str(model_config["backbone"])
        feature_dim = backbone_spec(backbone_name)["feature_dim"]
        finetune_from_block = int(model_config["finetune_from_block"])
        pretrained = bool(model_config.get("pretrained", True))

        classification_backbone = load_backbone(
            backbone_name,
            pretrained=pretrained,
        )
        contrastive_backbone = load_backbone(
            backbone_name,
            pretrained=pretrained,
        )
        set_finetune_blocks(classification_backbone, finetune_from_block)
        set_finetune_blocks(contrastive_backbone, finetune_from_block)

        classification_head = CosineClassifier(
            feature_dim,
            num_classes,
            norm_last_layer=bool(baseline_config["norm_last_layer"]),
        )
        contrastive_head = BaConProjectionHead(
            in_dim=feature_dim,
            out_dim=int(baseline_config["projection_dim"]),
            num_layers=int(baseline_config["num_mlp_layers"]),
            hidden_dim=int(baseline_config["hidden_dim"]),
            bottleneck_dim=int(baseline_config["bottleneck_dim"]),
            use_bn=bool(baseline_config["use_bn"]),
            norm_last_layer=bool(baseline_config["norm_last_layer"]),
        )
        return BaConNetwork(
            classification_backbone,
            classification_head,
            contrastive_backbone,
            contrastive_head,
            finetune_from_block=finetune_from_block,
        )

    def forward(
        self,
        model: BaConNetwork,
        batch: TrainingBatch,
    ) -> ModelOutput:
        if len(batch.views) < 2:
            raise ValueError("BaCon requires at least two augmented views")
        return model(torch.cat(tuple(batch.views), dim=0))

    def forward_eval(
        self,
        model: BaConNetwork,
        images: torch.Tensor,
    ) -> ModelOutput:
        return model(images)

    def create_training_context(
        self,
        config: Mapping[str, Any],
        *,
        num_views: int,
        total_epochs: int,
    ) -> Mapping[str, Any]:
        criterion = DistillLoss(
            warmup_epochs=int(config["warmup_teacher_temp_epochs"]),
            total_epochs=total_epochs,
            num_views=num_views,
            warmup_temperature=float(config["warmup_teacher_temp"]),
            teacher_temperature=float(config["teacher_temp"]),
            student_temperature=float(config["student_temp"]),
        )
        return {"cluster_criterion": criterion, "num_views": num_views}

    def compute_losses(
        self,
        output: ModelOutput,
        batch: TrainingBatch,
        context: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        class_logits = output.branches["classification"].logits
        contrastive_features = output.branches["contrastive"].projection
        if class_logits is None or contrastive_features is None:
            raise ValueError("BaCon requires class logits and CL projections")

        num_views = int(context["num_views"])
        batch_size = int(batch.sample_ids.numel())
        mask = torch.as_tensor(
            batch.labeled_mask,
            device=class_logits.device,
        ).bool()
        targets = torch.as_tensor(
            batch.known_targets,
            device=class_logits.device,
        ).long()
        if targets.numel() != batch_size or mask.numel() != batch_size:
            raise ValueError("BaCon targets/mask do not match batch size")

        frequencies = context.get("estimated_frequencies")
        adjustment = context.get("logit_adjustment")
        if frequencies is None or adjustment is None:
            raise RuntimeError(
                "BaCon frequency estimate is missing; call before_epoch"
            )

        logits_by_view = class_logits.chunk(num_views)
        labeled_logits = torch.cat(
            [view_logits[mask] for view_logits in logits_by_view],
            dim=0,
        )
        labeled_targets = targets[mask].repeat(num_views)
        if not labeled_targets.numel():
            raise ValueError("BaCon requires labeled samples in each batch")

        criterion = context["cluster_criterion"]
        teacher_logits = context.get("teacher_logits")
        if teacher_logits is None:
            teacher_logits = class_logits.detach()
        distillation = criterion(
            class_logits,
            teacher_logits,
            int(context["epoch"]),
        )
        distribution_loss = distribution_regularizer(
            class_logits,
            frequencies,
            temperature=float(config["student_temp"]),
            power=float(config["distribution_power"]),
        )
        supervised_logits = labeled_logits / float(config["student_temp"])
        if bool(config.get("use_logit_adjust", False)):
            supervised_logits = supervised_logits - adjustment
        classification_loss = F.cross_entropy(
            supervised_logits,
            labeled_targets,
        )
        classification_objective = (
            (1.0 - float(config["supervised_weight"]))
            * (
                distillation
                + float(config["distribution_loss_weight"])
                * distribution_loss
            )
            + float(config["supervised_weight"]) * classification_loss
        )

        info_logits, info_targets = info_nce_logits(
            contrastive_features,
            num_views=num_views,
            temperature=float(config["contrastive_temperature"]),
        )
        unsupervised_contrastive = F.cross_entropy(info_logits, info_targets)
        normalized_features = F.normalize(contrastive_features, dim=-1)
        feature_views = normalized_features.chunk(num_views)
        supervised_features = torch.stack(
            [view_features[mask] for view_features in feature_views],
            dim=1,
        )
        supervised_contrastive = SupConLoss(
            temperature=float(config["supervised_contrastive_temperature"])
        )(supervised_features, targets[mask])

        soft_contrastive = class_logits.new_zeros(())
        if int(context["epoch"]) >= int(config["contrastive_warmup_epochs"]):
            soft_contrastive = soft_pseudo_label_contrastive_loss(
                logits=class_logits,
                contrastive_features=contrastive_features,
                labeled_targets=targets,
                labeled_mask=mask,
                estimated_frequencies=frequencies,
                logit_adjustment=adjustment,
                num_views=num_views,
                temperature=float(config["student_temp"]),
                known_sampling_power=float(config["known_sampling_power"]),
                novel_sampling_power=float(config["novel_sampling_power"]),
            )
            contrastive_objective = (
                (1.0 - float(config["supervised_weight"]))
                * unsupervised_contrastive
                + 0.5 * float(config["supervised_weight"])
                * supervised_contrastive
                + 0.5 * float(config["supervised_weight"]) * soft_contrastive
            )
        else:
            contrastive_objective = (
                (1.0 - float(config["supervised_weight"]))
                * unsupervised_contrastive
                + float(config["supervised_weight"])
                * supervised_contrastive
            )

        return {
            "total": classification_objective + contrastive_objective,
            "loss_ce": classification_objective,
            "loss_cl": contrastive_objective,
            "classification": classification_loss,
            "cls_loss": classification_loss,
            "distillation": distillation,
            "cps_loss": distillation,
            "distribution": distribution_loss,
            "reg_loss": distribution_loss,
            "contrastive": unsupervised_contrastive,
            "contrastive_loss": unsupervised_contrastive,
            "supervised_contrastive": supervised_contrastive,
            "sup_con_loss": supervised_contrastive,
            "soft_contrastive": soft_contrastive,
            "classification_objective": classification_objective,
            "contrastive_objective": contrastive_objective,
        }

    def optimizer_groups(
        self,
        model: BaConNetwork,
        config: Mapping[str, Any],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        classification_model = nn.Sequential(
            model.classification_backbone,
            model.classification_head,
        )
        return {
            "classification": parameter_groups(classification_model),
            "contrastive": [
                {
                    "params": list(model.contrastive_backbone.parameters())
                    + list(model.contrastive_head.parameters())
                }
            ],
        }

    def before_epoch(
        self,
        model: BaConNetwork,
        epoch: int,
        context: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        interval = int(config["distribution_estimate_interval"])
        if epoch != 0 and epoch % interval:
            return None
        loader = context.get("distribution_loader")
        if loader is None:
            raise RuntimeError(
                "BaCon requires a distribution_loader in context"
            )
        device = next(model.contrastive_backbone.parameters()).device
        frequencies = estimate_class_frequencies(
            model,
            loader,
            num_classes=int(context["num_classes"]),
            num_known_classes=int(context["num_known_classes"]),
            finetune_from_block=model.finetune_from_block,
            random_state=int(config["distribution_cluster_seed"]),
            device=device,
        )
        adjustment = distribution_logit_adjustment(
            frequencies,
            strength=float(config["logit_adjustment_strength"]),
        )
        model.train()
        return {
            "estimated_frequencies": frequencies,
            "logit_adjustment": adjustment,
        }

    def after_optimizer_step(
        self,
        model,
        batch,
        output,
        epoch,
        context,
        config,
    ) -> None:
        return None

    def after_epoch(self, model, epoch, context, config):
        return None

    def predict(self, output: ModelOutput) -> PredictionOutput:
        features = output.branches["contrastive"].global_features
        return PredictionOutput(class_ids=None, features=features)
