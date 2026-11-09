"""SimGCD adapter; ProtoTail and pseudo-labeling are separate plugins."""

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototail_gcd.baselines.simgcd.components import (
    DINOHead,
    DistillLoss,
    SupConLoss,
    info_nce_logits,
    parameter_groups,
)
from prototail_gcd.contracts import (
    BranchOutput,
    ModelOutput,
    PredictionOutput,
    TrainingBatch,
)
from prototail_gcd.models.dino_backbone import (
    backbone_spec,
    forward_global_and_patches,
    load_backbone,
    set_finetune_blocks,
)


class SimGCDNetwork(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        projector: DINOHead,
        finetune_from_block: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.finetune_from_block = finetune_from_block

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, patch_tokens = forward_global_and_patches(
            self.backbone,
            images,
            finetune_from_block=self.finetune_from_block,
        )
        projection, logits = self.projector(features)
        return features, projection, logits, patch_tokens


class SimGCDAdapter:
    name = "simgcd"
    capabilities = frozenset(
        {
            "global_features",
            "patch_tokens",
            "projection",
            "parametric_logits",
            "distribution_estimator",
        }
    )

    def __init__(self, eval_head: str = "kmeans") -> None:
        self.eval_head = eval_head

    def create_training_context(
        self,
        baseline_config: Mapping[str, Any],
        *,
        num_views: int,
        total_epochs: int,
    ) -> dict[str, Any]:
        cluster_criterion = DistillLoss(
            warmup_epochs=int(baseline_config["warmup_teacher_temp_epochs"]),
            total_epochs=total_epochs,
            num_views=num_views,
            warmup_temperature=float(baseline_config["warmup_teacher_temp"]),
            teacher_temperature=float(baseline_config["teacher_temp"]),
            student_temperature=float(baseline_config["student_temp"]),
        )
        return {"cluster_criterion": cluster_criterion}

    def build_model(
        self,
        model_config: Mapping[str, Any],
        baseline_config: Mapping[str, Any],
        num_classes: int,
    ) -> SimGCDNetwork:
        backbone_name = str(model_config["backbone"])
        backbone = load_backbone(
            backbone_name,
            pretrained=bool(model_config["pretrained"]),
        )
        if model_config.get("warmup_model_path"):
            checkpoint = torch.load(
                model_config["warmup_model_path"],
                map_location="cpu",
            )
            backbone.load_state_dict(checkpoint)

        set_finetune_blocks(
            backbone,
            int(model_config["finetune_from_block"]),
        )
        feature_dim = backbone_spec(backbone_name)["feature_dim"]
        projector = DINOHead(
            in_dim=feature_dim,
            out_dim=num_classes,
            use_bn=bool(baseline_config["use_bn"]),
            norm_last_layer=bool(baseline_config["norm_last_layer"]),
            num_layers=int(baseline_config["num_mlp_layers"]),
            hidden_dim=int(baseline_config["hidden_dim"]),
            bottleneck_dim=int(baseline_config["bottleneck_dim"]),
        )
        return SimGCDNetwork(
            backbone,
            projector,
            int(model_config["finetune_from_block"]),
        )

    def forward(
        self,
        model: SimGCDNetwork,
        batch: TrainingBatch,
    ) -> ModelOutput:
        if len(batch.views) < 2:
            raise ValueError("SimGCD requires at least two augmented views")

        images = torch.cat(tuple(batch.views), dim=0)
        features, projection, logits, patch_tokens = model(images)
        return ModelOutput(
            branches={
                "student": BranchOutput(
                    global_features=features,
                    patch_tokens=(
                        patch_tokens if hasattr(model, "prototail") else None
                    ),
                    projection=projection,
                    logits=logits,
                )
            },
            auxiliary={
                "batch_size": len(batch.sample_ids),
                "num_views": len(batch.views),
            },
        )

    def forward_eval(
        self,
        model: SimGCDNetwork,
        images: torch.Tensor,
    ) -> ModelOutput:
        features, projection, logits, patch_tokens = model(images)
        return ModelOutput(
            branches={
                "student": BranchOutput(
                    global_features=features,
                    patch_tokens=(
                        patch_tokens if hasattr(model, "prototail") else None
                    ),
                    projection=projection,
                    logits=logits,
                )
            }
        )

    def compute_losses(
        self,
        output: ModelOutput,
        batch: TrainingBatch,
        context: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        branch = output.branches["student"]
        return compute_simgcd_losses(
            projection=branch.projection,
            logits=branch.logits,
            batch=batch,
            context=context,
            config=config,
        )

    def optimizer_groups(
        self,
        model: SimGCDNetwork,
        config: Mapping[str, Any],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        baseline_model = nn.Sequential(model.backbone, model.projector)
        return {"main": parameter_groups(baseline_model)}

    def before_epoch(
        self,
        model: SimGCDNetwork,
        epoch: int,
        context: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        interval = int(config.get("distribution_estimate_interval", 10))
        if epoch != 0 and epoch % interval:
            return None
        loader = context.get("distribution_loader")
        if loader is None:
            freq = context.get("class_frequencies")
            if freq is not None:
                if epoch == 0 and bool(context.get("gate_warm_start_from_support", False)):
                    if hasattr(model, "prototail"):
                        model.prototail.prototypes.warm_start_from_estimated_support(freq)
                return {"estimated_frequencies": freq}
            return None
        device = next(model.backbone.parameters()).device
        from prototail_gcd.baselines.bacon.losses import estimate_class_frequencies
        try:
            frequencies = estimate_class_frequencies(
                model,
                loader,
                num_classes=int(context["num_classes"]),
                num_known_classes=int(context["num_known_classes"]),
                finetune_from_block=model.finetune_from_block,
                random_state=int(config.get("distribution_cluster_seed", 0)),
                device=device,
                kmeans_backend=str(context.get("kmeans_backend", "sklearn_cpu")),
                kmeans_n_init=int(context.get("kmeans_n_init", 10)),
                kmeans_max_iter=int(context.get("kmeans_max_iter", 300)),
                known_frequency_mapping=str(context.get("known_frequency_mapping", "direct_hungarian")),
            )
            model.train()
            if epoch == 0 and bool(context.get("gate_warm_start_from_support", False)):
                if not hasattr(model, "prototail"):
                    raise RuntimeError("Gate warm-start requires an attached ProtoTail module")
                model.prototail.prototypes.warm_start_from_estimated_support(frequencies)
                targets = model.prototail.prototypes.target_capacities(frequencies)
                print(
                    f"\n[ProtoTail Warm-Start] Epoch 0: Initialized gates from estimated support! "
                    f"Head capacity ~ {targets.max().item():.2f}, Tail capacity ~ {targets.min().item():.2f}, "
                    f"Gate mean ~ {model.prototail.prototypes.gates().mean().item():.3f}",
                    flush=True,
                )
            return {"estimated_frequencies": frequencies}
        except Exception:
            freq = context.get("class_frequencies")
            if freq is not None:
                if epoch == 0 and bool(context.get("gate_warm_start_from_support", False)):
                    if hasattr(model, "prototail"):
                        model.prototail.prototypes.warm_start_from_estimated_support(freq)
                return {"estimated_frequencies": freq}
            return None

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
        branch = output.branches["student"]
        features = branch.global_features
        logits = branch.logits
        if self.eval_head == "kmeans" and features is not None:
            return PredictionOutput(
                class_ids=None,
                features=F.normalize(features, dim=-1),
            )
        if logits is None:
            raise ValueError("SimGCD prediction requires student logits or features")
        return PredictionOutput(class_ids=logits.argmax(dim=-1), scores=logits)

    def evaluation_diagnostics(
        self,
        output: ModelOutput,
    ) -> Mapping[str, torch.Tensor]:
        branch = output.branches["student"]
        logits = branch.logits
        if logits is None:
            return {}
        return {
            "parametric": logits.argmax(dim=-1),
        }


def compute_simgcd_losses(
    *,
    projection: torch.Tensor,
    logits: torch.Tensor,
    batch: TrainingBatch,
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    num_views = int(context.get("num_views", len(batch.views)))
    batch_size = int(context.get("batch_size", len(batch.sample_ids)))
    labeled_mask = torch.as_tensor(
        batch.labeled_mask,
        device=logits.device,
    ).bool()
    targets = torch.as_tensor(batch.known_targets, device=logits.device).long()
    if labeled_mask.numel() != batch_size or targets.numel() != batch_size:
        raise ValueError("Batch masks and targets must match the sample count")

    logits_by_view = logits.chunk(num_views)
    labeled_logits = torch.cat(
        [view_logits[labeled_mask] for view_logits in logits_by_view],
        dim=0,
    )
    labeled_targets = targets[labeled_mask].repeat(num_views)
    if labeled_targets.numel():
        classification_loss = F.cross_entropy(
            labeled_logits / float(config["student_temp"]),
            labeled_targets,
        )
    else:
        classification_loss = logits.sum() * 0.0

    teacher_logits = context.get("teacher_logits")
    if teacher_logits is None:
        teacher_logits = logits.detach()
    cluster_criterion = context["cluster_criterion"]
    epoch = int(context["epoch"])
    distillation_loss = cluster_criterion(logits, teacher_logits, epoch)

    mean_probabilities = (
        logits / float(config["student_temp"])
    ).softmax(dim=1).mean(dim=0)
    me_max_loss = -torch.sum(
        torch.log(mean_probabilities ** (-mean_probabilities))
    )
    me_max_loss = me_max_loss + math.log(float(mean_probabilities.numel()))
    cluster_objective = distillation_loss
    cluster_objective = cluster_objective + (
        float(config["memax_weight"]) * me_max_loss
    )

    contrastive_logits, contrastive_targets = info_nce_logits(
        projection,
        num_views=num_views,
        temperature=float(config["contrastive_temp"]),
    )
    contrastive_loss = F.cross_entropy(contrastive_logits, contrastive_targets)

    if labeled_targets.numel():
        normalized_projection = F.normalize(projection, dim=-1)
        projection_by_view = normalized_projection.chunk(num_views)
        supervised_features = torch.stack(
            [
                view_projection[labeled_mask]
                for view_projection in projection_by_view
            ],
            dim=1,
        )
        supervised_contrastive_loss = SupConLoss(
            temperature=float(config["supervised_contrastive_temp"])
        )(supervised_features, targets[labeled_mask])
    else:
        supervised_contrastive_loss = projection.sum() * 0.0

    supervised_weight = float(config["sup_weight"])
    total_loss = (1.0 - supervised_weight) * cluster_objective
    total_loss = total_loss + supervised_weight * classification_loss
    total_loss = total_loss + (1.0 - supervised_weight) * contrastive_loss
    total_loss = total_loss + supervised_weight * supervised_contrastive_loss

    # -------------------------------------------------------------------------
    # Pseudo-labeling (Semi-Supervised Consistency / Cross-Entropy on Unlabeled)
    # -------------------------------------------------------------------------
    enable_pseudo = bool(
        config.get("enable_pseudo_labeling", False)
        or context.get("enable_pseudo_labeling", False)
    )
    enable_novel = bool(
        config.get("enable_novel_pseudo", False)
        or context.get("enable_novel_pseudo", False)
    )
    epoch = int(context.get("epoch", 0))
    total_epochs = int(context.get("total_epochs", 200))
    pseudo_warmup = int(
        config.get("pseudo_warmup_epochs", 0 if total_epochs <= 5 else 10)
    )

    unlabeled_mask = ~labeled_mask
    pseudo_loss = logits.sum() * 0.0
    n_pseudo_known = 0.0
    n_pseudo_novel = 0.0

    if enable_pseudo and epoch >= pseudo_warmup and unlabeled_mask.any() and num_views >= 2:
        student_temp = float(config["student_temp"])
        pseudo_temp = float(config.get("pseudo_temp", student_temp))
        threshold = float(config.get("pseudo_confidence", 0.7))
        known_class_count = int(context.get("num_known_classes", 100))

        u_logits_v0 = logits_by_view[0][unlabeled_mask]
        u_logits_v1 = logits_by_view[1][unlabeled_mask]

        with torch.no_grad():
            probs_v0 = F.softmax(u_logits_v0 / pseudo_temp, dim=-1)
            conf_v0, pred_v0 = probs_v0.max(dim=-1)
            mask_v0_known = (pred_v0 < known_class_count) & (conf_v0 >= threshold)
            mask_v0_novel = (
                (pred_v0 >= known_class_count) & (conf_v0 >= threshold)
                if enable_novel
                else torch.zeros_like(mask_v0_known)
            )
            valid_v0 = mask_v0_known | mask_v0_novel

            probs_v1 = F.softmax(u_logits_v1 / pseudo_temp, dim=-1)
            conf_v1, pred_v1 = probs_v1.max(dim=-1)
            mask_v1_known = (pred_v1 < known_class_count) & (conf_v1 >= threshold)
            mask_v1_novel = (
                (pred_v1 >= known_class_count) & (conf_v1 >= threshold)
                if enable_novel
                else torch.zeros_like(mask_v1_known)
            )
            valid_v1 = mask_v1_known | mask_v1_novel

        loss_0_to_1 = (
            F.cross_entropy(
                u_logits_v1[valid_v0] / student_temp,
                pred_v0[valid_v0],
            )
            if valid_v0.any()
            else logits.sum() * 0.0
        )
        loss_1_to_0 = (
            F.cross_entropy(
                u_logits_v0[valid_v1] / student_temp,
                pred_v1[valid_v1],
            )
            if valid_v1.any()
            else logits.sum() * 0.0
        )

        pseudo_loss = 0.5 * (loss_0_to_1 + loss_1_to_0)
        n_pseudo_known = float(mask_v0_known.sum().item() + mask_v1_known.sum().item()) / 2.0
        n_pseudo_novel = float(mask_v0_novel.sum().item() + mask_v1_novel.sum().item()) / 2.0

        pseudo_weight = float(
            config.get("pseudo_loss_weight", config.get("sup_weight", 0.35))
        )
        total_loss = total_loss + pseudo_weight * pseudo_loss

    return {
        "total": total_loss,
        "classification": classification_loss,
        "cls_loss": classification_loss,
        "distillation": distillation_loss,
        "cluster_objective": cluster_objective,
        "cluster_loss": cluster_objective,
        "me_max": me_max_loss,
        "contrastive": contrastive_loss,
        "contrastive_loss": contrastive_loss,
        "supervised_contrastive": supervised_contrastive_loss,
        "sup_con_loss": supervised_contrastive_loss,
        "pseudo_loss": pseudo_loss,
        "pseudo": pseudo_loss,
        "n_pseudo_known": torch.tensor(n_pseudo_known, device=logits.device),
        "n_pseudo_novel": torch.tensor(n_pseudo_novel, device=logits.device),
    }
