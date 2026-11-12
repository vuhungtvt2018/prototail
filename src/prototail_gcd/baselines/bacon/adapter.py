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
        # Paper D1/D2 promotion store supersedes the inline pseudo-CE: the
        # store promotes vetted samples (persistent, never relabeled) as
        # supervised targets in the engine. Running both mechanisms at once
        # injects the old noisy inline path alongside the clean one.
        if context.get("promotion_store") is not None:
            enable_pseudo = False
            enable_novel = False
        epoch = int(context.get("epoch", 0))
        total_epochs = int(context.get("total_epochs", 200))
        pseudo_warmup = int(
            config.get("pseudo_warmup_epochs", 0 if total_epochs <= 5 else 10)
        )

        unlabeled_mask = ~mask
        pseudo_loss = class_logits.sum() * 0.0

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
                mask_0_known = (pred_v0 < known_class_count) & (conf_v0 >= threshold)
                mask_0_novel = (
                    (pred_v0 >= known_class_count) & (conf_v0 >= threshold)
                    if enable_novel
                    else torch.zeros_like(mask_0_known)
                )
                valid_v0 = mask_0_known | mask_0_novel

                probs_v1 = F.softmax(u_logits_v1 / pseudo_temp, dim=-1)
                conf_v1, pred_v1 = probs_v1.max(dim=-1)
                mask_1_known = (pred_v1 < known_class_count) & (conf_v1 >= threshold)
                mask_1_novel = (
                    (pred_v1 >= known_class_count) & (conf_v1 >= threshold)
                    if enable_novel
                    else torch.zeros_like(mask_1_known)
                )
                valid_v1 = mask_1_known | mask_1_novel

            loss_0_to_1 = (
                F.cross_entropy(
                    u_logits_v1[valid_v0] / student_temp,
                    pred_v0[valid_v0],
                )
                if valid_v0.any()
                else class_logits.sum() * 0.0
            )
            loss_1_to_0 = (
                F.cross_entropy(
                    u_logits_v0[valid_v1] / student_temp,
                    pred_v1[valid_v1],
                )
                if valid_v1.any()
                else class_logits.sum() * 0.0
            )

            pseudo_loss = 0.5 * (loss_0_to_1 + loss_1_to_0)
            pseudo_weight = float(
                config.get("pseudo_loss_weight", config.get("supervised_weight", 0.35))
            )
            classification_objective = classification_objective + pseudo_weight * pseudo_loss

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
            "pseudo_loss": pseudo_loss,
            "pseudo": pseudo_loss,
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
        estimation_mode = str(config.get("frequency_estimation_mode", "kmeans"))
        if estimation_mode == "labeled_bincount":
            from prototail_gcd.baselines.bacon.losses import (
                estimate_class_frequencies_labeled,
            )
            frequencies = estimate_class_frequencies_labeled(
                loader,
                num_classes=int(context["num_classes"]),
                num_known_classes=int(context["num_known_classes"]),
            )
            print(
                f"[FREQ ESTIMATE] Epoch {epoch + 1} mode=labeled_bincount "
                f"head={frequencies.max().item():.0f} tail={frequencies.min().item():.0f} "
                f"ratio={frequencies.max().item() / max(frequencies.min().item(), 1e-6):.1f}",
                flush=True,
            )
        elif estimation_mode == "parametric_hungarian":
            from prototail_gcd.baselines.bacon.losses import (
                _collect_logits_and_labels,
                estimate_class_frequencies_parametric,
            )
            logits, targets, labeled = _collect_logits_and_labels(
                model,
                loader,
                num_classes=int(context["num_classes"]),
                num_known_classes=int(context["num_known_classes"]),
                finetune_from_block=model.finetune_from_block,
                device=device,
            )
            raw = estimate_class_frequencies_parametric(
                logits,
                targets,
                labeled,
                num_classes=int(context["num_classes"]),
                num_known_classes=int(context["num_known_classes"]),
            )
            raw_t = torch.as_tensor(raw, dtype=torch.float32, device=device)
            previous = context.get("estimated_frequencies")
            if previous is not None:
                ema = float(config.get("frequency_estimation_ema", 0.5))
                frequencies = ema * torch.as_tensor(
                    previous, dtype=torch.float32, device=device
                ) + (1.0 - ema) * raw_t
            else:
                frequencies = raw_t
            print(
                f"[FREQ ESTIMATE] Epoch {epoch + 1} mode=parametric_hungarian "
                f"head={frequencies.max().item():.0f} tail={frequencies.min().item():.0f} "
                f"ratio={frequencies.max().item() / max(frequencies.min().item(), 1e-6):.1f}",
                flush=True,
            )
        else:
            try:
                frequencies = estimate_class_frequencies(
                    model,
                    loader,
                    num_classes=int(context["num_classes"]),
                    num_known_classes=int(context["num_known_classes"]),
                    finetune_from_block=model.finetune_from_block,
                    random_state=int(config["distribution_cluster_seed"]),
                    device=device,
                    kmeans_backend=str(
                        context.get("kmeans_backend", "sklearn_cpu")
                    ),
                    kmeans_n_init=int(context.get("kmeans_n_init", 10)),
                    kmeans_max_iter=int(context.get("kmeans_max_iter", 300)),
                    known_frequency_mapping=str(
                        context.get(
                            "known_frequency_mapping", "bacon_rank"
                        )
                    ),
                )
            except RuntimeError:
                # Fallback: plain sklearn KMeans with standard settings.
                frequencies = estimate_class_frequencies(
                    model,
                    loader,
                    num_classes=int(context["num_classes"]),
                    num_known_classes=int(context["num_known_classes"]),
                    finetune_from_block=model.finetune_from_block,
                    random_state=int(config["distribution_cluster_seed"]),
                    device=device,
                    kmeans_backend="sklearn_cpu",
                    kmeans_n_init=4,
                    kmeans_max_iter=300,
                )
        adjustment = distribution_logit_adjustment(
            frequencies,
            strength=float(config["logit_adjustment_strength"]),
        )
        model.train()
        if epoch == 0 and bool(context.get("gate_warm_start_from_support", False)):
            if hasattr(model, "prototail"):
                model.prototail.prototypes.warm_start_from_estimated_support(frequencies)
                targets = model.prototail.prototypes.target_capacities(frequencies)
                print(
                    f"\n[ProtoTail Warm-Start] Epoch 0 (BaCon): Initialized gates from estimated support! "
                    f"Head capacity ~ {targets.max().item():.2f}, Tail capacity ~ {targets.min().item():.2f}, "
                    f"Gate mean ~ {model.prototail.prototypes.gates().mean().item():.3f}",
                    flush=True,
                )
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
        if features is not None:
            features = F.normalize(features, dim=-1)
        return PredictionOutput(class_ids=None, features=features)
