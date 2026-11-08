"""ProtoTail parts, class-conditioned prototypes, and training objectives."""

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototail_gcd.contracts import (
    BranchOutput,
    ModelOutput,
    PredictionOutput,
    TrainingBatch,
)


class LatentPartModule(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_slots: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.queries = nn.Parameter(torch.empty(num_slots, feature_dim))
        nn.init.orthogonal_(self.queries)
        self.temperature = temperature

    def forward(
        self,
        patch_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_tokens.ndim != 3:
            raise ValueError("ProtoTail patch tokens must have shape (B, N, D)")
        feature_dim = patch_tokens.shape[-1]
        attention_logits = torch.einsum(
            "md,bnd->bmn",
            self.queries,
            patch_tokens,
        ) / (math.sqrt(feature_dim) * self.temperature)
        attention = F.softmax(attention_logits, dim=-1)
        parts = torch.einsum("bmn,bnd->bmd", attention, patch_tokens)
        return F.normalize(parts, dim=-1), attention


class PartPrototypeBank(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_slots: int,
        feature_dim: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_slots = num_slots
        prototypes = torch.empty(num_classes * num_slots, feature_dim)
        nn.init.orthogonal_(prototypes)
        self.register_buffer(
            "prototypes",
            prototypes.view(num_classes, num_slots, feature_dim),
        )
        self.register_buffer(
            "initial_prototypes",
            prototypes.view(num_classes, num_slots, feature_dim).clone(),
        )
        self.register_buffer(
            "update_counts",
            torch.zeros(num_classes, dtype=torch.long),
        )
        self.gate_logits = nn.Parameter(torch.zeros(num_classes, num_slots))
        nn.init.constant_(self.gate_logits, -1.0)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def target_capacities(self, frequencies: torch.Tensor) -> torch.Tensor:
        """Paper Appendix B: log-min-max support -> [1, num_slots]."""
        if frequencies.ndim != 1 or frequencies.numel() != self.num_classes:
            raise ValueError("frequencies must have one value per class")
        low, high = frequencies.min(), frequencies.max()
        if high <= low:
            # Preserve the current implementation's equal-support fallback.
            # Paper does not explicitly define this degenerate case.
            return torch.full_like(frequencies, float(self.num_slots))
        log_frequencies = torch.log(frequencies + 1e-8)
        log_low = torch.log(low + 1e-8)
        log_high = torch.log(high + 1e-8)
        targets = 1.0 + (
            float(self.num_slots - 1)
            * (log_frequencies - log_low)
            / (log_high - log_low + 1e-8)
        )
        return targets.clamp(1.0, float(self.num_slots))

    @torch.no_grad()
    def warm_start_from_estimated_support(self, frequencies: torch.Tensor) -> None:
        """NON-PAPER-DEFAULT: approximate the target gate mass at init."""
        target = self.target_capacities(frequencies.to(self.gate_logits.device))
        # Avoid exact 0/1: otherwise the sigmoid gradient would vanish.
        per_slot = (target / self.num_slots).clamp(0.02, 0.98)
        logits = torch.logit(per_slot).unsqueeze(-1)
        self.gate_logits.copy_(logits.expand_as(self.gate_logits))

    def forward(
        self,
        parts: torch.Tensor,
        shrinkage_m: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prototypes = self.prototypes
        initialized = self.update_counts.to(parts.device) > 0
        if shrinkage_m > 0 and initialized.any():
            # IMDb Empirical Bayes Shrinkage: R' = (v / (v + m)) * R + (m / (v + m)) * C
            counts = self.update_counts.to(parts.device).float().unsqueeze(-1).unsqueeze(-1)
            c_mean = prototypes[initialized].mean(dim=0, keepdim=True)
            weight = counts / (counts + shrinkage_m)
            shrunk = weight * prototypes + (1.0 - weight) * c_mean
            normalized_prototypes = F.normalize(shrunk, dim=-1)
        else:
            normalized_prototypes = F.normalize(prototypes, dim=-1)

        gates = self.gates()
        similarities = torch.einsum(
            "bmd,cmd->bcm",
            parts,
            normalized_prototypes,
        )
        scores = (similarities * gates.unsqueeze(0)).sum(dim=-1)
        scores = scores / (gates.sum(dim=-1).unsqueeze(0) + 1e-6)
        # An untouched prototype is an arbitrary orthogonal vector, not evidence.
        scores = scores.masked_fill(~initialized.unsqueeze(0), 0.0)
        return scores, similarities, gates

    @torch.no_grad()
    def update_known(
        self,
        parts: torch.Tensor,
        labels: torch.Tensor,
        momentum: float,
    ) -> None:
        for class_id in torch.unique(labels):
            mask = labels == class_id
            centroid = parts[mask].mean(dim=0)
            class_index = int(class_id.item())
            if self.update_counts[class_index] == 0:
                # EMA(0.99) of a random vector is not a valid first estimate.
                self.prototypes[class_index] = F.normalize(centroid, dim=-1)
            else:
                self.prototypes[class_index] = F.normalize(
                    momentum * self.prototypes[class_index]
                    + (1.0 - momentum) * centroid,
                    dim=-1,
                )
            self.update_counts[class_index] += 1

    @torch.no_grad()
    def update_novel(
        self,
        parts: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
        *,
        threshold: float,
        momentum: float,
    ) -> None:
        for class_id in torch.unique(labels):
            mask = (labels == class_id) & (confidence > threshold)
            if not torch.any(mask):
                continue
            weights = confidence[mask, None, None]
            centroid = (parts[mask] * weights).sum(dim=0)
            centroid = centroid / weights.sum(dim=0).clamp_min(1e-6)
            class_index = int(class_id.item())
            if self.update_counts[class_index] == 0:
                self.prototypes[class_index] = F.normalize(centroid, dim=-1)
            else:
                self.prototypes[class_index] = F.normalize(
                    momentum * self.prototypes[class_index]
                    + (1.0 - momentum) * centroid,
                    dim=-1,
                )
            self.update_counts[class_index] += 1

    @torch.no_grad()
    def prototype_diagnostics(self, num_known_classes: int) -> dict[str, Any]:
        curr_norm = F.normalize(self.prototypes, dim=-1)
        init_norm = F.normalize(self.initial_prototypes, dim=-1)
        cos_sim = (curr_norm * init_norm).sum(dim=-1).mean(dim=-1)
        drift = 1.0 - cos_sim

        k_cnt = self.update_counts[:num_known_classes]
        u_cnt = self.update_counts[num_known_classes:]
        k_drift = drift[:num_known_classes]
        u_drift = drift[num_known_classes:]

        return {
            "known_updated_classes": int((k_cnt > 0).sum().item()) if num_known_classes > 0 else 0,
            "known_total_classes": num_known_classes,
            "known_avg_updates": float(k_cnt.float().mean().item()) if num_known_classes > 0 else 0.0,
            "known_avg_drift": float(k_drift.mean().item()) if num_known_classes > 0 else 0.0,
            "novel_updated_classes": int((u_cnt > 0).sum().item()) if len(u_cnt) > 0 else 0,
            "novel_total_classes": len(u_cnt),
            "novel_untouched_classes": int((u_cnt == 0).sum().item()) if len(u_cnt) > 0 else 0,
            "novel_avg_updates": float(u_cnt.float().mean().item()) if len(u_cnt) > 0 else 0.0,
            "novel_avg_drift": float(u_drift.mean().item()) if len(u_drift) > 0 else 0.0,
        }


class ProtoTailComponents(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        slots = int(config["num_slots"])
        self.parts = LatentPartModule(
            feature_dim,
            slots,
            float(config.get("attention_temperature", 0.07)),
        )
        self.prototypes = PartPrototypeBank(
            num_classes,
            slots,
            feature_dim,
        )


class ProtoTailPlugin:
    name = "prototail"
    required_capabilities = frozenset(
        {"global_features", "patch_tokens", "parametric_logits"}
    )

    def __init__(self) -> None:
        self.config: Mapping[str, Any] = {}
        self.components: ProtoTailComponents | None = None
        self._shared_parameters: tuple[nn.Parameter, ...] = ()

    def attach(
        self,
        model: Any,
        config: Mapping[str, Any],
        num_classes: int,
        feature_dim: int,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("ProtoTail can only attach to a torch module")
        if "prototail" in model._modules:
            raise ValueError("Model already has a ProtoTail module attached")
        self.config = config
        self.components = ProtoTailComponents(
            num_classes,
            feature_dim,
            config,
        )
        model.add_module("prototail", self.components)

    def _components(self) -> ProtoTailComponents:
        if self.components is None:
            raise RuntimeError("ProtoTail must be attached before use")
        return self.components

    @staticmethod
    def _branches(
        output: ModelOutput,
    ) -> tuple[BranchOutput, BranchOutput]:
        if "student" in output.branches:
            classifier = output.branches["student"]
        elif "classification" in output.branches:
            classifier = output.branches["classification"]
        else:
            raise ValueError("ProtoTail requires a parametric class branch")
        patch_branch = output.branches.get("contrastive", classifier)
        if classifier.logits is None:
            raise ValueError("ProtoTail requires baseline class logits")
        if patch_branch.patch_tokens is None:
            raise ValueError("ProtoTail requires exposed patch tokens")
        return classifier, patch_branch

    def augment_output(self, output: ModelOutput) -> ModelOutput:
        classifier, patch_branch = self._branches(output)
        if classifier.logits is None or patch_branch.patch_tokens is None:
            raise ValueError("ProtoTail received incomplete baseline outputs")
        parts, attention = self._components().parts(
            patch_branch.patch_tokens
        )
        shrinkage_m = float(self.config.get("prototype_shrinkage_m", 0.0))
        part_scores, _similarities, gates = self._components().prototypes(
            parts, shrinkage_m=shrinkage_m
        )
        auxiliary = dict(output.auxiliary)
        auxiliary["prototail"] = {
            "parts": parts,
            "attention": attention,
            "part_scores": part_scores,
            "gates": gates,
            "global_logits": classifier.logits,
            "global_features": patch_branch.global_features,
        }
        return ModelOutput(branches=output.branches, auxiliary=auxiliary)

    def compute_losses(
        self,
        output: ModelOutput,
        batch: TrainingBatch,
        context: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        values = output.auxiliary.get("prototail")
        if not isinstance(values, Mapping):
            raise RuntimeError("ProtoTail output was not augmented")
        logits = values["global_logits"]
        part_scores = values["part_scores"]
        if not isinstance(logits, torch.Tensor) or not isinstance(
            part_scores, torch.Tensor
        ):
            raise TypeError("ProtoTail logits and part scores must be tensors")

        batch_size = int(batch.sample_ids.numel())
        if logits.shape[0] % batch_size:
            raise ValueError("ProtoTail output does not match the view count")
        labels = torch.as_tensor(
            batch.known_targets,
            device=logits.device,
        ).long()
        labeled = torch.as_tensor(
            batch.labeled_mask,
            device=logits.device,
        ).bool()
        if labels.numel() != batch_size or labeled.numel() != batch_size:
            raise ValueError("ProtoTail labels/mask do not match batch size")

        is_simgcd = "student" in output.branches
        has_labeled = bool(torch.any(labeled))
        losses: dict[str, torch.Tensor] = {}
        if is_simgcd:
            view_count = logits.shape[0] // batch_size
            # Base SimGCD already optimizes global CE. This auxiliary CE should
            # train local parts/gates, not add a second global-classifier gradient.
            fused_logits = (
                logits.detach()
                + float(self.config.get("part_weight", 0.5)) * part_scores
            )
            repeated_mask = labeled.repeat(view_count)
            supervised_logits = fused_logits[repeated_mask]
            supervised_labels = labels.repeat(view_count)[repeated_mask]
        else:
            fused_logits = (
                logits[:batch_size]
                + float(self.config.get("part_weight", 0.5))
                * part_scores[:batch_size]
            )
            supervised_logits = fused_logits[labeled]
            supervised_labels = labels[labeled]

        if has_labeled:
            scaled_fused = supervised_logits / float(
                self.config.get("temperature", 0.1)
            )
            if bool(
                self.config.get(
                    "use_logit_adjust",
                    context.get("use_logit_adjust", False),
                )
            ):
                adjustment = context.get("logit_adjustment")
                if adjustment is None:
                    raise RuntimeError(
                        "ProtoTail logit adjustment is enabled but "
                        "logit_adjustment is missing"
                    )
                scaled_fused = scaled_fused - torch.as_tensor(
                    adjustment,
                    device=logits.device,
                    dtype=logits.dtype,
                )
            losses["part_classification"] = F.cross_entropy(
                scaled_fused,
                supervised_labels,
            )
        else:
            losses["part_classification"] = logits.sum() * 0.0

        gates = values["gates"]
        epoch = int(context["epoch"])
        regularize_gates = is_simgcd or has_labeled
        gate_regularization = logits.sum() * 0.0
        if regularize_gates:
            gate_regularization = (
                gates * (1.0 - gates)
            ).mean() + F.relu(
                1.0 - gates.sum(dim=-1)
            ).mean() * float(self.config.get("gate_bound_weight", 0.05))
        losses["gate_regularization"] = gate_regularization

        capacity_weight = float(
            self.config.get("capacity_regularization_weight", 0.0)
        )
        capacity_loss = logits.sum() * 0.0
        if capacity_weight > 0 and regularize_gates:
            frequencies = context.get("estimated_frequencies")
            if frequencies is None:
                raise RuntimeError(
                    "ProtoTail capacity regularization requires "
                    "label-free estimated_frequencies; never use GT class_counts"
                )
            frequencies = torch.as_tensor(
                frequencies,
                device=logits.device,
                dtype=logits.dtype,
            )
            target_capacity = self._components().prototypes.target_capacities(
                frequencies
            )
            capacity_mode = str(self.config.get("capacity_target_mode", "frequency"))
            if capacity_mode == "uniform":
                # Control: same global capacity budget, no class-frequency allocation.
                target_capacity = target_capacity.mean().expand_as(target_capacity)
            elif capacity_mode != "frequency":
                raise ValueError(f"Unsupported capacity_target_mode: {capacity_mode}")
            capacity_loss = F.mse_loss(
                gates.sum(dim=-1),
                target_capacity,
            )
            losses["capacity_regularization"] = capacity_loss

        attention = values["attention"]
        if not isinstance(attention, torch.Tensor):
            raise TypeError("ProtoTail attention maps must be a tensor")
        spatial_weight = float(self.config.get("spatial_loss_weight", 0.0))
        spatial_loss = attention.sum() * 0.0
        if (
            bool(self.config.get("use_spatial_loss", False))
            and spatial_weight > 0
            and epoch >= int(self.config.get("spatial_loss_start_epoch", 50))
        ):
            batch_views, slots, _patch_count = attention.shape
            if slots > 1:
                overlap = torch.einsum(
                    "bmn,bkn->bmk",
                    attention,
                    attention,
                )
                off_diagonal = 1.0 - torch.eye(
                    slots,
                    device=attention.device,
                    dtype=attention.dtype,
                )
                overlap_loss = (
                    overlap * off_diagonal.unsqueeze(0)
                ).sum() / (batch_views * slots * (slots - 1))
            else:
                overlap_loss = attention.sum() * 0.0
            entropy = -(
                attention * torch.log(attention + 1e-8)
            ).sum(dim=-1).mean()
            spatial_loss = overlap_loss + float(
                self.config.get("spatial_entropy_weight", 0.1)
            ) * entropy
            losses["spatial"] = spatial_loss

        losses["total"] = (
            float(self.config.get("part_loss_weight", 1.0))
            * losses["part_classification"]
            + float(self.config.get("gate_regularization_weight", 0.05))
            * losses["gate_regularization"]
            + capacity_weight
            * capacity_loss
            + spatial_weight * spatial_loss
        )
        if is_simgcd:
            losses["aux_fused"] = losses["part_classification"]
            losses["gate"] = losses["gate_regularization"]
            if capacity_weight > 0:
                losses["dist_gate"] = capacity_loss
        else:
            losses["fused_ce"] = losses["part_classification"]
            losses["gate_max"] = gates.max(dim=-1).values.mean()
            if capacity_weight > 0:
                losses["dist_gate"] = capacity_loss
        return losses

    def optimizer_groups(
        self,
        model: Any,
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        components = self._components()
        # The baseline already owns the fine-tunable backbone parameters.
        # Its gradient includes BOTH baseline and ProtoTail loss terms.
        self._shared_parameters = ()
        part_groups: list[Mapping[str, Any]] = [
            {
                "params": list(components.parts.parameters()),
                "lr": float(self.config.get("learning_rate", 0.05)),
            }
        ]
        return {
            "part": part_groups,
            "gate": [
                {
                    "params": [components.prototypes.gate_logits],
                    "lr": float(
                        self.config.get("gate_learning_rate", 0.01)
                    ),
                    "weight_decay": 0.0,
                }
            ],
        }

    def shared_parameters(self) -> Sequence[nn.Parameter]:
        return self._shared_parameters

    @torch.no_grad()
    def after_optimizer_step(
        self,
        output: ModelOutput,
        batch: TrainingBatch,
        context: Mapping[str, Any],
    ) -> None:
        values = output.auxiliary.get("prototail")
        if not isinstance(values, Mapping):
            raise RuntimeError("ProtoTail output was not augmented")
        epoch = int(context["epoch"])
        total_epochs = int(context.get("total_epochs", epoch + 1))
        prototype_start = min(
            int(self.config.get("prototype_warmup_epochs", 30)),
            total_epochs - 1,
        )
        is_simgcd = "student" in output.branches

        batch_size = int(batch.sample_ids.numel())
        parts = values["parts"][:batch_size].detach()
        logits = values["global_logits"][:batch_size].detach()
        labels = torch.as_tensor(batch.known_targets, device=parts.device).long()
        labeled = torch.as_tensor(
            batch.labeled_mask,
            device=parts.device,
        ).bool()
        bank = self._components().prototypes
        momentum = float(self.config.get("prototype_momentum", 0.99))

        if torch.any(labeled):
            # Bootstrap known prototypes immediately from ground-truth labeled data.
            bank.update_known(parts[labeled], labels[labeled], momentum)

        # Novel unlabelled prototypes remain protected by their warmup schedule.
        if epoch < prototype_start:
            return
        unlabeled = ~labeled
        if not torch.any(unlabeled):
            return
        class_count = logits.shape[-1]
        known_class_count = int(context["num_known_classes"])
        if not 0 <= known_class_count <= class_count:
            raise ValueError("Invalid known class count for ProtoTail")
        if epoch > int(self.config.get("novel_prototype_end_epoch", 180)):
            return
        pseudo_label_logits = logits
        if not is_simgcd:
            shrinkage_m = float(self.config.get("prototype_shrinkage_m", 0.0))
            pseudo_label_logits = (
                logits
                + float(self.config.get("part_weight", 0.5))
                * bank(parts, shrinkage_m=shrinkage_m)[0]
            )
        probabilities = (
            pseudo_label_logits / float(self.config.get("temperature", 0.1))
        ).softmax(dim=-1)
        confidence, pseudo_labels = probabilities.max(dim=-1)
        if torch.any(unlabeled):
            conf_thresh = float(self.config.get("pseudo_confidence", 0.7))
            if bool(self.config.get("adaptive_pseudo_threshold", False)):
                conf_thresh = float(self.config.get("pseudo_confidence_min", 0.5))
            bank.update_novel(
                parts[unlabeled],
                pseudo_labels[unlabeled],
                confidence[unlabeled],
                threshold=conf_thresh,
                momentum=momentum,
            )

    def predict(self, output: ModelOutput) -> PredictionOutput:
        values = output.auxiliary.get("prototail")
        if not isinstance(values, Mapping):
            raise RuntimeError("ProtoTail output was not augmented")
        logits = values["global_logits"]
        part_scores = values["part_scores"]
        if not isinstance(logits, torch.Tensor) or not isinstance(
            part_scores, torch.Tensor
        ):
            raise TypeError("ProtoTail inference outputs must be tensors")

        parts = values["parts"]
        global_features = values["global_features"]
        if not isinstance(parts, torch.Tensor) or not isinstance(
            global_features, torch.Tensor
        ):
            raise TypeError("ProtoTail inference features must be tensors")
        part_classes = part_scores.argmax(dim=-1)
        gates = self._components().prototypes.gates()[part_classes]
        pooled_parts = (parts * gates.unsqueeze(-1)).sum(dim=1)
        pooled_parts = pooled_parts / gates.sum(dim=-1, keepdim=True).clamp_min(
            1e-6
        )
        feature_mode = self.config.get("eval_feature_mode", "legacy_concat")
        if feature_mode == "balanced_concat":
            # Evaluation-only change: scale each branch before concatenation.
            global_for_eval = F.normalize(global_features, dim=-1)
            part_for_eval = F.normalize(pooled_parts, dim=-1)
        elif feature_mode == "legacy_concat":
            global_for_eval, part_for_eval = global_features, pooled_parts
        else:
            raise ValueError(f"Unknown eval_feature_mode: {feature_mode}")
        features = F.normalize(
            torch.cat((global_for_eval, part_for_eval), dim=-1),
            dim=-1,
        )
        return PredictionOutput(class_ids=None, features=features)

    def evaluation_diagnostics(
        self,
        output: ModelOutput,
    ) -> Mapping[str, torch.Tensor]:
        if "student" not in output.branches:
            return {}
        values = output.auxiliary.get("prototail")
        if not isinstance(values, Mapping):
            raise RuntimeError("ProtoTail output was not augmented")
        global_logits = values["global_logits"]
        part_scores = values["part_scores"]
        if not isinstance(global_logits, torch.Tensor) or not isinstance(
            part_scores, torch.Tensor
        ):
            raise TypeError("ProtoTail diagnostic logits must be tensors")
        fused_logits = global_logits + float(
            self.config.get("part_weight", 0.5)
        ) * part_scores
        return {
            "fused": fused_logits.argmax(dim=-1),
            "global": global_logits.argmax(dim=-1),
        }
