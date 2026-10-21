"""Small training engine that delegates method semantics to adapters."""

from typing import Any, Iterable, Mapping

import torch

from prototail_gcd.contracts import TrainingBatch


class TrainingEngine:
    def __init__(
        self,
        *,
        model: Any,
        baseline: Any,
        optimizers: Mapping[str, Any],
        baseline_config: Mapping[str, Any],
        method: Any | None = None,
        method_config: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
        schedulers: Mapping[str, Any] | None = None,
    ) -> None:
        if not optimizers:
            raise ValueError("at least one optimizer is required")
        self.model = model
        self.baseline = baseline
        self.optimizers = dict(optimizers)
        self.baseline_config = baseline_config
        self.method = method
        self.method_config = dict(method_config or {})
        self.context = dict(context or {})
        self.schedulers = dict(schedulers or {})
        if set(self.schedulers) - set(self.optimizers):
            raise ValueError("each scheduler must match a named optimizer")
        self._validate_disjoint_optimizers()

    def _validate_disjoint_optimizers(self) -> None:
        seen_parameter_ids: set[int] = set()
        shared_method_parameter_ids = (
            {
                id(parameter)
                for parameter in self.method.shared_parameters()
            }
            if self.method is not None
            else set()
        )
        for optimizer_name, optimizer in self.optimizers.items():
            current_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            overlap = seen_parameter_ids & current_ids
            is_method_part_optimizer = (
                self.method is not None
                and optimizer_name == f"{self.method.name}_part"
            )
            if overlap and not (
                is_method_part_optimizer
                and overlap <= shared_method_parameter_ids
            ):
                raise ValueError(
                    f"optimizer {optimizer_name!r} shares parameters with "
                    "another optimizer"
                )
            seen_parameter_ids.update(current_ids)

    def train_batch(
        self,
        batch: TrainingBatch,
        epoch: int,
    ) -> dict[str, float]:
        device = next(self.model.parameters()).device
        batch = batch.to(device)
        self.model.train()
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        output = self.baseline.forward(self.model, batch)
        if self.method is not None:
            output = self.method.augment_output(output)
        step_context = {**self.context, "epoch": epoch}
        losses = self.baseline.compute_losses(
            output,
            batch,
            step_context,
            self.baseline_config,
        )
        total_loss = losses.get("total")
        if not isinstance(total_loss, torch.Tensor) or total_loss.ndim != 0:
            raise TypeError(
                "baseline losses must contain a scalar tensor named 'total'"
            )
        if self.method is not None:
            method_losses = self.method.compute_losses(
                output,
                batch,
                step_context,
            )
            method_total = method_losses.get("total")
            if (
                not isinstance(method_total, torch.Tensor)
                or method_total.ndim != 0
            ):
                raise TypeError(
                    "method losses must contain a scalar tensor named 'total'"
                )
            losses = {
                **losses,
                **{
                    (
                        name
                        if name
                        in {
                            "aux_fused",
                            "fused_ce",
                            "gate",
                            "gate_max",
                            "dist_gate",
                            "spatial",
                        }
                        else f"{self.method.name}_{name}"
                    ): value
                    for name, value in method_losses.items()
                    if name != "total"
                },
                "total": total_loss + method_total,
            }
            if "loss_ce" in losses:
                losses["loss_ce"] = losses["loss_ce"] + method_total
            total_loss = losses["total"]

        total_loss.backward()
        for optimizer in self.optimizers.values():
            optimizer.step()
        after_step = getattr(self.baseline, "after_optimizer_step", None)
        if after_step is not None:
            after_step(
                self.model,
                batch,
                output,
                epoch,
                self.context,
                self.baseline_config,
            )
        if self.method is not None:
            self.method.after_optimizer_step(output, batch, step_context)
        metrics = {
            name: float(value.detach().item())
            for name, value in losses.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        }
        metrics["loss"] = float(total_loss.detach().item())
        return metrics

    def train_epoch(
        self,
        batches: Iterable[TrainingBatch],
        epoch: int,
    ) -> dict[str, float]:
        before_epoch = getattr(self.baseline, "before_epoch", None)
        if before_epoch is not None:
            updates = before_epoch(
                self.model,
                epoch,
                self.context,
                self.baseline_config,
            )
            if updates:
                self.context.update(updates)
        totals: dict[str, float] = {}
        batch_count = 0
        for batch in batches:
            batch_losses = self.train_batch(batch, epoch)
            for name, value in batch_losses.items():
                totals[name] = totals.get(name, 0.0) + value
            batch_count += 1

        if batch_count == 0:
            raise ValueError("training loader produced no batches")
        after_epoch = getattr(self.baseline, "after_epoch", None)
        if after_epoch is not None:
            updates = after_epoch(
                self.model,
                epoch,
                self.context,
                self.baseline_config,
            )
            if updates:
                self.context.update(updates)
        for scheduler in self.schedulers.values():
            scheduler.step()
        return {
            name: value / batch_count
            for name, value in totals.items()
        }
