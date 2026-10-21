"""Train a configured GCD baseline and record epoch metrics."""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans

from prototail_gcd.baselines.registry import build_baseline
from prototail_gcd.data.cub import CUBLongTailAdapter
from prototail_gcd.evaluation.clustering import clustering_metrics
from prototail_gcd.methods.prototail import ProtoTailPlugin
from prototail_gcd.models.dino_backbone import backbone_spec
from prototail_gcd.training.engine import TrainingEngine


def load_experiment(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping")

    baseline_config = config.get("baseline")
    if not isinstance(baseline_config, dict):
        raise ValueError("Missing baseline configuration")
    baseline_name = baseline_config.get("name")
    if not isinstance(baseline_name, str):
        raise ValueError("baseline.name must be a string")
    baseline_configs = baseline_config.get("configs")
    if not isinstance(baseline_configs, dict):
        raise ValueError("Missing baseline.configs mapping")
    try:
        baseline_config["resolved"] = baseline_configs[baseline_name]
    except KeyError as error:
        raise ValueError(
            f"Missing baseline.configs.{baseline_name} configuration"
        ) from error

    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise ValueError("Missing data configuration")
    if data_config.get("name") != "cub":
        raise ValueError("This runner currently supports data: cub")
    return config


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def evaluate(
    model: Any,
    loader: Any,
    adapter: Any,
    device: torch.device,
    known_class_count: int,
    num_classes: int,
    random_state: int,
    method: Any | None = None,
    class_counts: tuple[int, ...] | None = None,
    accuracy_scale: float = 1.0,
    empty_metric_value: float | None = None,
) -> dict[str, float | None]:
    model.eval()
    target_batches: list[np.ndarray] = []
    class_id_batches: list[np.ndarray] = []
    feature_batches: list[np.ndarray] = []
    diagnostic_batches: dict[str, list[np.ndarray]] = {}
    prediction_kind: str | None = None

    with torch.no_grad():
        for images, targets, _sample_ids, _is_labeled in loader:
            output = adapter.forward_eval(
                model,
                images.to(device, non_blocking=True),
            )
            if method is not None:
                output = method.augment_output(output)
                predictions = method.predict(output)
                for name, values in method.evaluation_diagnostics(
                    output
                ).items():
                    diagnostic_batches.setdefault(name, []).append(
                        _as_numpy(values)
                    )
            else:
                predictions = adapter.predict(output)
            target_batches.append(_as_numpy(targets))

            if predictions.class_ids is not None:
                if prediction_kind not in (None, "class_ids"):
                    raise ValueError(
                        "Baseline changed prediction type during evaluation"
                    )
                prediction_kind = "class_ids"
                class_id_batches.append(_as_numpy(predictions.class_ids))
            elif predictions.features is not None:
                if prediction_kind not in (None, "features"):
                    raise ValueError(
                        "Baseline changed prediction type during evaluation"
                    )
                prediction_kind = "features"
                feature_batches.append(_as_numpy(predictions.features))
            else:
                raise ValueError(
                    "Baseline prediction must return class IDs or features"
                )

    if not target_batches:
        raise ValueError("Evaluation loader produced no batches")
    targets = np.concatenate(target_batches)
    if prediction_kind == "class_ids":
        predicted_ids = np.concatenate(class_id_batches)
    elif prediction_kind == "features":
        features = np.concatenate(feature_batches)
        if features.ndim != 2:
            raise ValueError("Clustering features must be a 2-D array")
        predicted_ids = KMeans(
            n_clusters=num_classes,
            random_state=random_state,
            n_init=10,
        ).fit_predict(features)
    else:
        raise ValueError("Evaluation loader produced no predictions")

    metrics = clustering_metrics(
        targets=targets,
        predictions=predicted_ids,
        known_class_count=known_class_count,
        class_counts=class_counts,
        accuracy_scale=accuracy_scale,
        empty_group_value=empty_metric_value,
    )
    for name, batches in diagnostic_batches.items():
        diagnostic_predictions = np.concatenate(batches)
        diagnostic_metrics = clustering_metrics(
            targets=targets,
            predictions=diagnostic_predictions,
            known_class_count=known_class_count,
            class_counts=class_counts,
            accuracy_scale=accuracy_scale,
            empty_group_value=empty_metric_value,
        )
        metrics.update(
            {
                f"{name}_{metric_name}": value
                for metric_name, value in diagnostic_metrics.items()
            }
        )
    return metrics


def _build_optimizers(
    model: Any,
    baseline: Any,
    baseline_config: Mapping[str, Any],
    optimization: Mapping[str, Any],
    method: Any | None = None,
    method_config: Mapping[str, Any] | None = None,
) -> dict[str, torch.optim.Optimizer]:
    optimizers = {
        name: torch.optim.SGD(
            parameter_groups,
            lr=float(optimization["learning_rate"]),
            momentum=float(optimization["momentum"]),
            weight_decay=float(optimization["weight_decay"]),
        )
        for name, parameter_groups in baseline.optimizer_groups(
            model,
            baseline_config,
        ).items()
    }
    if method is not None:
        method_groups = method.optimizer_groups(model)
        if not method_groups:
            raise ValueError("Enabled method returned no optimizer groups")
        method_config = method_config or {}
        for group_name, parameter_groups in method_groups.items():
            learning_rate = float(
                method_config.get(
                    "gate_learning_rate"
                    if group_name == "gate"
                    else "learning_rate",
                    0.01 if group_name == "gate" else 0.05,
                )
            )
            optimizers[f"{method.name}_{group_name}"] = torch.optim.SGD(
                parameter_groups,
                lr=learning_rate,
                momentum=float(optimization["momentum"]),
                weight_decay=float(optimization["weight_decay"]),
            )
    return optimizers


def main(allowed_baseline: str | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--splits-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--enable-prototail",
        action="store_true",
        help="Enable ProtoTail even if method.enabled is false in the YAML",
    )
    arguments = parser.parse_args()

    config = load_experiment(arguments.config)
    if arguments.run_name:
        config["experiment"]["name"] = arguments.run_name
    baseline_name = config["baseline"]["name"]
    if allowed_baseline is not None and baseline_name != allowed_baseline:
        raise ValueError(
            f"This command supports baseline: {allowed_baseline}"
        )
    if arguments.data_root is not None:
        config["data"]["root"] = str(arguments.data_root)
    if arguments.splits_root is not None:
        config["data"]["splits_root"] = str(arguments.splits_root)

    method_config = config.get("method", {})
    if not isinstance(method_config, dict):
        raise ValueError("method configuration must be a mapping")
    if arguments.enable_prototail:
        method_config["enabled"] = True
    method_enabled = method_config.get("enabled", False)
    if not isinstance(method_enabled, bool):
        raise ValueError("method.enabled must be a YAML boolean")
    method_settings = method_config.get("config", {})
    if not isinstance(method_settings, dict):
        raise ValueError("method.config must be a mapping")
    method = None
    if method_enabled:
        method_name = method_config.get("name")
        if method_name != "prototail":
            raise ValueError(
                f"Unknown method {method_name!r}; available methods: prototail"
            )

    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    baseline = build_baseline(baseline_name)
    dataset = CUBLongTailAdapter().build(config)
    baseline_config = config["baseline"]["resolved"]
    model = baseline.build_model(
        config["model"],
        baseline_config,
        dataset.num_classes,
    )
    if method_enabled:
        missing_capabilities = (
            ProtoTailPlugin.required_capabilities - baseline.capabilities
        )
        if missing_capabilities:
            missing = ", ".join(sorted(missing_capabilities))
            raise ValueError(
                f"Baseline {baseline_name!r} lacks ProtoTail capabilities: "
                f"{missing}"
            )
        method = ProtoTailPlugin()
        method.attach(
            model,
            method_settings,
            dataset.num_classes,
            backbone_spec(str(config["model"]["backbone"]))["feature_dim"],
        )
    model = model.to(device)
    optimization = config["optimization"]
    optimizers = _build_optimizers(
        model,
        baseline,
        baseline_config,
        optimization,
        method=method,
        method_config=method_settings,
    )
    epoch_count = int(optimization["epochs"])
    schedulers = {
        name: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epoch_count,
            eta_min=(
                float(
                    method_settings.get(
                        "gate_learning_rate"
                        if name == "prototail_gate"
                        else "learning_rate",
                        0.01 if name == "prototail_gate" else 0.05,
                    )
                    if name.startswith("prototail_")
                    else optimization["learning_rate"]
                )
                * float(optimization["scheduler_min_lr_ratio"])
            ),
        )
        for name, optimizer in optimizers.items()
    }
    context = dict(
        baseline.create_training_context(
            baseline_config,
            num_views=int(optimization["num_views"]),
            total_epochs=epoch_count,
        )
    )
    if "distribution_estimator" in baseline.capabilities:
        context.update(
            {
                "distribution_loader": dataset.evaluation_loaders[
                    "distribution"
                ],
                "num_classes": dataset.num_classes,
                "num_known_classes": len(dataset.known_class_ids),
            }
        )
    context["num_classes"] = dataset.num_classes
    context["num_known_classes"] = len(dataset.known_class_ids)
    context["total_epochs"] = epoch_count
    context["use_logit_adjust"] = bool(
        baseline_config.get("use_logit_adjust", False)
    )
    engine = TrainingEngine(
        model=model,
        baseline=baseline,
        optimizers=optimizers,
        baseline_config=baseline_config,
        method=method,
        method_config=method_settings,
        context=context,
        schedulers=schedulers,
    )

    output_directory = Path(config["experiment"]["output_dir"])
    output_directory.mkdir(parents=True, exist_ok=True)
    experiment_name = str(config["experiment"]["name"])
    checkpoint_path = output_directory / f"{experiment_name}.pt"
    metrics_path = output_directory / f"{experiment_name}.metrics.jsonl"
    config_path = output_directory / f"{experiment_name}.config.yaml"
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    known_class_count = len(dataset.known_class_ids)
    eval_interval = int(config["evaluation"]["interval"])
    metric_class_counts = (
        dataset.class_counts if baseline_name == "bacon" else None
    )
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(epoch_count):
            train_metrics = engine.train_epoch(dataset.train_loader, epoch)
            record: dict[str, Any] = {
                "baseline": baseline_name,
                "method": {
                    "name": method_config.get("name"),
                    "enabled": method_enabled,
                    "applied": method is not None,
                },
                "epoch": epoch + 1,
                "train": train_metrics,
            }
            if (epoch + 1) % eval_interval == 0:
                record["unlabeled_train"] = evaluate(
                    model,
                    dataset.evaluation_loaders["unlabeled_train"],
                    baseline,
                    device,
                    known_class_count,
                    dataset.num_classes,
                    0,
                    method=method,
                    class_counts=metric_class_counts,
                    accuracy_scale=100.0 if baseline_name == "bacon" else 1.0,
                    empty_metric_value=-1.0 if baseline_name == "bacon" else None,
                )
                record["test"] = evaluate(
                    model,
                    dataset.evaluation_loaders["test"],
                    baseline,
                    device,
                    known_class_count,
                    dataset.num_classes,
                    0,
                    method=method,
                    class_counts=metric_class_counts,
                    accuracy_scale=100.0 if baseline_name == "bacon" else 1.0,
                    empty_metric_value=-1.0 if baseline_name == "bacon" else None,
                )
            serialized_record = json.dumps(record, sort_keys=True, allow_nan=False)
            print(serialized_record, flush=True)
            metrics_file.write(serialized_record + "\n")
            metrics_file.flush()
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizers": {
                        name: optimizer.state_dict()
                        for name, optimizer in optimizers.items()
                    },
                    "schedulers": {
                        name: scheduler.state_dict()
                        for name, scheduler in schedulers.items()
                    },
                    "epoch": epoch + 1,
                    "config": config,
                },
                checkpoint_path,
            )


if __name__ == "__main__":
    main()
