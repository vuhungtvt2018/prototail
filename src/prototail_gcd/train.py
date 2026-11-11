"""Train a configured GCD baseline and record epoch metrics."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

# Ensure src is in sys.path when executed directly as a script
if __package__ is None or __package__ == "":
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans
from prototail_gcd.utils.kmeans_backend import (
    fit_predict_kmeans,
    require_kmeans_backend,
)

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
    valid_names = ("cub", "cub200", "cars", "stanford_cars", "scars", "aircraft", "fgvc_aircraft")
    if data_config.get("name") not in valid_names:
        data_config["name"] = "cub"
    return config




def autocast_cuda(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)

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
    use_fp16: bool = False,
    kmeans_backend: str = "sklearn_cpu",
    kmeans_n_init: int = 10,
    kmeans_max_iter: int = 300,
) -> dict[str, float | None]:
    if kmeans_backend == "faiss_gpu" and device.type == "cpu":
        kmeans_backend = "sklearn_cpu" 
    if hasattr(model, "eval"):
        model.eval()
    target_batches: list[np.ndarray] = []
    class_id_batches: list[np.ndarray] = []
    feature_batches: list[np.ndarray] = []
    diagnostic_batches: dict[str, list[np.ndarray]] = {}
    diagnostic_feature_batches: dict[str, list[np.ndarray]] = {}
    prediction_kind: str | None = None

    with torch.no_grad(), autocast_cuda(use_fp16 and device.type == "cuda"):
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
                if hasattr(method, "diagnostic_features"):
                    for name, values in method.diagnostic_features(
                        output
                    ).items():
                        diagnostic_feature_batches.setdefault(
                            name, []
                        ).append(_as_numpy(values))
            else:
                predictions = adapter.predict(output)
                if hasattr(adapter, "evaluation_diagnostics"):
                    for name, values in adapter.evaluation_diagnostics(
                        output
                    ).items():
                        diagnostic_batches.setdefault(name, []).append(
                            _as_numpy(values)
                        )
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
    targets = np.concatenate(target_batches).reshape(-1)
    if prediction_kind == "class_ids":
        predicted_ids = np.concatenate(class_id_batches).reshape(-1)
    elif prediction_kind == "features":
        features = np.concatenate(feature_batches)
        if features.ndim != 2:
            raise ValueError("Clustering features must be a 2-D array")
        predicted_ids = fit_predict_kmeans(
            features,
            num_classes,
            seed=random_state,
            backend=kmeans_backend,
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
        )
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
    for name, batches in diagnostic_feature_batches.items():
        diagnostic_features = np.concatenate(batches)
        if diagnostic_features.ndim != 2:
            raise ValueError(f"Diagnostic features must be 2-D: {name}")
        diagnostic_ids = fit_predict_kmeans(
            diagnostic_features,
            num_classes,
            seed=random_state,
            backend=kmeans_backend,
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
        )
        diagnostic_metrics = clustering_metrics(
            targets=targets,
            predictions=diagnostic_ids,
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
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--epochs", type=int, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override base learning rate")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--eval-interval", type=int, help="Override evaluation epoch interval")
    parser.add_argument("--num-workers", type=int, help="Override dataloader num_workers")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable FP16 AMP")
    parser.add_argument("--imb-ratio", "--imb_ratio", dest="imb_ratio", type=int, help="Override imbalance ratio")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", choices=["cub", "cub200", "cars", "car", "scars", "aircraft", "air", "fgvc_aircraft", "herbarium", "herbarium19", "herb19", "inat", "inat2018", "inaturalist", "inaturalist2018"], default=None, help="Dataset name: cub, cars, aircraft, herbarium19, inat2018")
    parser.add_argument(
        "--simgcd-eval-head",
        choices=["kmeans", "parametric"],
        default="kmeans",
        help="Evaluation head for SimGCD baseline: 'kmeans' (matches ProtoTail/BaCon track) or 'parametric' (logits argmax)",
    )
    parser.add_argument("--use-parts", "--use_parts", dest="use_parts", action="store_true", default=False, help="Enable ProtoTail parts")
    parser.add_argument("--ablate-adaptive-capacity", "--ablate_adaptive_capacity", dest="ablate_adaptive_capacity", action="store_true", default=False, help="Ablate adaptive capacity")
    parser.add_argument("--enable-pseudo-labeling", "--enable_pseudo_labeling", dest="enable_pseudo_labeling", action="store_true", default=False, help="Enable pseudo labeling")
    parser.add_argument("--enable-novel-pseudo", "--enable_novel_pseudo", dest="enable_novel_pseudo", action="store_true", default=False, help="Enable novel pseudo labeling")
    parser.add_argument("--pseudo-warmup-epochs", "--pseudo_warmup_epochs", dest="pseudo_warmup_epochs", type=int, default=None, help="Override pseudo labeling warmup epochs")
    parser.add_argument("--pseudo-confidence", "--pseudo_confidence", dest="pseudo_confidence", type=float, default=None, help="Override pseudo confidence threshold")
    parser.add_argument(
        "--kmeans-backend",
        choices=["sklearn_cpu", "faiss_gpu"],
        default="sklearn_cpu",
        help="KMeans backend for clustering eval and distribution estimation",
    )
    parser.add_argument(
        "--use-momentum-teacher",
        "--use_momentum_teacher",
        dest="use_momentum_teacher",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable GlobalEMATeacher with momentum 0.996",
    )
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument(
        "--eval-feature-mode",
        choices=["original_parts_concat", "legacy_concat", "balanced_concat"],
        default="original_parts_concat",
        help="Feature mode for ProtoTail evaluation",
    )
    parser.add_argument(
        "--gate-warm-start-from-support",
        action="store_true",
        default=False,
        help="Init gates from estimated training supports at epoch 0",
    )
    parser.add_argument(
        "--no-gate-warm-start",
        action="store_true",
        default=False,
        help="Explicitly disable gate warm-start even in tail-aware configurations",
    )
    parser.add_argument(
        "--known-frequency-mapping",
        choices=["direct_hungarian", "bacon_rank"],
        default="direct_hungarian",
        help="Known frequency mapping mode",
    )
    arguments, _ = parser.parse_known_args()

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
    if arguments.output_dir is not None:
        config["experiment"]["output_dir"] = str(arguments.output_dir)
    if arguments.seed is not None:
        config["experiment"]["seed"] = arguments.seed
    if arguments.epochs is not None:
        config["optimization"]["epochs"] = arguments.epochs
    if arguments.batch_size is not None:
        config["optimization"]["batch_size"] = arguments.batch_size
    if arguments.lr is not None:
        config["optimization"]["learning_rate"] = arguments.lr
    if arguments.num_workers is not None:
        config["optimization"]["num_workers"] = arguments.num_workers

    if arguments.imb_ratio is not None and "protocol" in config.get("data", {}):
        config["data"]["protocol"]["imbalance_ratio"] = arguments.imb_ratio

    use_parts = bool(arguments.use_parts)
    ablate_capacity = bool(arguments.ablate_adaptive_capacity)
    enable_pseudo = bool(arguments.enable_pseudo_labeling)
    enable_novel = bool(arguments.enable_novel_pseudo)

    method_config = config.get("method", {})
    if not isinstance(method_config, dict):
        raise ValueError("method configuration must be a mapping")
    if arguments.enable_prototail or use_parts:
        method_config["enabled"] = True
    method_enabled = method_config.get("enabled", False)
    if not isinstance(method_enabled, bool):
        raise ValueError("method.enabled must be a YAML boolean")
    method_settings = method_config.get("config", {})
    if not isinstance(method_settings, dict):
        raise ValueError("method.config must be a mapping")
    if ablate_capacity:
        method_settings["capacity_regularization_weight"] = 0.0
    elif use_parts and not ablate_capacity:
        if float(method_settings.get("capacity_regularization_weight", 0.0)) == 0.0:
            method_settings["capacity_regularization_weight"] = 0.05
    if enable_pseudo:
        method_settings["enable_pseudo_labeling"] = True
    if enable_novel:
        method_settings["enable_novel_pseudo"] = True
    if arguments.pseudo_warmup_epochs is not None:
        method_settings["pseudo_warmup_epochs"] = arguments.pseudo_warmup_epochs
    if arguments.pseudo_confidence is not None:
        method_settings["pseudo_confidence"] = arguments.pseudo_confidence
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

    dataset_name = arguments.dataset_name or config["data"].get("name", "cub")
    config["data"]["name"] = dataset_name
    if "protocol" not in config["data"]:
        config["data"]["protocol"] = {}

    if dataset_name in ("cub", "cub200"):
        config["data"]["protocol"]["known_class_count"] = 100
    elif dataset_name in ("cars", "stanford_cars", "scars", "car"):
        config["data"]["protocol"]["known_class_count"] = 98
    elif dataset_name in ("aircraft", "fgvc_aircraft", "air"):
        config["data"]["protocol"]["known_class_count"] = 80
    elif dataset_name in ("herbarium19", "herb19", "herbarium"):
        config["data"]["protocol"]["known_class_count"] = 341
    elif dataset_name in ("inat2018", "inat", "inaturalist", "inaturalist2018"):
        config["data"]["protocol"]["known_class_count"] = 4071

    if arguments.imb_ratio is not None:
        config["data"]["protocol"]["imbalance_ratio"] = int(arguments.imb_ratio)
    else:
        config["data"]["protocol"].setdefault("imbalance_ratio", 20)

    from prototail_gcd.data import get_dataset_adapter
    dataset = get_dataset_adapter(dataset_name).build(config)

    baseline = build_baseline(baseline_name)
    if baseline_name == "simgcd" and hasattr(baseline, "eval_head"):
        baseline.eval_head = str(arguments.simgcd_eval_head or "kmeans")
    baseline_config = config["baseline"]["resolved"]
    if enable_pseudo:
        baseline_config["enable_pseudo_labeling"] = True
    if enable_novel:
        baseline_config["enable_novel_pseudo"] = True
    if arguments.pseudo_warmup_epochs is not None:
        baseline_config["pseudo_warmup_epochs"] = arguments.pseudo_warmup_epochs
    if arguments.pseudo_confidence is not None:
        baseline_config["pseudo_confidence"] = arguments.pseudo_confidence

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
    if "distribution_estimator" in baseline.capabilities or "distribution" in dataset.evaluation_loaders:
        context["distribution_loader"] = dataset.evaluation_loaders.get("distribution")
    context["num_classes"] = dataset.num_classes
    context["num_known_classes"] = len(dataset.known_class_ids)
    if hasattr(dataset, "class_counts") and dataset.class_counts is not None:
        context["class_counts"] = dataset.class_counts
        context["class_frequencies"] = torch.as_tensor(
            dataset.class_counts, dtype=torch.float32, device=device
        )
    context["total_epochs"] = epoch_count
    context["use_logit_adjust"] = bool(
        baseline_config.get("use_logit_adjust", False)
    )
    context["enable_pseudo_labeling"] = bool(
        baseline_config.get("enable_pseudo_labeling", False)
    )
    context["enable_novel_pseudo"] = bool(
        baseline_config.get("enable_novel_pseudo", False)
    )
    context["kmeans_backend"] = arguments.kmeans_backend
    context["kmeans_n_init"] = arguments.kmeans_n_init
    context["kmeans_max_iter"] = arguments.kmeans_max_iter
    context["known_frequency_mapping"] = arguments.known_frequency_mapping
    # Auto-enable gate warm-start for tail-aware configurations (Prototype_Tailaware, Full)
    is_tailaware = method_enabled and not ablate_capacity
    auto_warm_start = is_tailaware and not getattr(arguments, "no_gate_warm_start", False)
    context["gate_warm_start_from_support"] = bool(
        arguments.gate_warm_start_from_support or auto_warm_start
    )
    if arguments.eval_feature_mode is not None:
        method_settings["eval_feature_mode"] = arguments.eval_feature_mode

    # Determine explicit ablation configuration name
    if not method_enabled and not enable_pseudo:
        detected_config = "Base"
    elif not method_enabled and enable_pseudo:
        detected_config = "Pseudo"
    elif method_enabled and ablate_capacity and not enable_pseudo:
        detected_config = "Prototype"
    elif method_enabled and not ablate_capacity and not enable_pseudo:
        detected_config = "Prototype_Tailaware"
    elif method_enabled and ablate_capacity and enable_pseudo:
        detected_config = "Full_wo_Tail"
    elif method_enabled and not ablate_capacity and enable_pseudo:
        detected_config = "Full"
    else:
        detected_config = "Custom"

    # Teacher init must precede the banner below ([7] line reads `teacher`).
    known_class_count_teacher = len(dataset.known_class_ids)
    use_teacher = (
        arguments.use_momentum_teacher is True
        or (
            arguments.use_momentum_teacher is None
            and bool(
                baseline_config.get("use_momentum_teacher", False)
                or detected_config.startswith("Full")
            )
        )
    )
    teacher = None
    if use_teacher:
        from prototail_gcd.training.momentum_teacher import GlobalEMATeacher
        if baseline_name == "simgcd":
            teacher = GlobalEMATeacher(model.backbone, model.projector, m0=0.996).to(device)
        elif baseline_name == "bacon":
            teacher = GlobalEMATeacher(model.classification_backbone, model.classification_head, m0=0.996).to(device)
        print(f"[TEACHER] GlobalEMATeacher initialized (m0=0.996, device={device})", flush=True)

    print("\n" + "=" * 68)
    print("           PROTOTAIL GCD TRAINING CONFIGURATION")
    print("=" * 68)
    print(f"Run Name:                 {config['experiment']['name']}")
    print(f"Ablation Setting:         {detected_config}")
    print(f"Baseline Architecture:    {baseline_name.upper()}")
    print(f"Dataset / Split:          {dataset_name} (imb_ratio={config['data']['protocol'].get('imbalance_ratio', 'N/A')}, known={len(dataset.known_class_ids)}, total={dataset.num_classes})")
    print(f"Total Epochs / Workers:   {epoch_count} epochs / {optimization.get('num_workers', 2)} workers")
    print(f"Ablation Component Verification:")
    print(f"  [1] Baseline Objectives:  ACTIVE")
    print(f"  [2] ProtoTail Parts:      {'ACTIVE (attached)' if method_enabled else 'DISABLED'}")
    tail_status = 'ACTIVE (capacity_weight=0.05, warm_start=ON)' if (method_enabled and not ablate_capacity and context['gate_warm_start_from_support']) else ('ACTIVE (capacity_weight=0.05, warm_start=OFF)' if (method_enabled and not ablate_capacity) else 'DISABLED (weight=0.0)')
    print(f"  [3] Adaptive Tail-aware:  {tail_status}")
    print(f"  [4] Evaluation KMeans:    {arguments.kmeans_backend.upper()} (n_init={arguments.kmeans_n_init}, max_iter={arguments.kmeans_max_iter})")
    print(f"  [5] Pseudo-Labeling (D1): {'ACTIVE (known classes)' if enable_pseudo else 'DISABLED'}")
    print(f"  [6] Pseudo-Labeling (D2): {'ACTIVE (novel classes)' if (enable_pseudo and enable_novel) else 'DISABLED'}")
    print(f"  [7] Momentum Teacher:     {'ACTIVE (m0=0.996)' if teacher is not None else 'DISABLED'}")
    print("=" * 68 + "\n")

    from prototail_gcd.pseudo import PromotionStore, PaperDualSelector
    # Paper D1/D2 selection applies only to method-enabled configurations
    # (Full / Full_wo_Tail). The standalone "Pseudo" ablation keeps the
    # original inline pseudo-CE unchanged so the comparison stays clean.
    promotion_store = PromotionStore(dataset.num_classes, known_class_count_teacher)
    selector = PaperDualSelector(
        n_known=known_class_count_teacher,
        n_classes=dataset.num_classes,
        seed=seed,
    )
    if not method_enabled:
        # Empty store + disabled selector: adapters see no promotion_store
        # in context, inline pseudo stays active for the Pseudo ablation.
        selector = None
    else:
        context["promotion_store"] = promotion_store

    engine = TrainingEngine(
        model=model,
        baseline=baseline,
        optimizers=optimizers,
        baseline_config=baseline_config,
        method=method,
        method_config=method_settings,
        context=context,
        schedulers=schedulers,
        use_fp16=bool(arguments.fp16 and device.type == "cuda"),
        teacher=teacher,
    )

    output_directory = Path(config["experiment"]["output_dir"])
    output_directory.mkdir(parents=True, exist_ok=True)
    experiment_name = str(config["experiment"]["name"])
    checkpoint_path = output_directory / f"{experiment_name}.pt"
    best_checkpoint_path = output_directory / f"{experiment_name}.best.pt"
    best_unl_checkpoint_path = output_directory / f"{experiment_name}.best_unl.pt"
    metrics_path = output_directory / f"{experiment_name}.metrics.jsonl"
    config_path = output_directory / f"{experiment_name}.config.yaml"
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    best_test_acc = -1.0
    best_test_epoch = -1
    best_test_metrics: dict[str, Any] = {}
    best_unl_acc = -1.0
    best_unl_epoch = -1
    best_unl_metrics: dict[str, Any] = {}
    final_test_metrics: dict[str, Any] = {}
    final_unl_metrics: dict[str, Any] = {}
    known_class_count = len(dataset.known_class_ids)
    eval_interval = (
        arguments.eval_interval
        if arguments.eval_interval is not None
        else int(config["evaluation"]["interval"])
    )
    metric_class_counts = dataset.class_counts
    accuracy_scale = 100.0
    empty_metric_val = -1.0
    use_fp16_flag = bool(arguments.fp16 and device.type == "cuda")
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(epoch_count):
            if enable_pseudo and selector is not None and selector.scheduled(epoch):
                dist_loader = dataset.evaluation_loaders.get("distribution")
                if dist_loader is not None:
                    sel_event = selector.run(epoch, model, baseline, dist_loader, device, promotion_store)
                    print(f"[D1/D2 SELECTOR] Epoch {epoch + 1}: Promoted total={len(promotion_store)} | Details: {sel_event}", flush=True)
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
            is_last_epoch = (epoch + 1 == epoch_count)
            if (epoch + 1) % eval_interval == 0 or is_last_epoch:
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
                    accuracy_scale=accuracy_scale,
                    empty_metric_value=empty_metric_val,
                    use_fp16=use_fp16_flag,
                    kmeans_backend=arguments.kmeans_backend,
                    kmeans_n_init=arguments.kmeans_n_init,
                    kmeans_max_iter=arguments.kmeans_max_iter,
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
                    accuracy_scale=accuracy_scale,
                    empty_metric_value=empty_metric_val,
                    use_fp16=use_fp16_flag,
                    kmeans_backend=arguments.kmeans_backend,
                    kmeans_n_init=arguments.kmeans_n_init,
                    kmeans_max_iter=arguments.kmeans_max_iter,
                )
            serialized_record = json.dumps(record, sort_keys=True, allow_nan=False)
            print(serialized_record, flush=True)
            metrics_file.write(serialized_record + "\n")
            metrics_file.flush()
            if "unlabeled_train" in record:
                ut = record["unlabeled_train"]
                all_u = ut.get("all_acc", 0.0)
                print(f"[UNLABELED-TRAIN EVAL] Epoch {epoch + 1}: All {all_u:.2f} | Known {ut.get('known_acc', 0.0):.2f} | Novel {ut.get('novel_acc', 0.0):.2f} | H-Score {ut.get('h_score', 0.0):.2f}", flush=True)
                if "k_many" in ut and ut["k_many"] is not None and ut["k_many"] >= 0:
                    print(f"  Shot Breakdown (Train): Known [Many: {ut.get('k_many', 0.0):.2f}, Med: {ut.get('k_med', 0.0):.2f}, Few: {ut.get('k_few', 0.0):.2f}] | Novel [Many: {ut.get('u_many', 0.0):.2f}, Med: {ut.get('u_med', 0.0):.2f}, Few: {ut.get('u_few', 0.0):.2f}]", flush=True)

                if all_u > best_unl_acc:
                    best_unl_acc = all_u
                    best_unl_epoch = epoch + 1
                    best_unl_metrics = dict(ut)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "epoch": epoch + 1,
                            "best_unlabeled_metrics": ut,
                            "config": config,
                        },
                        best_unl_checkpoint_path,
                    )
                    print(f"  [*] New best Unlabeled-Train All Acc: {best_unl_acc:.2f}% -> saved to {best_unl_checkpoint_path.name}", flush=True)

            if "test" in record:
                test_m = record["test"]
                all_a = test_m.get("all_acc", 0.0)
                known_a = test_m.get("known_acc", 0.0)
                novel_a = test_m.get("novel_acc", 0.0)
                h_sc = test_m.get("h_score", 0.0)
                print(f"[DISJOINT-TEST EVAL] Epoch {epoch + 1}: All {all_a:.2f} | Known {known_a:.2f} | Novel {novel_a:.2f} | H-Score {h_sc:.2f}", flush=True)
                if "k_many" in test_m and test_m["k_many"] is not None and test_m["k_many"] >= 0:
                    print(f"  Shot Breakdown (Test):  Known [Many: {test_m.get('k_many', 0.0):.2f}, Med: {test_m.get('k_med', 0.0):.2f}, Few: {test_m.get('k_few', 0.0):.2f}] | Novel [Many: {test_m.get('u_many', 0.0):.2f}, Med: {test_m.get('u_med', 0.0):.2f}, Few: {test_m.get('u_few', 0.0):.2f}]", flush=True)
                if "fused_all_acc" in test_m and "global_all_acc" in test_m:
                    f_acc = test_m["fused_all_acc"]
                    g_acc = test_m["global_all_acc"]
                    print(f"  [ProtoTail Diagnostics] Fused: {f_acc:.2f} | Global: {g_acc:.2f} | Part Boost: {(f_acc - g_acc):+.2f}", flush=True)
                if "global_kmeans_all_acc" in test_m:
                    gk_acc = test_m["global_kmeans_all_acc"]
                    main_acc = test_m.get("all_acc", 0.0)
                    print(f"  [ProtoTail KMeans Split] Main(concat): {main_acc:.2f} | Global-only: {gk_acc:.2f} | Part Contribution: {(main_acc - gk_acc):+.2f}", flush=True)
                elif "parametric_all_acc" in test_m:
                    p_acc = test_m["parametric_all_acc"]
                    p_old = test_m.get("parametric_known_acc", 0.0)
                    p_new = test_m.get("parametric_novel_acc", 0.0)
                    print(f"  [SimGCD Diagnostics] Parametric Head: All {p_acc:.2f} | Known {p_old:.2f} | Novel {p_new:.2f}", flush=True)

                if all_a > best_test_acc:
                    best_test_acc = all_a
                    best_test_epoch = epoch + 1
                    best_test_metrics = dict(test_m)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "epoch": epoch + 1,
                            "best_test_metrics": test_m,
                            "config": config,
                        },
                        best_checkpoint_path,
                    )
                    print(f"  [*] New best test All Acc: {best_test_acc:.2f}% -> saved to {best_checkpoint_path.name}", flush=True)

            if is_last_epoch:
                if "test" in record:
                    final_test_metrics = dict(record["test"])
                if "unlabeled_train" in record:
                    final_unl_metrics = dict(record["unlabeled_train"])
            if method is not None and ((epoch + 1) % 10 == 0 or is_last_epoch):
                try:
                    with torch.no_grad():
                        bank = method._components().prototypes
                        gates = bank.gates()
                        mean_g = float(gates.mean().item())
                        max_g = float(gates.max().item())
                        frac_o = float((gates > 0.5).float().mean().item())
                        active_s = float(gates.sum(dim=-1).mean().item())
                        k_slots = float(gates[:known_class_count].sum(dim=-1).mean().item()) if known_class_count > 0 else 0.0
                        u_slots = float(gates[known_class_count:].sum(dim=-1).mean().item()) if gates.shape[0] > known_class_count else 0.0
                        freq = engine.context.get("estimated_frequencies")
                        mae_str = ""
                        if freq is not None:
                            f_t = torch.as_tensor(freq, dtype=gates.dtype, device=gates.device)
                            desired = bank.target_capacities(f_t)
                            c_mae = float((gates.sum(dim=-1) - desired).abs().mean().item())
                            mae_str = f" capacity_mae={c_mae:.4f} (vs estimated_frequencies, same source as dist_gate)"
                        print(f"[PROTOTYPE CAMERA] Epoch {epoch + 1}: Gate mean={mean_g:.3f} max={max_g:.3f} frac_open(>0.5)={frac_o:.1%} active_slots_per_class~={active_s:.2f}{mae_str}", flush=True)
                        print(f"  Active Slots/Class: Known ~ {k_slots:.2f} | Novel ~ {u_slots:.2f}", flush=True)
                except Exception:
                    pass
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

    summary_path = output_directory / f"{experiment_name}.summary.json"
    summary_data = {
        "experiment_name": experiment_name,
        "baseline": baseline_name,
        "ablation_config": detected_config,
        "dataset_name": dataset_name,
        "imbalance_ratio": config["data"]["protocol"].get("imbalance_ratio"),
        "total_epochs": epoch_count,
        "best_unlabeled_epoch": best_unl_epoch,
        "best_test_epoch": best_test_epoch,
        "best_unlabeled_train_metrics": best_unl_metrics,
        "best_test_metrics": best_test_metrics,
        "final_test_metrics": final_test_metrics,
        "final_unlabeled_train_metrics": final_unl_metrics,
        "checkpoint_final": str(checkpoint_path.name),
        "checkpoint_best_unlabeled": str(best_unl_checkpoint_path.name),
        "checkpoint_best_test": str(best_checkpoint_path.name),
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary_data, stream, indent=2, sort_keys=True)

    def _format_eval_line(label: str, m: dict[str, Any]) -> str:
        if not m:
            return f"  {label:<18} (No evaluation record)"
        all_v = m.get("all_acc", 0.0)
        k_v = m.get("known_acc", 0.0)
        n_v = m.get("novel_acc", 0.0)
        h_v = m.get("h_score", 0.0)
        line = f"  {label:<18} All: {all_v:6.2f}% | Known: {k_v:6.2f}% | Novel: {n_v:6.2f}% | H-Score: {h_v:6.2f}%"
        if "k_many" in m and m["k_many"] is not None and m["k_many"] >= 0:
            line += (
                f"\n    Shot Breakdown:   Known [Many: {m.get('k_many', 0.0):.2f}, Med: {m.get('k_med', 0.0):.2f}, Few: {m.get('k_few', 0.0):.2f}]"
                f"\n                      Novel [Many: {m.get('u_many', 0.0):.2f}, Med: {m.get('u_med', 0.0):.2f}, Few: {m.get('u_few', 0.0):.2f}]"
            )
        return line

    print("\n" + "=" * 78)
    print("                      PROTOTAIL TRAINING SUMMARY")
    print("=" * 78)
    print(f"Run Name:          {experiment_name}")
    print(f"Ablation Setting:  {detected_config}")
    print(f"Dataset / Split:   {dataset_name} (imb_ratio={config['data']['protocol'].get('imbalance_ratio', 'N/A')})")
    print(f"Total Epochs:      {epoch_count}")
    print("-" * 78)
    print(f"[FINAL MODEL - PAPER PROTOCOL] (Epoch {epoch_count}):")
    print(_format_eval_line("Disjoint Test:", final_test_metrics))
    if final_unl_metrics:
        print(_format_eval_line("Unlabeled Train:", final_unl_metrics))
    print(f"  Checkpoint File:   {checkpoint_path.name}")
    print("-" * 78)
    print(f"[BEST UNLABELED-TRAIN MODEL]   (Achieved at Epoch {best_unl_epoch}):")
    print(_format_eval_line("Unlabeled Train:", best_unl_metrics))
    print(f"  Checkpoint File:   {best_unl_checkpoint_path.name}")
    print("-" * 78)
    print(f"[BEST DISJOINT-TEST MODEL]     (Achieved at Epoch {best_test_epoch}):")
    print(_format_eval_line("Disjoint Test:", best_test_metrics))
    print(f"  Checkpoint File:   {best_checkpoint_path.name}")
    print("-" * 78)
    print(f"Summary JSON:      {summary_path.name}")
    print("=" * 78 + "\n", flush=True)


if __name__ == "__main__":
    main()
