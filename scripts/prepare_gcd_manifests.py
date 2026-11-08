"""Prepare reproducible, benchmark-specific manifests without copying images.

Only annotation/metadata files are written. No model code reads novel GT during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _read_bacon_list(path: Path) -> list[dict]:
    rows = []
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            relpath, category = line.rsplit(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Bad BaCon list row {path}:{line_num}") from exc
        rows.append({"path": relpath, "category_id": int(category)})
    return rows


def _read_herb_folders(root: Path, subdir: str, class_index: dict[str, int]) -> list[dict]:
    folder = root / subdir
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    rows = []
    for path in sorted(folder.rglob("*.jpg")):
        category = path.parent.name
        if category not in class_index:
            raise ValueError(f"Category not in training set: {path}")
        rows.append({"path": path.relative_to(root).as_posix(), "category_id": class_index[category]})
    return rows


def _read_coco(path: Path) -> list[dict]:
    source = json.loads(path.read_text(encoding="utf-8"))
    image_paths = {int(x["id"]): str(x["file_name"]) for x in source["images"]}
    if len(image_paths) != len(source["images"]):
        raise ValueError(f"Duplicate COCO image ID in {path}")
    if len(source["annotations"]) != len(source["images"]):
        raise ValueError("Expected exactly one class annotation per image")
    rows = []
    used_ids: set[int] = set()
    for ann in source["annotations"]:
        image_id = int(ann["image_id"])
        if image_id in used_ids:
            raise ValueError(f"Duplicate image annotation: {image_id}")
        used_ids.add(image_id)
        rows.append({
            "path": image_paths[image_id],
            "category_id": int(ann["category_id"]),
        })
    return sorted(rows, key=lambda row: row["path"])


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select_labeled(
    rows: list[dict], known_ids: list[int], fraction: float, seed: int,
    mode: str,
) -> list[int]:
    if not (0 < fraction < 1):
        raise ValueError("labeled fraction must lie strictly within (0,1)")
    rng = np.random.RandomState(seed)  # Match the BaCon-style RNG family.
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[int(row["category_id"])].append(idx)
    missing = [cat for cat in known_ids if not grouped[cat]]
    if missing:
        raise ValueError(f"Known classes absent in training: {missing[:10]}")
    known_set = set(known_ids)
    if mode == "per_class":
        chosen = []
        for cat in known_ids:
            indices = np.asarray(grouped[cat], dtype=np.int64)
            # BaCon samples the unlabeled half of each known class.
            num_unlabeled = int((1.0 - fraction) * len(indices))
            unlab = set(int(i) for i in rng.choice(indices, size=num_unlabeled, replace=False))
            chosen.extend(int(i) for i in indices if int(i) not in unlab)
    elif mode == "global":
        candidates = [i for i, r in enumerate(rows) if int(r["category_id"]) in known_set]
        num_labeled = int(fraction * len(candidates))
        chosen = [int(i) for i in rng.choice(candidates, size=num_labeled, replace=False)]
    else:
        raise ValueError(f"Unknown selection mode: {mode}")
    chosen = sorted(chosen)
    if set(known_ids) - {int(rows[i]["category_id"]) for i in chosen}:
        raise ValueError("Some known classes have no labeled examples; change seed/protocol")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["imagenet100_lt", "herbarium19", "inat2018"], required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=416)
    parser.add_argument("--labeled-fraction", type=float, default=0.5)
    parser.add_argument("--selection", choices=["per_class", "global"])
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--bacon-train-list", type=Path)
    parser.add_argument("--bacon-eval-list", type=Path)
    parser.add_argument("--herb-splits-json", type=Path)
    parser.add_argument("--herb-train-dir", default="small-train")
    parser.add_argument("--herb-eval-dir", default="small-validation")
    parser.add_argument("--inat-train-json", type=Path)
    parser.add_argument("--inat-val-json", type=Path)
    parser.add_argument("--inat-known-ids-json", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    if args.dataset == "imagenet100_lt":
        if args.bacon_train_list is None or args.bacon_eval_list is None:
            parser.error("ImageNet-100-LT requires both official BaCon txt lists")
        train = _read_bacon_list(args.bacon_train_list)
        evaluation = _read_bacon_list(args.bacon_eval_list)
        known, novel = list(range(50)), list(range(50, 100))
        protocol_source = "BaCon ImageNet-100-LT class ordering and file lists"
        default_mode = "per_class"
    elif args.dataset == "herbarium19":
        if args.herb_splits_json is None:
            parser.error("Herbarium19 requires a JSON conversion of SimGCD's official Old/New pkl")
        class_dirs = sorted(p.name for p in (root / args.herb_train_dir).iterdir() if p.is_dir())
        if len(class_dirs) != 683:
            raise ValueError(f"Expected 683 Herbarium19 class dirs, got {len(class_dirs)}")
        class_to_int = {name: idx for idx, name in enumerate(class_dirs)}
        train = _read_herb_folders(root, args.herb_train_dir, class_to_int)
        evaluation = _read_herb_folders(root, args.herb_eval_dir, class_to_int)
        splits = json.loads(args.herb_splits_json.read_text(encoding="utf-8"))
        known, novel = [int(x) for x in splits["Old"]], [int(x) for x in splits["New"]]
        protocol_source = "SimGCD Herbarium19 Old/New class split"
        default_mode = "global"
    else:
        if args.inat_train_json is None or args.inat_val_json is None:
            parser.error("iNaturalist2018 requires train2018.json and val2018.json")
        train = _read_coco(args.inat_train_json)
        evaluation = _read_coco(args.inat_val_json)
        categories = sorted({int(row["category_id"]) for row in train})
        if len(categories) != 8142:
            raise ValueError(f"Expected 8142 iNat2018 categories, got {len(categories)}")
        if args.inat_known_ids_json is None:
            # No shared canonical iNat2018 GCD class split is assumed.
            shuffled = np.random.RandomState(args.seed).permutation(categories).tolist()
            known = sorted(int(x) for x in shuffled[:4071])
        else:
            known = [int(x) for x in json.loads(args.inat_known_ids_json.read_text(encoding="utf-8"))]
        novel = sorted(set(categories) - set(known))
        protocol_source = "New, frozen iNat2018 GCD protocol; not an official baseline split"
        default_mode = "per_class"

    if len(known) + len(novel) != len(set(known) | set(novel)):
        raise ValueError("Overlapping known/novel category IDs")
    if len(known) != {"imagenet100_lt": 50, "herbarium19": 341, "inat2018": 4071}[args.dataset]:
        raise ValueError("Unexpected number of known classes")
    all_classes = set(known + novel)
    if {int(r["category_id"]) for r in train} != all_classes:
        raise ValueError("Training labels do not match requested Old/New class split")
    if {int(r["category_id"]) for r in evaluation} != all_classes:
        raise ValueError("Evaluation labels do not cover the same Old/New class split")
    for label, rows in (("train", train), ("eval", evaluation)):
        if len({row["path"] for row in rows}) != len(rows):
            raise ValueError(f"Repeated relative path in {label}")
    if args.check_images:
        for row in train + evaluation:
            path = (root / row["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(path)

    mode = args.selection or default_mode
    labeled = _select_labeled(train, known, args.labeled_fraction, args.seed, mode)
    args.output.mkdir(parents=True, exist_ok=True)
    train_sha = _write_jsonl(args.output / "train.jsonl", train)
    eval_sha = _write_jsonl(args.output / "eval.jsonl", evaluation)
    metadata = {
        "dataset_id": args.dataset,
        "source_protocol": protocol_source,
        "seed": args.seed,
        "labeled_fraction": args.labeled_fraction,
        "selection": mode,
        "known_ids": known,
        "novel_ids": novel,
        "labeled_train_indices": labeled,
        "train_sha256": train_sha,
        "eval_sha256": eval_sha,
        "training_class_counts": {str(k): int(v) for k, v in sorted(Counter(x["category_id"] for x in train).items())},
        "n_train": len(train),
        "n_eval": len(evaluation),
        "n_labeled": len(labeled),
    }
    (args.output / "split.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(f"{args.dataset}: train={len(train)} labeled={len(labeled)} eval={len(evaluation)} ")
    print(f"split={args.output / 'split.json'} sha(train)={train_sha[:12]}")


if __name__ == "__main__":
    main()
