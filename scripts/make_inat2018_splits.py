"""Generate Gold-Standard Long-Tail Splits for iNaturalist 2018 (k=4071, natural + imb=10, 20, 50, 100).

Strictly matches the BaCon / ProtoTail gold-standard split format:
- l_k_uq_idxs.pt (torch.Tensor[int64])
- unl_k_uq_idxs.pt (torch.Tensor[int64])
- unl_unk_uq_idxs.pt (torch.Tensor[int64])
- meta.json
- split_meta.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch


def get_lt_dist(cls_num: int, img_max: int, imb_factor: float) -> list[int]:
    """Compute exponential Pareto long-tail distribution profile."""
    if imb_factor <= 0.01 and img_max <= 30:
        return [
            max(1, int(np.ceil(img_max * (imb_factor ** (i / (cls_num - 1.0))))))
            for i in range(cls_num)
        ]
    return [
        max(1, int(round(img_max * (imb_factor ** (i / (cls_num - 1.0))))))
        for i in range(cls_num)
    ]


def bacon_buckets(counts: list[int] | np.ndarray) -> tuple[int, int, int, int, int]:
    """Calculate BaCon Many / Medium / Few bucket thresholds and counts."""
    c = np.asarray(counts)
    s = np.sort(c)[::-1]
    mt = s[len(c) // 3]
    ft = s[(2 * len(c)) // 3]
    many = int((c > mt).sum())
    med = int(((c <= mt) & (c >= ft)).sum())
    few = int((c < ft).sum())
    return many, med, few, int(mt), int(ft)


def generate_inat2018_splits(
    json_path: Path,
    output_roots: list[Path],
    n_total: int = 8142,
    k: int = 4071,
    img_max: int = 50,
    seed: int = 416,
    imbs: list[int] = [10, 20, 50, 100],
    include_natural: bool = True,
) -> list[dict]:
    """Generate canonical splits for iNaturalist 2018."""
    print("=" * 80)
    print(f"LOADING INATURALIST 2018 METADATA FROM: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    images = meta["images"]
    annotations = meta["annotations"]
    print(f"Loaded {len(images):,} images and {len(annotations):,} annotations.")

    ann_by_image_id = {a["image_id"]: a["category_id"] for a in annotations}
    targets = np.array([ann_by_image_id[img["id"]] for img in images], dtype=np.int64)

    assert len(targets) == len(images), f"Mismatch: {len(targets)} vs {len(images)}"
    unique_cats = np.unique(targets)
    assert len(unique_cats) == n_total, f"Expected {n_total} categories, got {len(unique_cats)}"
    print(f"Verified {n_total} contiguous categories [0 .. {n_total - 1}].")

    cls_to_pos = {c: np.where(targets == c)[0] for c in range(n_total)}
    known_classes = list(range(k))
    novel_classes = list(range(k, n_total))

    splits_to_process = []
    if include_natural:
        splits_to_process.append(("natural", None))
    for imb in imbs:
        splits_to_process.append((f"imb{imb}", imb))

    summary = []

    for split_tag, imb in splits_to_process:
        folder_name = f"inat2018_k{k}_{split_tag}"
        print(f"\nProcessing {folder_name}...")
        rng = np.random.RandomState(seed)

        counts = {}
        if imb is None:
            # Natural real-world long-tail distribution: keep 100% of samples
            for c in range(n_total):
                counts[c] = len(cls_to_pos[c])
        else:
            # Controlled Pareto long-tail distribution
            prof_k = get_lt_dist(k, img_max, 1.0 / imb)
            prof_u = get_lt_dist(n_total - k, img_max, 1.0 / imb)

            # Sort classes by available count descending so head classes have ample images
            k_sorted = sorted(known_classes, key=lambda c: len(cls_to_pos[c]), reverse=True)
            u_sorted = sorted(novel_classes, key=lambda c: len(cls_to_pos[c]), reverse=True)

            for rank, c in enumerate(k_sorted):
                counts[c] = min(prof_k[rank], len(cls_to_pos[c]))
            for rank, c in enumerate(u_sorted):
                counts[c] = min(prof_u[rank], len(cls_to_pos[c]))

        l_k, unl_k, unl_unk = [], [], []
        for c in range(n_total):
            pos = cls_to_pos[c]
            req = counts[c]
            chosen = rng.choice(pos, size=req, replace=False)

            if c < k:
                # 50/50 split for Known classes
                n_l = int(round(len(chosen) * 0.5))
                if len(chosen) >= 2 and n_l == 0:
                    n_l = 1
                if len(chosen) >= 2 and n_l == len(chosen):
                    n_l = len(chosen) - 1
                if len(chosen) == 1:
                    n_l = 1

                take = rng.choice(chosen, size=n_l, replace=False)
                rest = np.setdiff1d(chosen, take)
                l_k.extend(take.tolist())
                unl_k.extend(rest.tolist())
            else:
                unl_unk.extend(chosen.tolist())

        l_k = np.sort(np.asarray(l_k, dtype=np.int64))
        unl_k = np.sort(np.asarray(unl_k, dtype=np.int64))
        unl_unk = np.sort(np.asarray(unl_unk, dtype=np.int64))

        # Strict validation assertions
        assert len(set(l_k) & set(unl_k)) == 0, "l_k and unl_k overlap!"
        assert len(set(l_k) & set(unl_unk)) == 0, "l_k and unl_unk overlap!"
        assert len(set(unl_k) & set(unl_unk)) == 0, "unl_k and unl_unk overlap!"
        assert max(l_k.max(), unl_k.max(), unl_unk.max()) < len(targets), "Index out of range!"

        c_lk = Counter(targets[l_k])
        c_uk = Counter(targets[unl_k])
        c_uunk = Counter(targets[unl_unk])

        assert set(c_lk.keys()).issubset(set(known_classes)), "Novel in l_k!"
        assert set(c_uk.keys()).issubset(set(known_classes)), "Novel in unl_k!"
        assert set(c_uunk.keys()) == set(novel_classes), "Novel class missing in unl_unk!"
        assert set(c_lk.keys()) == set(known_classes), "Known class missing in l_k!"

        k_counts = [c_lk[c] + c_uk[c] for c in known_classes]
        u_counts = [c_uunk[c] for c in novel_classes]

        km, kmed, kf, kmt, kft = bacon_buckets(k_counts)
        um, umed, uf, umt, uft = bacon_buckets(u_counts)

        assert km > 0 and kmed > 0 and kf > 0, f"Empty bucket in Known: {km}/{kmed}/{kf}"
        assert um > 0 and umed > 0 and uf > 0, f"Empty bucket in Novel: {um}/{umed}/{uf}"

        effective_imb = imb if imb is not None else 500
        meta_dict = {
            "dataset": "inat2018",
            "split_name": folder_name,
            "imb_ratio": effective_imb,
            "seed": seed,
            "img_max": img_max if imb is not None else int(max(k_counts + u_counts)),
            "anno_ratio": 50.0,
            "num_classes": n_total,
            "num_known_classes": k,
            "known_classes": known_classes,
            "novel_classes": novel_classes,
            "counts_summary": {
                "l_k": len(l_k),
                "unl_k": len(unl_k),
                "unl_unk": len(unl_unk),
                "total_train": len(l_k) + len(unl_k) + len(unl_unk),
            },
            "buckets": {"K": [km, kmed, kf], "U": [um, umed, uf]},
            "bucket_thresholds": {"K_mt": kmt, "K_ft": kft, "U_mt": umt, "U_ft": uft},
        }

        # Write to all requested destination directories
        for out_root in output_roots:
            dest_dir = out_root / folder_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            torch.save(torch.from_numpy(l_k), dest_dir / "l_k_uq_idxs.pt")
            torch.save(torch.from_numpy(unl_k), dest_dir / "unl_k_uq_idxs.pt")
            torch.save(torch.from_numpy(unl_unk), dest_dir / "unl_unk_uq_idxs.pt")

            with open(dest_dir / "split_meta.json", "w", encoding="utf-8") as fp:
                json.dump(meta_dict, fp, indent=2)
            with open(dest_dir / "meta.json", "w", encoding="utf-8") as fp:
                json.dump(meta_dict, fp, indent=2)

        zero_uk = sum(1 for c in known_classes if c_uk[c] == 0)
        row = {
            "folder": folder_name,
            "imb": effective_imb,
            "l_k": len(l_k),
            "unl_k": len(unl_k),
            "unl_unk": len(unl_unk),
            "total": len(l_k) + len(unl_k) + len(unl_unk),
            "K_MMF": f"{km}/{kmed}/{kf}",
            "U_MMF": f"{um}/{umed}/{uf}",
            "0_uk": zero_uk,
        }
        summary.append(row)
        print(
            f"  [SUCCESS] {folder_name}: Total={row['total']:,} | "
            f"l_k={row['l_k']:,}, unl_k={row['unl_k']:,}, unl_unk={row['unl_unk']:,} | "
            f"K_MMF={row['K_MMF']}, U_MMF={row['U_MMF']}"
        )

    print("\n" + "=" * 80)
    print("ALL INATURALIST 2018 SPLITS GENERATED AND VERIFIED SUCCESSFULLY!")
    print("=" * 80)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate Gold-Standard Splits for iNaturalist 2018.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Path containing train2018.json (default: current directory)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        nargs="+",
        default=[Path("data_uq_idxs")],
        help="Output directories for generated splits (default: data_uq_idxs)",
    )
    parser.add_argument("--img-max", type=int, default=50, help="Max images per class for controlled imb (default: 50)")
    parser.add_argument("--seed", type=int, default=416, help="Random seed (default: 416)")
    parser.add_argument(
        "--imbs",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100],
        help="Imbalance ratios to generate (default: 10 20 50 100)",
    )
    parser.add_argument(
        "--no-natural",
        action="store_true",
        help="Skip generating inat2018_k4071_natural",
    )
    args = parser.parse_args()

    json_path = args.data_dir / "train2018.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Cannot find train2018.json at: {json_path}")

    generate_inat2018_splits(
        json_path=json_path,
        output_roots=args.out_dir,
        img_max=args.img_max,
        seed=args.seed,
        imbs=args.imbs,
        include_natural=not args.no_natural,
    )


if __name__ == "__main__":
    main()
