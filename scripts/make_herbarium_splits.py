"""Generate Gold-Standard Long-Tail Splits for Herbarium 19 (k=341, imb=10, 20, 50, 100).

Strictly matches the BaCon / ProtoTail gold-standard split format:
- l_k_uq_idxs.pt (torch.Tensor[int64])
- unl_k_uq_idxs.pt (torch.Tensor[int64])
- unl_unk_uq_idxs.pt (torch.Tensor[int64])
- meta.json
- split_meta.json
"""

import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import ImageFolder


def get_lt_dist(cls_num: int, img_max: int, imb_factor: float) -> list[int]:
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
    c = np.asarray(counts)
    s = np.sort(c)[::-1]
    mt = s[len(c) // 3]
    ft = s[(2 * len(c)) // 3]
    many = int((c > mt).sum())
    med = int(((c <= mt) & (c >= ft)).sum())
    few = int((c < ft).sum())
    return many, med, few, int(mt), int(ft)


def generate_herbarium_splits(
    data_dir: Path,
    output_roots: list[Path],
    n_total: int = 683,
    k: int = 341,
    img_max: int = 50,
    seed: int = 416,
    imbs: list[int] = [10, 20, 50, 100],
) -> list[dict]:
    print("=" * 80)
    print(f"LOADING HERBARIUM 19 DATASET FROM: {data_dir}")
    ds = ImageFolder(str(data_dir))
    targets = np.array(ds.targets)
    assert len(targets) == 34225, f"Expected 34225 targets, got {len(targets)}"
    assert len(ds.classes) == n_total, f"Expected {n_total} classes, got {len(ds.classes)}"
    print(f"Dataset successfully loaded: {len(targets)} images across {n_total} classes.")

    cls_to_pos = {c: np.where(targets == c)[0] for c in range(n_total)}
    summary = []

    for imb in imbs:
        folder_name = f"herbarium19_k{k}_imb{imb}"
        print(f"\nProcessing {folder_name} (imb_ratio={imb})...")

        rng = np.random.RandomState(seed)
        prof = get_lt_dist(n_total, img_max, 1.0 / imb)
        ranks = rng.permutation(n_total)

        known_classes = list(range(k))
        novel_classes = list(range(k, n_total))

        counts = {}
        for c in range(n_total):
            counts[c] = min(prof[ranks[c]], len(cls_to_pos[c]))

        l_k, unl_k, unl_unk = [], [], []
        for c in range(n_total):
            pos = cls_to_pos[c]
            chosen = rng.choice(pos, size=counts[c], replace=False)
            if c < k:
                # 50/50 split for known classes
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

        meta_dict = {
            "dataset": "herbarium19",
            "imb_ratio": imb,
            "seed": seed,
            "img_max": img_max,
            "anno_ratio": 50.0,
            "num_classes": n_total,
            "num_known_classes": k,
            "known_classes": known_classes,
            "novel_classes": novel_classes,
            "rank_order": ranks.tolist(),
            "requested_counts": {str(c): int(counts[c]) for c in range(n_total)},
            "actual_counts": {
                str(c): int(c_lk[c] + c_uk[c] if c < k else c_uunk[c])
                for c in range(n_total)
            },
            "buckets": {"K": [km, kmed, kf], "U": [um, umed, uf]},
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
        summary.append({
            "folder": folder_name,
            "imb": imb,
            "l_k": len(l_k),
            "unl_k": len(unl_k),
            "unl_unk": len(unl_unk),
            "total": len(l_k) + len(unl_k) + len(unl_unk),
            "K_MMF": f"{km}/{kmed}/{kf}",
            "U_MMF": f"{um}/{umed}/{uf}",
            "0_uk": zero_uk,
        })
        print(
            f"  [SUCCESS] {folder_name}: Total={len(l_k)+len(unl_k)+len(unl_unk)} | "
            f"l_k={len(l_k)}, unl_k={len(unl_k)}, unl_unk={len(unl_unk)} | "
            f"K_MMF={km}/{kmed}/{kf}, U_MMF={um}/{umed}/{uf}"
        )

    print("\n" + "=" * 80)
    print("ALL 4 HERBARIUM 19 SPLITS GENERATED AND VERIFIED SUCCESSFULLY!")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    data_path = Path(r"D:\data\Herbarium 19\small-train\small-train")
    if not data_path.is_dir():
        data_path = Path(r"D:\data\Herbarium 19\train\train")

    dest_roots = [
        Path(r"C:\Users\tam\Documents\GitHub\gcd-hosts\prototail\data_uq_idxs"),
        Path(r"C:\Users\tam\Documents\GitHub\BaCon\data_uq_idxs"),
    ]

    generate_herbarium_splits(data_path, dest_roots)
