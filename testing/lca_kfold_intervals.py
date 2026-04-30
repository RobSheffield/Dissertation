"""
LCA over dataset fractions using folder-based 5-fold cross-validation.

- Uses Castings-style folder independence: each top-level folder is one unit in the split
- Performs K=5 cross-validation with no separate holdout set
- For each fold, evaluates fixed validation data while expanding the training subset in 10% steps
- Writes a CSV report with per-fold rows and fraction-level averages

Usage:
    python testing/lca_kfold_intervals.py --data-root Castings --out testing/lca_kfold_intervals_report.csv
"""
from pathlib import Path
import argparse
import random
import csv
import os
import sys
from typing import List, Dict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


# Placeholder LCA computation; replace with real implementation
def compute_lca(train_image_paths: List[Path], val_image_paths: List[Path]) -> Dict[str, float]:
    """Compute LCA metric on val images using model/data from train images.
    This is a deterministic placeholder returning synthetic metrics.
    Replace with actual LCA pipeline call.
    """
    # For now return random-ish metrics for demonstration
    # In real use, this function should train/model or compute scores and return metrics
    rng = np.random.default_rng(len(train_image_paths) + len(val_image_paths))
    return {
        "lca_score": float(rng.uniform(0.2, 0.9)),
        "precision": float(rng.uniform(0.2, 0.9)),
        "recall": float(rng.uniform(0.1, 0.8)),
    }


def gather_groups(data_root: Path) -> List[Path]:
    """Return list of group directories under data_root.
    Each group is a folder that contains `images/` (or images directly).
    """
    groups = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    return groups


def list_images_in_group(group: Path) -> List[Path]:
    """List image files for a group. Supports nested `images/` folder or direct images.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = []
    images_dir = group / "images"
    if images_dir.exists() and images_dir.is_dir():
        for p in images_dir.rglob("*"):
            if p.suffix.lower() in exts:
                images.append(p)
    else:
        for p in group.rglob("*"):
            if p.suffix.lower() in exts:
                images.append(p)
    return sorted(images)


def iter_kfold_indices(item_count: int, k: int, seed: int = 42):
    """Yield train/test index splits for a shuffled K-fold partition."""
    if k < 2:
        raise ValueError("k must be at least 2")
    if item_count < k:
        raise ValueError(f"Need at least {k} items, found {item_count}")

    indices = list(range(item_count))
    random.Random(seed).shuffle(indices)

    base_size, remainder = divmod(item_count, k)
    fold_sizes = [base_size + (1 if fold_index < remainder else 0) for fold_index in range(k)]

    start = 0
    for fold_size in fold_sizes:
        test_indices = indices[start:start + fold_size]
        train_indices = indices[:start] + indices[start + fold_size:]
        yield train_indices, test_indices
        start += fold_size


def run_intervals(data_root: Path, out_csv: Path, k: int = 5):
    groups = gather_groups(data_root)
    if len(groups) < k:
        raise SystemExit(f"Need at least {k} groups; found {len(groups)}")

    # Build a list of (group, images)
    group_images = [(g, list_images_in_group(g)) for g in groups]

    fractions = [i / 10.0 for i in range(1, 11)]  # 0.1 .. 1.0

    header = ["fraction", "fold", "lca_score", "precision", "recall", "train_images", "val_images"]
    rows = []

    rng = random.Random(42)

    for frac in fractions:
        for fold_idx, (train_idx, test_idx) in enumerate(iter_kfold_indices(len(group_images), k, seed=42), start=1):
            # Build train image list (folder-based sampling)
            train_groups = [group_images[i] for i in train_idx]
            test_groups = [group_images[i] for i in test_idx]

            # Select training data by whole groups (folders) rather than
            # sampling individual images. This preserves the folder-level
            # independence characteristics required by the evaluation.
            if not train_groups:
                print(f"Warning: fold {fold_idx} has zero train groups")
                continue

            shuffled_groups = list(train_groups)
            rng.shuffle(shuffled_groups)
            sample_g = max(1, int(len(shuffled_groups) * frac))
            selected_groups = shuffled_groups[:sample_g]

            sampled_train = []
            for g, imgs in selected_groups:
                sampled_train.extend(imgs)

            if len(sampled_train) == 0:
                print(f"Warning: fold {fold_idx} selected groups contain zero images")
                continue

            # Validation/test images are all images in test_groups
            val_image_paths = []
            for g, imgs in test_groups:
                val_image_paths.extend(imgs)

            metrics = compute_lca(sampled_train, val_image_paths)
            rows.append([
                f"{int(frac * 100)}%",
                fold_idx,
                metrics["lca_score"],
                metrics["precision"],
                metrics["recall"],
                len(sampled_train),
                len(val_image_paths),
            ])

    # Fraction-level summary rows
    summary_rows = []
    for frac_label in [f"{i}%" for i in range(10, 101, 10)]:
        frac_rows = [r for r in rows if r[0] == frac_label]
        if not frac_rows:
            continue
        lca_values = [float(r[2]) for r in frac_rows]
        precision_values = [float(r[3]) for r in frac_rows]
        recall_values = [float(r[4]) for r in frac_rows]
        summary_rows.append([
            frac_label,
            "AVERAGE",
            float(np.mean(lca_values)),
            float(np.mean(precision_values)),
            float(np.mean(recall_values)),
            float(np.mean([float(r[5]) for r in frac_rows])),
            float(np.mean([float(r[6]) for r in frac_rows])),
        ])

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)
        writer.writerow([])
        writer.writerow(["fraction", "fold", "lca_score", "precision", "recall", "train_images", "val_images"])
        for r in summary_rows:
            writer.writerow(r)

    print(f"Wrote report to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="Castings", help="Path to folder containing top-level group folders (default: Castings)")
    parser.add_argument("--out", default="testing/lca_kfold_intervals_report.csv", help="CSV output path")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default 5)")
    args = parser.parse_args()

    data_root = _resolve_path(args.data_root)
    out_csv = _resolve_path(args.out)
    run_intervals(data_root, out_csv, k=args.k)
