from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from helpers import k_fold


REPORT_NAME = "lca_kfold_intervals_report.csv"
WORK_ROOT_NAME = "_lca_kfold_intervals_work"


def _resolve_output_path(raw_out: str) -> Path:
    output = Path(raw_out).expanduser()
    if output.suffix.lower() == ".csv":
        return output
    if output.exists() and output.is_dir():
        return output / REPORT_NAME
    if not output.exists() and str(raw_out).endswith(("/", "\\")):
        return output / REPORT_NAME
    return output / REPORT_NAME if not output.suffix else output


def _prepare_work_dirs(report_path: Path, partition_mode: str) -> tuple[Path, Path]:
    work_root = report_path.parent / WORK_ROOT_NAME
    if work_root.exists():
        shutil.rmtree(work_root)

    folds_root = work_root / partition_mode
    models_root = work_root / "models"
    folds_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    return folds_root, models_root


def _create_partitions(data_root: str, folds_root: Path, k: int, holdout: float, partition_mode: str) -> None:
    if partition_mode == "image":
        k_fold.create_bias_folds(data_root, str(folds_root), k=k, testSize=holdout, seed=42)
        return

    k_fold.create_folds(data_root, str(folds_root), k=k, testSize=holdout, seed=42)


def run_experiment(
    data_root: str,
    out: str,
    k: int = 5,
    epochs: int = 150,
    batch_size: int = 12,
    img_size: int = 640,
    weights: str = "yolo11n.pt",
    apply_augmentations: bool = False,
    horizontal_mirror: bool = True,
    vertical_mirror: bool = False,
    rotate_90: bool = False,
    rotate_180: bool = False,
    rotate_270: bool = False,
    initial_train: float = 0.2,
    expansion_step: float = 0.1,
    holdout: float = 0.2,
    partition_mode: str = "folder",
) -> Path:
    report_path = _resolve_output_path(out)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    folds_root, _ = _prepare_work_dirs(report_path, partition_mode)
    results_root = report_path.parent
    test_dir = folds_root / "test"

    augmentation_options = {
        "horizontal_mirror": horizontal_mirror,
        "vertical_mirror": vertical_mirror,
        "rot90": rotate_90,
        "rot180": rotate_180,
        "rot270": rotate_270,
    }
    if apply_augmentations and not any(augmentation_options.values()):
        augmentation_options = {
            "horizontal_mirror": True,
            "vertical_mirror": True,
            "rot90": True,
            "rot180": True,
            "rot270": True,
        }

    try:
        _create_partitions(data_root, folds_root, k=k, holdout=holdout, partition_mode=partition_mode)
        k_fold.build_train_val_sets(
            str(folds_root),
            apply_training_augmentations=apply_augmentations,
            augmentation_options=augmentation_options,
        )
        k_fold.train_all(
            str(folds_root),
            model_dir=str(results_root),
            device="auto",
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            weights=weights,
            group_outputs_by_fold=True,
        )

        test_report = None
        if test_dir.is_dir():
            test_report = k_fold.mAP_on_test_set(
                str(test_dir),
                str(results_root),
                imgsz=img_size,
                validation_output_root=str(results_root),
            )

        with open(report_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=[
                    "data_root",
                    "partition_mode",
                    "k",
                    "epochs",
                    "batch_size",
                    "img_size",
                    "weights",
                    "apply_augmentations",
                    "horizontal_mirror",
                    "vertical_mirror",
                    "rotate_90",
                    "rotate_180",
                    "rotate_270",
                    "initial_train",
                    "expansion_step",
                    "holdout",
                    "folds_root",
                    "models_root",
                    "test_report",
                    "status",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "data_root": str(Path(data_root).expanduser()),
                    "partition_mode": partition_mode,
                    "k": k,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "img_size": img_size,
                    "weights": weights,
                    "apply_augmentations": apply_augmentations,
                    "horizontal_mirror": horizontal_mirror,
                    "vertical_mirror": vertical_mirror,
                    "rotate_90": rotate_90,
                    "rotate_180": rotate_180,
                    "rotate_270": rotate_270,
                    "initial_train": initial_train,
                    "expansion_step": expansion_step,
                    "holdout": holdout,
                    "folds_root": "",
                    "models_root": str(results_root),
                    "test_report": test_report or "",
                    "status": "ok",
                }
            )

        print(f"Saved report to {report_path}")
        return report_path
    finally:
        shutil.rmtree(folds_root.parent, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LCA k-fold interval experiment.")
    parser.add_argument("--data-root", default="Castings", help="Root folder containing the castings dataset.")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "experiments" / REPORT_NAME), help="Output CSV path or directory.")
    parser.add_argument("--k", type=int, default=5, help="Number of folds to create.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs for each fold.")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size for training each fold.")
    parser.add_argument("--img-size", type=int, default=640, choices=(640, 1280), help="Training resolution size.")
    parser.add_argument("--weights", default="yolo11n.pt", help="Pretrained weights file to start each fold from.")
    parser.add_argument("--apply-augmentations", action="store_true", help="Apply training augmentations to each fold.")
    parser.add_argument("--horizontal-mirror", dest="horizontal_mirror", action="store_true", help="Enable horizontal mirroring augmentation.")
    parser.add_argument("--no-horizontal-mirror", dest="horizontal_mirror", action="store_false", help="Disable horizontal mirroring augmentation.")
    parser.add_argument("--vertical-mirror", action="store_true", help="Enable vertical mirroring augmentation.")
    parser.add_argument("--rotate-90", action="store_true", help="Enable 90 degree rotation augmentation.")
    parser.add_argument("--rotate-180", action="store_true", help="Enable 180 degree rotation augmentation.")
    parser.add_argument("--rotate-270", action="store_true", help="Enable 270 degree rotation augmentation.")
    parser.set_defaults(horizontal_mirror=True)
    parser.add_argument("--initial-train", type=float, default=0.2, help="Initial training fraction for interval runs.")
    parser.add_argument("--expansion-step", type=float, default=0.1, help="Training fraction increment for interval runs.")
    parser.add_argument("--holdout", type=float, default=0.2, help="Holdout fraction used to build the test split.")
    parser.add_argument(
        "--partition-mode",
        choices=("folder", "image"),
        default="folder",
        help="How to partition the dataset into folds.",
    )
    args = parser.parse_args(argv)

    run_experiment(
        data_root=args.data_root,
        out=args.out,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        weights=args.weights,
        apply_augmentations=args.apply_augmentations,
        horizontal_mirror=args.horizontal_mirror,
        vertical_mirror=args.vertical_mirror,
        rotate_90=args.rotate_90,
        rotate_180=args.rotate_180,
        rotate_270=args.rotate_270,
        initial_train=args.initial_train,
        expansion_step=args.expansion_step,
        holdout=args.holdout,
        partition_mode=args.partition_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
