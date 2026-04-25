import os
from pathlib import Path

import k_fold


def _assert_train_val_ready(folds_root):
    folds_root = Path(folds_root)
    for fold_dir in sorted(p for p in folds_root.iterdir() if p.is_dir() and p.name.startswith("fold_")):
        train_img = fold_dir / "images" / "train"
        train_lbl = fold_dir / "labels" / "train"
        val_img = fold_dir / "images" / "val"
        val_lbl = fold_dir / "labels" / "val"
        missing = [p for p in (train_img, train_lbl, val_img, val_lbl) if not p.is_dir()]
        if missing:
            missing_text = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(f"{fold_dir.name} is missing expected train/val directories: {missing_text}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    castings_root = project_root / "Castings"
    folds_non_aug = project_root / "Folds_non_augmented"
    folds_aug = project_root / "Folds_augmented"

    k_fold.create_folds(str(castings_root), str(folds_aug), k=10, testSize=0, seed=42)
    k_fold.create_folds(str(castings_root), str(folds_non_aug), k=10, testSize=0, seed=42)

    k_fold.build_train_val_sets(str(folds_non_aug), apply_training_augmentations=False)
    k_fold.build_train_val_sets(str(folds_aug), apply_training_augmentations=True)

    _assert_train_val_ready(folds_non_aug)
    _assert_train_val_ready(folds_aug)

    k_fold.train_all(str(folds_non_aug), "models_vertical_flip", device="auto", flips=True, epochs=250)
    k_fold.train_all(str(folds_aug), "models_augmented", device="auto", epochs=250)
    k_fold.train_all(str(folds_non_aug), "models_non_augmented", device="auto", epochs=250)

