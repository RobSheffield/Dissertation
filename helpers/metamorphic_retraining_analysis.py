import os
import k_fold


if __name__ == "__main__":
    k_fold.create_folds("Castings", "Folds_augmented", k=5, testSize=0,seed=42)
    k_fold.create_folds("Castings", "Folds_non_augmented", k=5, testSize=0,seed=42)
    k_fold.build_train_val_sets("Folds_non_augmented", apply_training_augmentations=False)
    k_fold.build_train_val_sets("Folds_non_augmented", apply_training_augmentations=True)
    k_fold.train_all("Folds_non_augmented","models_vertical_flip", device="auto",flips=True)
    k_fold.train_all("Folds_augmented","models_augmented", device="auto")
    k_fold.train_all("Folds_non_augmented","models_non_augmented", device="auto")
