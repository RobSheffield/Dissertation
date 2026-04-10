import os
import k_fold


if __name__ == "__main__":
    k_fold.create_folds("Castings", "Folds_augmented", 10, testSize=0,seed=42,augment=True)
    k_fold.create_folds("Castings", "Folds_non_augmented", 10, testSize=0,seed=42,augment=False)
    k_fold.train_all("Folds_augmented","models_augmented", device="auto")
    k_fold.train_all("Folds_non_augmented","models_non_augmented", device="auto")
    k_fold.train_all("Folds_non_augmented","models_vertical_flip", device="auto",flips=True)