import os
import shutil
import k_fold

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    folds_non_aug = os.path.join(project_root, "Folds_non_augmented")
    folds_aug = os.path.join(project_root, "Folds_augmented")

    k_fold.create_folds("Castings", "Folds_augmented", k=10, testSize=0, seed=42)
    k_fold.create_folds("Castings", "Folds_non_augmented", k=10, testSize=0, seed=42) 
    k_fold.build_train_val_sets("Folds_non_augmented", apply_training_augmentations=False)
    k_fold.build_train_val_sets("Folds_non_augmented", apply_training_augmentations=True)
    k_fold.train_all("Folds_non_augmented", "models_vertical_flip", device="auto", flips=True,epochs=100)
    k_fold.train_all("Folds_augmented", "models_augmented", device="auto", epochs=100)
    k_fold.train_all("Folds_non_augmented", "models_non_augmented", device="auto", epochs=100)

