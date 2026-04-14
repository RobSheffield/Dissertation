import os
import shutil
import k_fold


def cleanup_dir(path):
    # Resolve exactly like k_fold does for relative paths.
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))
    if os.path.isdir(abs_path):
        shutil.rmtree(abs_path, ignore_errors=True)


if __name__ == "__main__":
    device = os.getenv("K_FOLD_DEVICE", "auto")

    # k = 5 with test split
    try:
        k = 5
        folds_path = "Folds_strat_5"
        model_path = "strat_5_unbiased_models"

        k_fold.create_folds("Castings", folds_path, k=k, testSize=0.2, seed=42)
        k_fold.build_train_val_sets(folds_path)
        k_fold.train_all(folds_path, model_path, device=device)
        k_fold.mAP_on_test_set(f"{folds_path}/test", model_path)
    finally:
        cleanup_dir(folds_path)

    # k = 5 no test split
    try:
        k = 5
        folds_path = "Folds_strat_5_no_test"
        model_path = "strat_5_unbiased_models"

        k_fold.create_folds("Castings", folds_path, k=k, testSize=0, seed=42)
        k_fold.build_train_val_sets(folds_path)
        k_fold.train_all(folds_path, model_path, device=device)
    finally:
        cleanup_dir(folds_path)

    # k = 10 with test split
    try:
        k = 10
        folds_path = "Folds_strat_10"
        model_path = "strat_10_unbiased_models"

        k_fold.create_folds("Castings", folds_path, k=k, testSize=0.2, seed=42)
        k_fold.build_train_val_sets(folds_path)
        k_fold.train_all(folds_path, model_path, device=device)
        k_fold.mAP_on_test_set(f"{folds_path}/test", model_path)
    finally:
        cleanup_dir(folds_path)
