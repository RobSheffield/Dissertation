import os
import k_fold


if __name__ == "__main__":
    k=10
    device = os.getenv("K_FOLD_DEVICE", "auto")
    k_fold.create_bias_folds("Castings", "Bias_folds", k, testSize=0.1,seed=42)
    k_fold.build_train_val_sets("Bias_folds")
    k_fold.train_all("Bias_folds","biased_models", device=device)
    k_fold.mAP_on_test_set("Bias_folds/test","biased_models")   

    k_fold.create_folds("Castings", "Folds", k, testSize=0.1,seed=42)
    k_fold.build_train_val_sets("Folds")
    k_fold.train_all("Folds","unbiased_models", device=device)
    k_fold.mAP_on_test_set("Folds/test","unbiased_models")