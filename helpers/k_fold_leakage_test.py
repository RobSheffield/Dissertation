import os
import k_fold


if __name__ == "__main__":
    k=5
    device = os.getenv("K_FOLD_DEVICE", "auto")
    '''    k_fold.create_bias_folds("Castings", "Bias_folds", k, testSize=0.1,seed=42)
    k_fold.build_train_val_sets("Bias_folds")
    k_fold.train_all("Bias_folds","biased_models", device=device)
    k_fold.mAP_on_test_set("Bias_folds/test","biased_models")   '''
    k_fold.create_folds("Castings", "Folds_strat_5", k=k, testSize=0.2, seed=42)
    k_fold.build_train_val_sets("Folds_strat_5")
    k_fold.train_all("Folds_strat_5", "strat_5_unbiased_models", device=device)
    k_fold.mAP_on_test_set("Folds_strat_5/test", "strat_5_unbiased_models")
    k_fold.create_folds("Castings", "Folds_strat_5_no_test", k=k, testSize=0, seed=42)
    k_fold.build_train_val_sets("Folds_strat_5_no_test")
    k_fold.train_all("Folds_strat_5_no_test", "strat_5_unbiased_models", device=device)
    k=10
    k_fold.create_folds("Castings", "Folds_strat_10", k=k, testSize=0.2, seed=42)
    k_fold.build_train_val_sets("Folds_strat_10")
    k_fold.train_all("Folds_strat_10", "strat_10_unbiased_models", device=device)
    k_fold.mAP_on_test_set("Folds_strat_10/test", "strat_10_unbiased_models")