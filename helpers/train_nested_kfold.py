"""
Nested K-Fold Training Pipeline
Orchestrates: create folds → build train/val → train with inner tuning → evaluate outer folds
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from helpers.k_fold import (
    create_nested_folds,
    build_nested_train_val_sets,
    train_nested_kfold,
    evaluate_nested_kfold_outer
)

def main():
    """Execute full nested K-fold pipeline."""
    
    # Load parameters from environment variables
    k_outer = int(os.getenv("K_OUTER", "5"))
    k_inner = int(os.getenv("K_INNER", "4"))
    test_size = float(os.getenv("TEST_SIZE", "0.2"))
    epochs = int(os.getenv("EPOCHS", "150"))
    apply_augmentations = os.getenv("APPLY_AUGMENTATIONS", "false").lower() == "true"
    device = os.getenv("K_FOLD_DEVICE", "0,1,2,3")
    
    # Define paths
    image_path = "Castings"
    folds_path = f"nested_folds_k{k_outer}_inner_k{k_inner}"
    model_dir = f"models_nested_k{k_outer}_inner_k{k_inner}"
    results_output = f"nested_kfold_results_k{k_outer}_k{k_inner}.txt"
    
    print("="*70)
    print("NESTED K-FOLD CROSS-VALIDATION PIPELINE")
    print("="*70)
    print(f"Configuration:")
    print(f"  Outer Folds (K): {k_outer}")
    print(f"  Inner Folds per Outer (K): {k_inner}")
    print(f"  Test Set Size: {test_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Apply Augmentations: {apply_augmentations}")
    print(f"  GPU Device(s): {device}")
    print(f"  Folds Path: {folds_path}")
    print(f"  Model Directory: {model_dir}")
    print("="*70)
    
    # STEP 1: Create nested folds
    print(f"\n[STEP 1/4] Creating nested fold structure...")
    print(f"  Splitting {image_path} into {k_outer} outer folds × {k_inner} inner folds")
    create_nested_folds(
        image_path=image_path,
        output_path=folds_path,
        k_outer=k_outer,
        k_inner=k_inner,
        testSize=test_size,
        seed=42
    )
    print(f"✓ Nested folds created at: {folds_path}")
    
    # STEP 2: Build train/val sets for each inner fold
    print(f"\n[STEP 2/4] Building train/val splits for inner folds...")
    build_nested_train_val_sets(
        nested_folds_path=folds_path,
        apply_training_augmentations=apply_augmentations
    )
    print(f"✓ Train/val splits created")
    
    # STEP 3: Train with inner fold hyperparameter tuning
    print(f"\n[STEP 3/4] Training models with inner fold validation...")
    print(f"  This will train {k_outer} × {k_inner} = {k_outer * k_inner} models total")
    inner_results = train_nested_kfold(
        nested_folds_path=folds_path,
        model_dir=model_dir,
        device=device,
        flips=apply_augmentations,
        epochs=epochs,
        save_best_inner=True
    )
    print(f"✓ Training completed")
    
    # STEP 4: Evaluate on unbiased outer folds
    print(f"\n[STEP 4/4] Evaluating on outer folds (unbiased evaluation)...")
    outer_results = evaluate_nested_kfold_outer(
        nested_folds_path=folds_path,
        model_dir=model_dir,
        results_output=results_output
    )
    print(f"✓ Evaluation completed")
    
    print("\n" + "="*70)
    print("NESTED K-FOLD PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nResults saved to: {results_output}")
    print(f"Models saved to: {model_dir}")
    print(f"Fold structure at: {folds_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
