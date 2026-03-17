import sys
import os
import random
import shutil
import datetime
import yaml
import re
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stages.train_model import train_yolo

from data.format_converter import convert_gt_to_yolo

def run_k_fold(image_path, output_path, k=5):
    '''K-fold across dataset - balances folds by image count per folder'''
    defect_folders = [f for f in os.listdir(image_path) 
                      if os.path.isdir(os.path.join(image_path, f))]
    
    # 1. Count images in each folder and store as (folder_name, count)
    folder_counts = []
    for folder in defect_folders:
        path = os.path.join(image_path, folder)
        count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if count > 0:
            folder_counts.append((folder, count))
    
    # 2. Sort folders by size (descending) - Largest first helps balancing
    folder_counts.sort(key=lambda x: x[1], reverse=True)

    # 3. Use Greedy Partitioning: Assign folder to the fold that currently has the fewest images
    folds = [[] for _ in range(k)]
    fold_totals = [0] * k

    for folder, count in folder_counts:
        # Find the fold index with the minimum current total images
        min_fold_idx = fold_totals.index(min(fold_totals))
        folds[min_fold_idx].append(folder)
        fold_totals[min_fold_idx] += count

    print(f"Fold distribution (image counts): {fold_totals}")

    # 4. proceed with building the folds
    for fold_idx, fold_folders in enumerate(folds):
        fold_name = f"fold_{fold_idx + 1}"
        fold_dir = os.path.join(output_path, fold_name)

        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)
            print(f"Cleared existing {fold_name}")

        img_dir = os.path.join(fold_dir, "images")
        lbl_dir = os.path.join(fold_dir, "labels")
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)

        print(f"Building {fold_name} ({len(fold_folders)} defect folders)...")

        for folder in fold_folders:
            folder_path = os.path.join(image_path, folder)
            # Find gt file directly in folder (e.g. ground_truth.txt)
            gt_file = None
            direct_gt = os.path.join(folder_path, "ground_truth.txt")
            if os.path.isfile(direct_gt):
                gt_file = direct_gt


            if gt_file:
                # Convert labels first
                output_labels = os.path.join(fold_dir, "labels_temp")
                os.makedirs(output_labels, exist_ok=True)
                success = convert_gt_to_yolo(gt_file, folder_path, output_labels, class_id=0)
                if not success:
                    print(f"WARNING: No gt file found for {folder}, skipping entire folder.")
                    continue

                # Track which stems have labels
                converted_stems = set()
                for lbl_file in os.listdir(output_labels):
                    item_path = os.path.join(output_labels, lbl_file)
                    if os.path.isfile(item_path) and lbl_file.endswith('.txt'):
                        dst_lbl = os.path.join(lbl_dir, lbl_file)
                        shutil.move(item_path, dst_lbl)
                        stem = os.path.splitext(lbl_file)[0]
                        converted_stems.add(stem)

                shutil.rmtree(output_labels)

                # Only copy images that have a matching label
                for file in os.listdir(folder_path):
                    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    stem = os.path.splitext(file)[0]
                    if stem in converted_stems:
                        src = os.path.join(folder_path, file)
                        dst = os.path.join(img_dir, f"{folder}_{file}")
                        shutil.copy(src, dst)
                    else:
                        print(f"  Skipped (no label): {file}")
            else:
                print(f"WARNING: No gt file found for {folder}, skipping entire folder.")
                continue

        print(f"  -> {len(os.listdir(img_dir))} images, {len(os.listdir(lbl_dir))} labels in {fold_name}")

def train_k_fold(folds_path="Folds"):
    all_folds = sorted([
        d for d in os.listdir(folds_path) 
        if d.startswith("fold_") 
        and os.path.isdir(os.path.join(folds_path, d)) 
        and not d.endswith(".cache")                    
    ])

    print(f"Found {len(all_folds)} folds: {all_folds}")

    model_info_json = '{"name":"k_fold","model":"YOLOv5","number_of_images":"","date_time_trained":"","total_training_time":"","path":"","epoch":"","box_loss":"","cls_loss":"","mAP_50":"","mAP_50_95":"","precision":"","recall":"","dataset_config":"K-Fold","starting_model":"","folder_name":"","metamorphic_test_result":"","differential_test_result":"","fuzzing_test_result":""}'

    for fold in all_folds:
        print(f"\n{'='*60}")
        print(f"Processing {fold}")
        print(f"{'='*60}")
        
        fold_path = os.path.join(folds_path, fold)
        
        # Paths to restructured directories
        train_img_dir = os.path.join(fold_path, "images", "train")
        val_img_dir = os.path.join(fold_path, "images", "val")
        train_lbl_dir = os.path.join(fold_path, "labels", "train")
        val_lbl_dir = os.path.join(fold_path, "labels", "val")
        
        # Create COCO structure (fresh directories)
        for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Copy validation set (current fold's original images/labels)
        print(f"Setting up validation set from {fold}...")
        old_img_dir = os.path.join(fold_path, "images")
        old_lbl_dir = os.path.join(fold_path, "labels")
        
        # First, collect all label stems from validation set
        val_label_stems = set()
        if os.path.exists(old_lbl_dir) and os.path.isdir(old_lbl_dir):
            for lbl_file in os.listdir(old_lbl_dir):
                if os.path.isfile(os.path.join(old_lbl_dir, lbl_file)) and lbl_file.endswith('.txt'):
                    stem = os.path.splitext(lbl_file)[0]
                    val_label_stems.add(stem)
                    src = os.path.join(old_lbl_dir, lbl_file)
                    dst = os.path.join(val_lbl_dir, lbl_file)
                    shutil.copy2(src, dst)
        
        # Copy images that have matching labels (including empty/background)
        if os.path.exists(old_img_dir) and os.path.isdir(old_img_dir):
            for img_file in os.listdir(old_img_dir):
                if os.path.isdir(os.path.join(old_img_dir, img_file)):  # ← Skip directories
                    continue
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    stem = os.path.splitext(img_file)[0]
                    if stem in val_label_stems:  # Only copy if label exists (even if empty)
                        src = os.path.join(old_img_dir, img_file)
                        dst = os.path.join(val_img_dir, img_file)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
        
        val_img_count = len([f for f in os.listdir(val_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        val_lbl_count = len([f for f in os.listdir(val_lbl_dir) if f.endswith('.txt')])
        print(f"✓ Val: {val_img_count} images, {val_lbl_count} labels")
        
        # Copy training set (all other folds)
        print(f"Setting up training set from other folds...")
        for other_fold in all_folds:
            if other_fold == fold:
                continue
            
            src_img = os.path.join(folds_path, other_fold, "images")
            src_lbl = os.path.join(folds_path, other_fold, "labels")
            
            # Collect label stems from training set
            train_label_stems = set()
            if os.path.exists(src_lbl) and os.path.isdir(src_lbl):
                for lbl_file in os.listdir(src_lbl):
                    if os.path.isfile(os.path.join(src_lbl, lbl_file)) and lbl_file.endswith('.txt'):
                        stem = os.path.splitext(lbl_file)[0]
                        train_label_stems.add(stem)
                        src = os.path.join(src_lbl, lbl_file)
                        dst = os.path.join(train_lbl_dir, lbl_file)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
            
            # Copy images that have matching labels (including empty/background)
            if os.path.exists(src_img) and os.path.isdir(src_img):
                for img_file in os.listdir(src_img):
                    if os.path.isdir(os.path.join(src_img, img_file)):  # ← Skip directories
                        continue
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        stem = os.path.splitext(img_file)[0]
                        if stem in train_label_stems:  # Only copy if label exists (even if empty)
                            src = os.path.join(src_img, img_file)
                            dst = os.path.join(train_img_dir, img_file)
                            if not os.path.exists(dst):
                                shutil.copy2(src, dst)
        
        train_img_count = len([f for f in os.listdir(train_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        train_lbl_count = len([f for f in os.listdir(train_lbl_dir) if f.endswith('.txt')])
        print(f"✓ Train: {train_img_count} images, {train_lbl_count} labels")

        # Create YAML
        yaml_content = {
            'path': os.path.abspath(fold_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['defect']
        }

        yaml_path = os.path.join(fold_path, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        print(f"\nTraining {fold} (Normal)...")
        train_yolo(
            data_yaml=yaml_path,
            model_info=model_info_json,
            training_start=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            model_dir=os.path.join("models_normal", fold),
            weights="yolov5m.pt",
            img_size="768",
            batch_size="16",
            epochs="150"
        )

        print(f"\nTraining {fold} (Flipped)...")
        train_yolo(
            data_yaml=yaml_path,
            model_info=model_info_json,
            training_start=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            model_dir=os.path.join("models_flips", fold),
            weights="yolov5m.pt",
            img_size="768",
            batch_size="16",
            epochs="150",
            flips=True
        )

        print(f"✓ Finished {fold}\n")

    print("="*60)
    print("All folds complete!")
    print("="*60)

if __name__ == '__main__':
    run_k_fold("Castings", output_path="Folds", k=4)
    train_k_fold("Folds")
