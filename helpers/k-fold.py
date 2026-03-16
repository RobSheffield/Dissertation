import sys
import os
import random
import shutil
import datetime
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stages.train_model import train_yolo

from data.format_converter import convert_gt_to_yolo

def run_k_fold(image_path, output_path, k=5):
    '''K-fold across dataset - detefects_by_folder defines whether folders seperate groups of images of the same defect. (required to avoid training images leaking into test set)'''
    defect_folders = [f for f in os.listdir(image_path) 
                      if os.path.isdir(os.path.join(image_path, f))]
    
    random.shuffle(defect_folders)

    folds = [defect_folders[i::k] for i in range(k)]

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
                temp_lbl_dir = os.path.join(fold_dir, "labels_temp")
                os.makedirs(temp_lbl_dir, exist_ok=True)
                convert_gt_to_yolo(gt_file, folder_path, temp_lbl_dir, class_id=0)

                # Track which stems have labels
                converted_stems = set()
                for lbl_file in os.listdir(temp_lbl_dir):
                    src_lbl = os.path.join(temp_lbl_dir, lbl_file)
                    dst_lbl = os.path.join(lbl_dir, f"{folder}_{lbl_file}")
                    shutil.move(src_lbl, dst_lbl)
                    converted_stems.add(os.path.splitext(lbl_file)[0])

                shutil.rmtree(temp_lbl_dir)

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
        # Add this block to clear stale caches
        print(f"Cleaning stale caches for {fold}...")
        for f in all_folds:
            cache_path = os.path.join(folds_path, f, "labels.cache")
            if os.path.exists(cache_path):
                os.remove(cache_path)

        fold_path = os.path.join(folds_path, fold)
        
        train_dirs = [
            os.path.join(os.path.abspath(folds_path), d, "images") 
            for d in all_folds if d != fold
        ]
        val_dir = os.path.join(os.path.abspath(fold_path), "images")

        yaml_content = {
            'path': os.path.abspath(folds_path),
            'train': train_dirs,
            'val': val_dir,
            'nc': 1,
            'names': ['defect']
        }

        yaml_path = os.path.join(fold_path, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        print(f"Training fold {fold} ({all_folds.index(fold)+1}/{len(all_folds)})...")
        print(f"  val:   {val_dir}")
        print(f"  train: {train_dirs}")

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
        train_yolo(
            data_yaml=yaml_path,
            model_info=model_info_json,
            training_start=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            model_dir=os.path.join("models_flip_vert", fold),
            weights="yolov5m.pt",
            img_size="768",
            batch_size="16",
            epochs="150",
            flips=True
        )

        print(f"Finished fold {fold}")

    print("All folds complete!")

if __name__ == '__main__':
    run_k_fold("Castings", output_path="Folds", k=4)
    train_k_fold("Folds")
