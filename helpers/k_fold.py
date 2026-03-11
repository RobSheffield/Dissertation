import sys
import os
import random
import shutil
import datetime
import yaml
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stages.train_model import train_yolo
from data.format_converter import convert_gt_to_yolo

def make_k_folds(image_path, output_path, k=5):
    '''K-fold across dataset - detefects_by_folder defines whether folders seperate groups of images of the same defect. (required to avoid training images leaking into test set)'''
    # Get ALL directories first
    all_folders = [f for f in os.listdir(image_path) 
                   if os.path.isdir(os.path.join(image_path, f))]

    castings = sorted(all_folders)
    castings_with_gt = [f for f in castings
                        if os.path.isfile(os.path.join(image_path, f, "ground_truth.txt"))]

    print(f"Found {len(castings)} Castings total ({len(castings_with_gt)} with ground_truth.txt)")

    if not castings:
        raise ValueError("No Castings folders found!")

    random.shuffle(castings)
    folds = [castings[i::k] for i in range(k)]

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

        print(f"Building {fold_name} ({len(fold_folders)} casting folders)...")

        for folder in fold_folders:
            folder_path = os.path.join(image_path, folder)
            direct_gt = os.path.join(folder_path, "ground_truth.txt")
            converted_stems = set()

            if os.path.isfile(direct_gt):
                # Convert labels first
                temp_lbl_dir = os.path.join(fold_dir, "labels_temp")
                os.makedirs(temp_lbl_dir, exist_ok=True)
                convert_gt_to_yolo(direct_gt, folder_path, temp_lbl_dir, class_id=0)

                # Track which stems have labels
                for lbl_file in os.listdir(temp_lbl_dir):
                    src_lbl = os.path.join(temp_lbl_dir, lbl_file)
                    dst_lbl = os.path.join(lbl_dir, f"{folder}_{lbl_file}")
                    shutil.move(src_lbl, dst_lbl)
                    converted_stems.add(os.path.splitext(lbl_file)[0])

                shutil.rmtree(temp_lbl_dir)
            else:
                # No ground truth means all images are negative examples (empty label files).
                print(f"WARNING: No gt file found for {folder}, treating images as negatives.")
                for file in os.listdir(folder_path):
                    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    stem = os.path.splitext(file)[0]
                    empty_label_path = os.path.join(lbl_dir, f"{folder}_{stem}.txt")
                    open(empty_label_path, 'w').close()
                    converted_stems.add(stem)

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

        print(f"  -> {len(os.listdir(img_dir))} images, {len(os.listdir(lbl_dir))} labels in {fold_name}")
    fold_info = folds
    return fold_info


def run_k_fold_temp(image_path, output_path, k=5):
    '''Creates fold directories, then for each fold iteration creates a single merged train 
    directory, trains, then deletes it before moving to the next fold - to save file quota'''

    os.makedirs(output_path, exist_ok=True)
    for d in os.listdir(output_path):
        d_path = os.path.join(output_path, d)
        if d.startswith("fold_") and os.path.isdir(d_path):
            shutil.rmtree(d_path)
            print(f"Cleared stale fold directory: {d}")
    
    # First build the base folds
    fold_info = make_k_folds(image_path, output_path, k)

    all_folds = [f"fold_{i + 1}" for i in range(len(fold_info))]

    for fold in all_folds[:1]:  # TEMP - just do first fold for testing
        fold_path = os.path.join(output_path, fold)

        # Create merged train dir just for this fold
        temp_dir = os.path.join(fold_path, "train_merged")
        temp_img_dir = os.path.join(temp_dir, "images")
        temp_lbl_dir = os.path.join(temp_dir, "labels")
        os.makedirs(temp_img_dir, exist_ok=True)
        os.makedirs(temp_lbl_dir, exist_ok=True)

        # Merge k-1 folds into temp dir
        for other_fold in all_folds:
            if other_fold == fold:
                continue

            src_img_dir = os.path.join(output_path, other_fold, "images")
            src_lbl_dir = os.path.join(output_path, other_fold, "labels")

            for file in os.listdir(src_img_dir):
                shutil.copy(os.path.join(src_img_dir, file), os.path.join(temp_img_dir, file))
            for file in os.listdir(src_lbl_dir):
                shutil.copy(os.path.join(src_lbl_dir, file), os.path.join(temp_lbl_dir, file))

        num_training_images = len(os.listdir(temp_img_dir))
        print(f"{fold} merged train: {num_training_images} images, {len(os.listdir(temp_lbl_dir))} labels")


        model_info_dict = {
            "name": "k_fold",
            "model": "YOLOv5",
            "number_of_images": str(num_training_images),
            "date_time_trained": "",
            "total_training_time": "",
            "path": "",
            "epoch": "",
            "box_loss": "",
            "cls_loss": "",
            "mAP_50": "",
            "mAP_50_95": "",
            "precision": "",
            "recall": "",
            "dataset_config": "K-Fold",
            "starting_model": "",
            "folder_name": "",
            "metamorphic_test_result": "",
            "differential_test_result": "",
            "fuzzing_test_result": "",
        }
        model_info_json = json.dumps(model_info_dict)

        abs_train = os.path.abspath(temp_img_dir)
        abs_val   = os.path.abspath(os.path.join(fold_path, "images"))
        yaml_content = {
            'path': '.',        
            'train': abs_train,
            'val': abs_val,
            'nc': 1,
            'names': ['defect']
        }
        print(f"  YAML train path: {abs_train} (exists: {os.path.exists(abs_train)})")
        print(f"  YAML val   path: {abs_val}   (exists: {os.path.exists(abs_val)})")
        yaml_path = os.path.join(fold_path, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        print(f"Training fold {fold} ({all_folds.index(fold)+1}/{len(all_folds)})...")
        model_dir = os.path.join(fold_path, "models")
        train_yolo(
            data_yaml=yaml_path,
            model_info=model_info_json,
            training_start=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            model_dir=model_dir + "_no_flips_vertical_test",
            weights="yolov5m.pt",
            img_size="768",
            batch_size="16",
            epochs="120",
            flips = False
        )
        train_yolo(
            data_yaml=yaml_path,
            model_info=model_info_json,
            training_start=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            model_dir=model_dir+"_vertical_flips",
            weights="yolov5m.pt",
            img_size="768",
            batch_size="16",
            epochs="120",
            flips = True
        )

        # Delete merged dir immediately after training to save file quota
        shutil.rmtree(temp_dir)

        #store what castings were used for this fold (important)
        with open(os.path.join(model_dir, "castings_used.txt"), 'w') as f:
            f.write(f"Fold info: {fold_info}\n")

if __name__ == '__main__':
    run_k_fold_temp("Castings", output_path="fold_paths", k=10)
