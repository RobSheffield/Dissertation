import os
from random import random, shuffle
import sys
import shutil
import yaml
from collections import defaultdict
import datetime

# Add parent directory to path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.format_converter import convert_gt_to_yolo
from stages.train_model import train_yolo


# --------------------------------------------------
# STEP 1: CREATE FOLDS (FOLDER-LEVEL SPLIT)
# --------------------------------------------------

def create_folds(image_path, output_path, k=4):
    folders = [f for f in os.listdir(image_path)
               if os.path.isdir(os.path.join(image_path, f))]

    # Count images per folder
    folder_counts = []
    for folder in folders:
        path = os.path.join(image_path, folder)
        count = len([f for f in os.listdir(path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if count > 0:
            folder_counts.append((folder, count))

    # Sort largest → smallest
    folder_counts.sort(key=lambda x: x[1], reverse=True)

    # Greedy balance
    folds = [[] for _ in range(k)]
    fold_sizes = [0] * k

    for folder, count in folder_counts:
        idx = fold_sizes.index(min(fold_sizes))
        folds[idx].append(folder)
        fold_sizes[idx] += count

    print(f"Fold sizes: {fold_sizes}")

    # Build fold directories
    for i, fold in enumerate(folds):
        fold_dir = os.path.join(output_path, f"fold_{i+1}")
        img_dir = os.path.join(fold_dir, "images")
        lbl_dir = os.path.join(fold_dir, "labels")

        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)

        os.makedirs(img_dir)
        os.makedirs(lbl_dir)

        for folder in fold:
            folder_path = os.path.join(image_path, folder)
            gt_file = os.path.join(folder_path, "ground_truth.txt")

            if not os.path.isfile(gt_file):
                print(f"Skipping {folder} (no GT)")
                continue

            temp_labels = os.path.join(fold_dir, "temp_labels")
            os.makedirs(temp_labels, exist_ok=True)

            convert_gt_to_yolo(gt_file, folder_path, temp_labels, class_id=0)

            # Track labeled stems
            labeled = set()

            for lbl in os.listdir(temp_labels):
                if lbl.endswith(".txt"):
                    stem = os.path.splitext(lbl)[0]
                    labeled.add(stem)

                    shutil.move(
                        os.path.join(temp_labels, lbl),
                        os.path.join(lbl_dir, f"{folder}_{lbl}")
                    )

            shutil.rmtree(temp_labels)

            # Copy ALL images + ensure empty labels exist
            for img in os.listdir(folder_path):
                if not img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                stem = os.path.splitext(img)[0]

                # copy image
                shutil.copy(
                    os.path.join(folder_path, img),
                    os.path.join(img_dir, f"{folder}_{img}")
                )

                # ensure label exists
                lbl_path = os.path.join(lbl_dir, f"{folder}_{stem}.txt")
                if not os.path.exists(lbl_path):
                    open(lbl_path, 'w').close()

        print(f"Built fold_{i+1}")

def create_bias_folds(image_path, output_path, k=4):
    '''k-fold where all images are shuffled together, ignoring folder structure. This is a more traditional k-fold but risks leakage if multiple images of the same defect are present.'''
    all_images = []
    for root, dirs, files in os.walk(image_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))
    shuffle(all_images)
    folds = [all_images[i::k] for i in range(k)]
    for i, fold in enumerate(folds):
        fold_dir = os.path.join(output_path, f"fold_{i+1}")
        img_dir = os.path.join(fold_dir, "images")
        lbl_dir = os.path.join(fold_dir, "labels")

        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)

        os.makedirs(img_dir)
        os.makedirs(lbl_dir)

        for img_path in fold:
            folder = os.path.basename(os.path.dirname(img_path))
            img_name = os.path.basename(img_path)
            stem = os.path.splitext(img_name)[0]

            gt_file = os.path.join(os.path.dirname(img_path), "ground_truth.txt")
            if not os.path.isfile(gt_file):
                print(f"Skipping {img_path} (no GT)")
                continue

            temp_labels = os.path.join(fold_dir, "temp_labels")
            os.makedirs(temp_labels, exist_ok=True)

            convert_gt_to_yolo(gt_file, os.path.dirname(img_path), temp_labels, class_id=0)

            lbl_src = os.path.join(temp_labels, f"{stem}.txt")
            lbl_dst = os.path.join(lbl_dir, f"{folder}_{stem}.txt")

            if os.path.exists(lbl_src):
                shutil.move(lbl_src, lbl_dst)
            else:
                open(lbl_dst, 'w').close()

            shutil.copy(img_path, os.path.join(img_dir, f"{folder}_{img_name}"))

        print(f"Built fold_{i+1}")

# --------------------------------------------------
# STEP 2: BUILD TRAIN/VAL FOR EACH FOLD
# --------------------------------------------------

def build_train_val_sets(folds_path):
    folds = sorted([f for f in os.listdir(folds_path) if f.startswith("fold_")])

    for fold in folds:
        print(f"\nProcessing {fold}")

        fold_dir = os.path.join(folds_path, fold)

        train_img = os.path.join(fold_dir, "images/train")
        val_img = os.path.join(fold_dir, "images/val")
        train_lbl = os.path.join(fold_dir, "labels/train")
        val_lbl = os.path.join(fold_dir, "labels/val")

        for d in [train_img, val_img, train_lbl, val_lbl]:
            os.makedirs(d, exist_ok=True)

        # -----------------------
        # VALIDATION = this fold
        # -----------------------
        src_img = os.path.join(fold_dir, "images")
        src_lbl = os.path.join(fold_dir, "labels")

        for f in os.listdir(src_img):
            if os.path.isfile(os.path.join(src_img, f)):
                shutil.copy2(os.path.join(src_img, f), os.path.join(val_img, f))

        for f in os.listdir(src_lbl):
            if os.path.isfile(os.path.join(src_lbl, f)):
                shutil.copy2(os.path.join(src_lbl, f), os.path.join(val_lbl, f))

        # -----------------------
        # TRAIN = all OTHER folds
        # -----------------------
        for other in folds:
            if other == fold:
                continue

            other_dir = os.path.join(folds_path, other)

            o_img = os.path.join(other_dir, "images")
            o_lbl = os.path.join(other_dir, "labels")

            for f in os.listdir(o_img):
                if os.path.isfile(os.path.join(o_img, f)):
                    shutil.copy2(os.path.join(o_img, f), os.path.join(train_img, f))

            for f in os.listdir(o_lbl):
                if os.path.isfile(os.path.join(o_lbl, f)):
                    shutil.copy2(os.path.join(o_lbl, f), os.path.join(train_lbl, f))

        print(f"✓ {fold} ready")


# --------------------------------------------------
# STEP 3: TRAIN
# --------------------------------------------------

def train_all(folds_path,model_dir="models"):
    folds = sorted([f for f in os.listdir(folds_path) if f.startswith("fold_")])

    for fold in folds:
        fold_dir = os.path.join(folds_path, fold)
        yaml_path = os.path.join(fold_dir, "data.yaml")

        yaml_data = {
            'path': os.path.abspath(fold_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['defect']
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)

        print(f"\nTraining {fold}")

        # Count training images
        train_img_count = len([f for f in os.listdir(os.path.join(fold_dir, "images/train")) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Create proper model_info JSON
        model_info_json = {
            "name": fold,
            "model": "yolo11n",
            "date_time_trained": datetime.datetime.now().isoformat(),
            "total_training_time": 0,
            "number_of_images": train_img_count
        }

        import json
        model_info_str = json.dumps(model_info_json)

        train_yolo(
            data_yaml=yaml_path,
            model_info=model_info_str,
            training_start=datetime.datetime.now().isoformat(),
            model_dir=os.path.join("models_folder_split", fold),
            weights="yolov5mu.pt",   
            img_size="640",
            batch_size="16",
            epochs="250",
            patience = "0"
        )

              

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    create_folds("Castings", "Folds", k=4)
    build_train_val_sets("Folds")
    train_all("Folds","unbiased_models")

    create_bias_folds("Castings", "Bias_folds", k=4)
    build_train_val_sets("Bias_folds")
    train_all("Bias_folds","biased_models")