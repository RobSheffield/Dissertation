import os
import random
from random import shuffle
import sys
import shutil
import yaml
from collections import defaultdict
import datetime
from ultralytics import YOLO

# Add parent directory to path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.format_converter import convert_gt_to_yolo
from stages.train_model import train_yolo


# --------------------------------------------------
# STEP 1: CREATE FOLDS (FOLDER-LEVEL SPLIT)
# --------------------------------------------------

def create_folds(image_path, output_path, k=4, testSize=0.2, seed=42):
    rng = random.Random(seed)

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
    
    rng.shuffle(folder_counts)
    
    if testSize > 0:
        remaining_folders = build_test_set(image_path, output_path, testSize, folder_counts)
        folder_counts = [fc for fc in folder_counts if fc[0] in remaining_folders]
    
    # Randomly assign folders to folds (not greedy)
    folds = [[] for _ in range(k)]
    fold_sizes = [0] * k
    
    for folder, count in folder_counts:
        # Randomly pick a fold instead of greedy minimum
        idx = rng.randint(0, k - 1)
        folds[idx].append(folder)
        fold_sizes[idx] += count

    print(f"Fold sizes (random assignment): {fold_sizes}")

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
                shutil.copy2(
                    os.path.join(folder_path, img),
                    os.path.join(img_dir, f"{folder}_{img}")
                )

                # ensure label exists
                lbl_path = os.path.join(lbl_dir, f"{folder}_{stem}.txt")
                if not os.path.exists(lbl_path):
                    open(lbl_path, 'w').close()

        print(f"Built fold_{i+1}")

def build_test_set(image_path, output_path, testSize, folder_counts):
    if testSize > 0:
        split_idx = int(len(folder_counts) * testSize)
        test_set = folder_counts[:split_idx]
        folder_counts = folder_counts[split_idx:]
        test_dir = os.path.join(output_path, "test")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)
        for folder, _ in test_set:
            folder_path = os.path.join(image_path, folder)
            gt_file = os.path.join(folder_path, "ground_truth.txt")

            if not os.path.isfile(gt_file):
                print(f"Skipping {folder} (no GT)")
                continue

            temp_labels = os.path.join(test_dir, "temp_labels")
            os.makedirs(temp_labels, exist_ok=True)

            convert_gt_to_yolo(gt_file, folder_path, temp_labels, class_id=0)

            for lbl in os.listdir(temp_labels):
                if lbl.endswith(".txt"):
                    shutil.move(
                        os.path.join(temp_labels, lbl),
                        os.path.join(test_dir, "labels", f"{folder}_{lbl}")
                    )

            shutil.rmtree(temp_labels)

            for img in os.listdir(folder_path):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy(
                        os.path.join(folder_path, img),
                        os.path.join(test_dir, "images", f"{folder}_{img}")
                    )
    return [f[0] for f in folder_counts]

def create_bias_folds(image_path, output_path, k=4, testSize=0.2, seed=42):
    '''k-fold where all images are shuffled together, ignoring folder structure. This is a more traditional k-fold but risks leakage if multiple images of the same defect are present.'''

    rng = random.Random(seed)

    folders = [f for f in os.listdir(image_path)
               if os.path.isdir(os.path.join(image_path, f))]

    folder_counts = []
    for folder in folders:
        folder_path = os.path.join(image_path, folder)
        count = len([f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if count > 0:
            folder_counts.append((folder, count))

    rng.shuffle(folder_counts)
    remaining_folders = [f[0] for f in folder_counts]
    if testSize > 0:
        remaining_folders = build_test_set(image_path, output_path, testSize, folder_counts)

    all_images = []
    for folder in remaining_folders:
        folder_path = os.path.join(image_path, folder)
        if os.path.isdir(folder_path):
            for img in os.listdir(folder_path):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(folder_path, img))

    rng.shuffle(all_images)
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

            # copy image
            shutil.copy2(
                os.path.join(folder_path, img),
                os.path.join(img_dir, f"{folder}_{img_name}")
            )

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

def train_all(folds_path, model_dir="models"):
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
        preweights = "yolo11n.pt"
        model_info_json = {
            "name": fold,
            "model": preweights,
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
            model_dir=os.path.join(model_dir, fold),
            weights=preweights,
            img_size="1280",
            batch_size="12",
            epochs="200",
                )


def mAP_on_test_set(test_dir, model_dir):
    images_dir = os.path.join(test_dir, "images")
    labels_dir = os.path.join(test_dir, "labels")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        raise FileNotFoundError(
            f"Expected test dataset in YOLO format at '{test_dir}' with 'images' and 'labels' folders."
        )

    yaml_path = os.path.join(test_dir, "test_data.yaml")
    yaml_data = {
        'path': os.path.abspath(test_dir),
        'train': 'images',
        'val': 'images',
        'test': 'images',
        'nc': 1,
        'names': ['defect']
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)

    model_paths = []

    if os.path.isfile(model_dir) and model_dir.lower().endswith('.pt'):
        model_paths.append(model_dir)
    elif os.path.isdir(model_dir):
        for item in sorted(os.listdir(model_dir)):
            fold_path = os.path.join(model_dir, item)
            if not os.path.isdir(fold_path):
                continue

            best_pt = os.path.join(fold_path, "weights", "best.pt")
            if os.path.isfile(best_pt):
                model_paths.append(best_pt)
    else:
        raise FileNotFoundError(f"Model path not found: {model_dir}")

    if not model_paths:
        raise FileNotFoundError(
            f"No model weights found. Expected '*.pt' or fold directories containing 'weights/best.pt' under '{model_dir}'."
        )

    def _safe_get_metric(metrics_obj, key):
        try:
            return getattr(metrics_obj.box, key)
        except Exception:
            return None

    report_lines = []
    report_lines.append(f"Test dataset: {os.path.abspath(test_dir)}")
    report_lines.append(f"Models source: {os.path.abspath(model_dir)}")
    report_lines.append("=" * 60)

    map50_values = []
    map5095_values = []
    precision_values = []
    recall_values = []

    for model_path in model_paths:
        print(f"Evaluating on test set: {model_path}")
        model = YOLO(model_path)

        metrics = model.val(
            data=yaml_path,
            split="test",
            imgsz=1280,
            batch=16,
            verbose=False
        )

        map50 = _safe_get_metric(metrics, "map50")
        map5095 = _safe_get_metric(metrics, "map")
        precision = _safe_get_metric(metrics, "mp")
        recall = _safe_get_metric(metrics, "mr")

        if map50 is not None:
            map50_values.append(float(map50))
        if map5095 is not None:
            map5095_values.append(float(map5095))
        if precision is not None:
            precision_values.append(float(precision))
        if recall is not None:
            recall_values.append(float(recall))

        report_lines.append(f"Model: {model_path}")
        report_lines.append(f"  Precision: {precision if precision is not None else 'N/A'}")
        report_lines.append(f"  Recall: {recall if recall is not None else 'N/A'}")
        report_lines.append(f"  mAP@0.50: {map50 if map50 is not None else 'N/A'}")
        report_lines.append(f"  mAP@0.50:0.95: {map5095 if map5095 is not None else 'N/A'}")
        report_lines.append("-" * 60)

    def _avg(values):
        return sum(values) / len(values) if values else None

    avg_precision = _avg(precision_values)
    avg_recall = _avg(recall_values)
    avg_map50 = _avg(map50_values)
    avg_map5095 = _avg(map5095_values)

    report_lines.append("Overall average across evaluated models")
    report_lines.append(f"  Precision: {avg_precision if avg_precision is not None else 'N/A'}")
    report_lines.append(f"  Recall: {avg_recall if avg_recall is not None else 'N/A'}")
    report_lines.append(f"  mAP@0.50: {avg_map50 if avg_map50 is not None else 'N/A'}")
    report_lines.append(f"  mAP@0.50:0.95: {avg_map5095 if avg_map5095 is not None else 'N/A'}")

    output_file = os.path.join(model_dir if os.path.isdir(model_dir) else test_dir, "test_map_results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Saved test mAP report to: {output_file}")
    return output_file

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    create_bias_folds("Castings", "Bias_folds_high_res_v11", k=3,testSize=0.2)
    build_train_val_sets("Bias_folds_high_res_v11")
    train_all("Bias_folds_high_res_v11","biased_models_high_res_v11")
    mAP_on_test_set("Bias_folds_high_res_v11/test","biased_models_high_res_v11")   

    create_folds("Castings", "Folds_high_res_v11", k=3,testSize=0.2)
    build_train_val_sets("Folds_high_res_v11")
    train_all("Folds_high_res_v11","unbiased_models_high_res_v11")
    mAP_on_test_set("Folds_high_res_v11/test","unbiased_models_high_res_v11")
