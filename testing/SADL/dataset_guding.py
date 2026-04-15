import os
import random
import shutil
import heapq
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data.format_converter import convert_gt_to_yolo
import run_SADL
from stages import train_model
from stages.model_training import create_yaml
from helpers import k_fold
from map_eval import evaluate_map50_on_image_subset
from ultralytics import YOLO
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def balanced_split(folders, counts, k, seed=42):
    """
    Near-optimal k-way balancing using LPT + heap.
    """
    rng = random.Random(seed)
    items = list(zip(folders, counts))
    rng.shuffle(items)
    items.sort(key=lambda x: -x[1])

    heap = [(0, i) for i in range(k)]
    heapq.heapify(heap)

    folds = [[] for _ in range(k)]
    fold_sizes = [0] * k

    for folder, count in items:
        _, idx = heapq.heappop(heap)
        folds[idx].append(folder)
        fold_sizes[idx] += count
        heapq.heappush(heap, (fold_sizes[idx], idx))

    return folds, fold_sizes


def _list_eligible_folders(castings_root):
    """
    Return (folder_name, image_count) only for folders that include ground_truth.txt.
    """
    folder_counts = []
    for name in sorted(os.listdir(castings_root)):
        folder_path = os.path.join(castings_root, name)
        if not os.path.isdir(folder_path):
            continue

        gt_file = os.path.join(folder_path, "ground_truth.txt")
        if not os.path.isfile(gt_file):
            continue

        image_count = len([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])
        if image_count > 0:
            folder_counts.append((name, image_count))

    return folder_counts


def _reset_split_dirs(output_root):
    if os.path.exists(output_root):
        shutil.rmtree(output_root)

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(output_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_root, "labels", split), exist_ok=True)


def _normalize_portions(portion_1, portion_2, portion_3):
    portions = [float(portion_1), float(portion_2), float(portion_3)]
    if any(p < 0 for p in portions):
        raise ValueError("Portions must be non-negative.")

    total = sum(portions)
    if total <= 0:
        raise ValueError("At least one portion must be greater than 0.")

    return [p / total for p in portions]


def _assign_folders_by_portion(folder_counts, portions, seed=42):
    """
    Assign whole folders to train/val/test using a greedy target-deficit strategy
    based on image counts.
    """
    rng = random.Random(seed)
    items = list(folder_counts)
    rng.shuffle(items)
    items.sort(key=lambda x: -x[1])

    split_names = ["train", "val", "test"]
    total_images = sum(count for _, count in items)
    targets = {
        name: total_images * portion for name, portion in zip(split_names, portions)
    }
    assignments = {name: [] for name in split_names}
    assigned_counts = {name: 0 for name in split_names}

    # Seed each non-zero target split with one folder when possible.
    nonzero_splits = [name for name in split_names if targets[name] > 0]
    for i, split in enumerate(nonzero_splits):
        if i >= len(items):
            break
        folder, count = items[i]
        assignments[split].append(folder)
        assigned_counts[split] += count

    remaining_items = items[len(nonzero_splits):]

    for folder, count in remaining_items:
        # Prefer the split furthest below its target.
        deficits = {
            split: targets[split] - assigned_counts[split]
            for split in split_names
        }
        best_split = max(split_names, key=lambda s: deficits[s])
        assignments[best_split].append(folder)
        assigned_counts[best_split] += count

    return assignments, assigned_counts, targets


def _copy_folder_data(folder_name, castings_root, split, output_root):
    folder_path = os.path.join(castings_root, folder_name)
    gt_file = os.path.join(folder_path, "ground_truth.txt")

    temp_labels_dir = os.path.join(output_root, "_temp_labels", folder_name)
    os.makedirs(temp_labels_dir, exist_ok=True)
    convert_gt_to_yolo(gt_file, folder_path, temp_labels_dir, class_id=0)

    split_img_dir = os.path.join(output_root, "images", split)
    split_lbl_dir = os.path.join(output_root, "labels", split)

    for image_name in os.listdir(folder_path):
        if not image_name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        stem, _ = os.path.splitext(image_name)
        out_image_name = f"{folder_name}_{image_name}"
        out_label_name = f"{folder_name}_{stem}.txt"

        shutil.copy2(
            os.path.join(folder_path, image_name),
            os.path.join(split_img_dir, out_image_name),
        )

        temp_label_file = os.path.join(temp_labels_dir, f"{stem}.txt")
        out_label_file = os.path.join(split_lbl_dir, out_label_name)
        if os.path.isfile(temp_label_file):
            shutil.copy2(temp_label_file, out_label_file)
        else:
            open(out_label_file, "w", encoding="utf-8").close()


def build_partition(castings_dir="Castings", output_dir="testing/First_60", seed=42, portion_1=0.5, portion_2=0.3, portion_3=0.2):
    castings_root = os.path.join(PROJECT_ROOT, castings_dir)
    output_root = os.path.join(PROJECT_ROOT, output_dir)

    folder_counts = _list_eligible_folders(castings_root)
    if not folder_counts:
        raise ValueError("No eligible folders found with ground_truth.txt and images.")

    portions = _normalize_portions(portion_1, portion_2, portion_3)
    assignments, assigned_counts, targets = _assign_folders_by_portion(folder_counts, portions, seed=seed)

    train_folders = assignments["train"]
    val_folders = assignments["val"]
    test_folders = assignments["test"]

    _reset_split_dirs(output_root)

    for folder in train_folders:
        _copy_folder_data(folder, castings_root, "train", output_root)
    for folder in val_folders:
        _copy_folder_data(folder, castings_root, "val", output_root)
    for folder in test_folders:
        _copy_folder_data(folder, castings_root, "test", output_root)

    shutil.rmtree(os.path.join(output_root, "_temp_labels"), ignore_errors=True)

    train_count = len(os.listdir(os.path.join(output_root, "images", "train")))
    val_count = len(os.listdir(os.path.join(output_root, "images", "val")))
    test_count = len(os.listdir(os.path.join(output_root, "images", "test")))

    return {
        "train_folders": train_folders,
        "val_folders": val_folders,
        "test_folders": test_folders,
        "train_images": train_count,
        "val_images": val_count,
        "test_images": test_count,
        "requested_portions": {
            "train": portions[0],
            "val": portions[1],
            "test": portions[2],
        },
        "assigned_images": assigned_counts,
        "output_root": output_root,
    }

def move_folders_to_train(folders,output_root):
    for folder in folders:
        for split in ("val", "test"):
            split_img_dir = os.path.join(output_root, "images", split)
            split_lbl_dir = os.path.join(output_root, "labels", split)

            for image_name in os.listdir(split_img_dir):
                if image_name.startswith(f"{folder}_"):
                    shutil.move(
                        os.path.join(split_img_dir, image_name),
                        os.path.join(output_root, "images", "train", image_name),
                    )

            for label_name in os.listdir(split_lbl_dir):
                if label_name.startswith(f"{folder}_"):
                    shutil.move(
                        os.path.join(split_lbl_dir, label_name),
                        os.path.join(output_root, "labels", "train", label_name),
                    )


def _train_minimal(data_dir, model_dir, epochs=50):
    data_yaml = os.path.join(data_dir, "data.yaml")
    create_yaml(data_dir, data_yaml, ["defect"])

    train_img_dir = os.path.join(data_dir, "images", "train")
    train_img_count = len([f for f in os.listdir(train_img_dir) if f.lower().endswith(IMAGE_EXTENSIONS)])
    model_info = json.dumps(
        {
            "name": os.path.basename(model_dir),
            "model": "yolo11n.pt",
            "date_time_trained": datetime.now().isoformat(),
            "total_training_time": 0,
            "number_of_images": train_img_count,
        }
    )

    return train_model.train_yolo(
        data_yaml=data_yaml,
        model_info=model_info,
        training_start=datetime.now().isoformat(),
        model_dir=model_dir,
        img_size="1280",
        epochs=str(epochs),
        device="auto",
    )

if __name__ == "__main__":
    portion_1 = 0.5
    portion_2 = 0.3
    portion_3 = 0.2
    build_partition(castings_dir="Castings", output_dir="testing/First_60_guide", seed=42,portion_1=portion_1, portion_2=portion_2, portion_3=portion_3)
    build_partition(castings_dir="Castings", output_dir="testing/First_60_rand", seed=42,portion_1=portion_1, portion_2=portion_2, portion_3=portion_3)

    baseline = _train_minimal("testing/First_60_guide", "testing/First_60_baseline_model", epochs=50)
    run_SADL.run_sadl(
        model_path="testing/First_60_baseline_model/weights/best.pt",
        train_path="testing/First_60_guide/images/train",
        val_path="testing/First_60_guide/images/val",
        train_labels_path="testing/First_60_guide/labels/train",
        val_labels_path="testing/First_60_guide/labels/val",
    )

    test_image_names = [
        f for f in os.listdir("testing/First_60_guide/images/test")
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    evaluate_map50_on_image_subset(
        yolo_model=YOLO("testing/First_60_baseline_model/weights/best.pt"),
        image_names=test_image_names,
        images_dir="testing/First_60_guide/images/test",
        labels_dir="testing/First_60_guide/labels/test",
        bin_root="binned_results/_temp_eval_guide_test",
    )
    folders = run_SADL.score_folder(
        image_paths=[os.path.join("testing/First_60_guide/images/test", f) for f in test_image_names],
        values=[0.5] * len(test_image_names)
    )
    folders = sorted(folders.items(), key=lambda x: x[1], reverse=True)
    selected_folders = [folder for folder, score in folders[:int(len(folders) * 0.5)]]
    random_folders = [folder for folder, score in folders[:int(len(folders) * 0.5)]]
    random.Random(42).shuffle(random_folders)
    move_folders_to_train(selected_folders, "testing/First_60_guide")
    move_folders_to_train(random_folders, "testing/First_60_rand")
    guided_model = _train_minimal("testing/First_60_guide", "testing/First_60_guide_model", epochs=50)

    random_model = _train_minimal("testing/First_60_rand", "testing/First_60_rand_model", epochs=50)
    
        
