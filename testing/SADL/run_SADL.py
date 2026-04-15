import DSA
import LSA
import helpers
import torch
import numpy as np
import pickle
import os
import cv2
import csv
import shutil
import yaml
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO


class ImageDataset(Dataset):
    def __init__(self, image_dir, image_size=640):
        self.image_dir = image_dir
        self.image_size = image_size
        self.image_files = sorted(
            [
                f for f in os.listdir(image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image


def pack_bins(image_files, results, bin_amounts, prefix=''):
    paired_results = sorted(zip(image_files, results), key=lambda item: item[1])
    bins = np.array_split(paired_results, bin_amounts)
    if not os.path.exists('binned_results'):
        os.makedirs('binned_results')
    # Use image filenames to create a file list of each image in bins.
    bin_paths = []
    for i, bin in enumerate(bins):
        binned_items = []
        with open(f"binned_results/{prefix}bin_{i}.txt", "w") as f:
            for result in bin:
                image_name, score = result
                binned_items.append(image_name)
                f.write(f"{image_name},{score}\n")
        bin_paths.append(binned_items)
    return bin_paths


def get_labels(label_path, image_files):
    labels = []
    for image_name in image_files:
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_file = os.path.join(label_path, label_name)
        if not os.path.exists(label_file):
            labels.append(0)
            continue

        with open(label_file, "r") as f:
            labels.append(1 if f.read().strip() else 0)
    return labels


def save_named_results(output_path, image_files, scores):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "score"])
        for image_name, score in zip(image_files, scores):
            writer.writerow([image_name, score])

def compute_mAP_for_bins(paths_in_bin, model_path, images_dir, labels_dir, prefix='', imgsz=1280, batch=16):
    yolo_model = YOLO(model_path)

    output_root = "binned_results"
    temp_root = os.path.join(output_root, f"_temp_eval_{prefix.strip('_') or 'bin'}")
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    os.makedirs(temp_root, exist_ok=True)

    summary_rows = []

    for i, bin_paths in enumerate(paths_in_bin):
        print(f"Computing mAP50 for bin {i} with {len(bin_paths)} images...")

        bin_root = os.path.join(temp_root, f"bin_{i}")
        bin_images = os.path.join(bin_root, "images")
        bin_labels = os.path.join(bin_root, "labels")
        os.makedirs(bin_images, exist_ok=True)
        os.makedirs(bin_labels, exist_ok=True)

        copied_images = 0
        for image_name in bin_paths:
            src_img = os.path.join(images_dir, image_name)
            if not os.path.isfile(src_img):
                continue

            dst_img = os.path.join(bin_images, image_name)
            shutil.copy2(src_img, dst_img)
            copied_images += 1

            label_name = os.path.splitext(image_name)[0] + ".txt"
            src_lbl = os.path.join(labels_dir, label_name)
            dst_lbl = os.path.join(bin_labels, label_name)
            if os.path.isfile(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                open(dst_lbl, "w").close()

        if copied_images == 0:
            summary_rows.append((i, 0, float("nan")))
            print(f"Bin {i} has no valid images. mAP50 = NaN")
            continue

        data_yaml_path = os.path.join(bin_root, "data.yaml")
        data_yaml = {
            "path": os.path.abspath(bin_root),
            "train": "images",
            "val": "images",
            "test": "images",
            "nc": 1,
            "names": ["defect"],
        }
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f)

        metrics = yolo_model.val(
            data=data_yaml_path,
            split="val",
            imgsz=imgsz,
            batch=min(batch, copied_images),
            verbose=False,
        )
        map50 = float(getattr(metrics.box, "map50", float("nan")))
        summary_rows.append((i, copied_images, map50))
        print(f"Bin {i} mAP50: {map50}")

    summary_path = os.path.join(output_root, f"{prefix}bin_map50.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin", "images", "map50"])
        writer.writerows(summary_rows)

    print(f"Saved per-bin mAP50 summary: {summary_path}")
    return summary_rows

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "testing","final_datas","strat_10_unbiased_models", "fold_1", "weights", "best.pt")
    )
    train_path = os.path.join(project_root,  "testing","final_datas","Folds_strat_10","fold_1","images", "train")
    val_path = os.path.join(project_root,  "testing","final_datas","Folds_strat_10","fold_1","images", "val")
    train_labels_path = os.path.join(project_root,  "testing","final_datas","Folds_strat_10","fold_1","labels", "train")
    val_labels_path = os.path.join(project_root,  "testing","final_datas","Folds_strat_10","fold_1","labels", "val")
    #TODO fix this up
    model = torch.load(model_path, weights_only=False)
    if isinstance(model, dict) and "model" in model:
        model = model["model"]
    train_dataset = ImageDataset(train_path)
    val_dataset = ImageDataset(val_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ats = helpers.get_ats(model, train_loader, device)
    val_ats = helpers.get_ats(model, val_loader, device)
    bin_amounts = 10

    lsa_results = LSA.fetch_lsa(train_ats, val_ats)

    print("LSA results:", lsa_results)
    save_named_results("lsa_results_with_filenames.csv", val_dataset.image_files, lsa_results)

    # DSA requires labels for both train and target activation traces.
    # This script currently has image-only datasets, so use a single fallback class
    # and skip DSA when class diversity is insufficient.
    train_labels = get_labels(train_labels_path, train_dataset.image_files)
    target_labels = get_labels(val_labels_path, val_dataset.image_files)
    if np.unique(train_labels).size >= 2:
        dsa_results = DSA.fetch_dsa(train_ats, train_labels, val_ats, target_labels)
        save_named_results("dsa_results_with_filenames.csv", val_dataset.image_files, dsa_results)
    else:
        dsa_results = []
        print("Skipping DSA: requires at least two classes in train labels.")

    paths_in_bin = pack_bins(val_dataset.image_files, lsa_results, bin_amounts, prefix='lsa_')
    compute_mAP_for_bins(paths_in_bin, model_path, val_path, val_labels_path, prefix='lsa_')

    paths_in_bin = pack_bins(val_dataset.image_files, dsa_results, bin_amounts, prefix='dsa_')
    compute_mAP_for_bins(paths_in_bin, model_path, val_path, val_labels_path, prefix='dsa_')
