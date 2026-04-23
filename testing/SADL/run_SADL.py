import LSA
import importlib.util
import torch
import numpy as np
import pickle
import os
import cv2
import csv
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from map_eval import compute_mAP_for_bins
import sadl_helpers



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


def _infer_castings_folder(image_name, castings_root):
    candidate = image_name.split("_", 1)[0]
    if candidate and os.path.isdir(os.path.join(castings_root, candidate)):
        return candidate

    for folder in os.listdir(castings_root):
        folder_path = os.path.join(castings_root, folder)
        if not os.path.isdir(folder_path):
            continue
        if os.path.isfile(os.path.join(folder_path, image_name)):
            return folder
    return None


def _label_from_ground_truth_file(gt_file):
    if not os.path.isfile(gt_file):
        return 0

    with open(gt_file, "r", encoding="utf-8") as f:
        non_empty_lines = [line for line in f if line.strip()]

    # User rule: folder is defect-positive when ground-truth file has > 1 entries.
    return 1 if len(non_empty_lines) > 1 else 0


def build_folder_defect_labels(image_files, castings_root):
    labels = []
    folder_label_cache = {}

    for image_name in image_files:
        folder = _infer_castings_folder(image_name, castings_root)
        if folder is None:
            labels.append(0)
            continue

        if folder not in folder_label_cache:
            gt_file = os.path.join(castings_root, folder, "ground_truth.txt")
            folder_label_cache[folder] = _label_from_ground_truth_file(gt_file)

        labels.append(folder_label_cache[folder])

    return labels


def save_named_results(output_path, image_files, scores):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "score"])
        for image_name, score in zip(image_files, scores):
            writer.writerow([image_name, score])


def run_sadl(model_path, train_path, val_path, train_labels_path, val_labels_path):
    model = torch.load(model_path, weights_only=False)
    if isinstance(model, dict) and "model" in model:
        model = model["model"]
    train_dataset = ImageDataset(train_path)
    val_dataset = ImageDataset(val_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ats = sadl_helpers.get_ats(model, train_loader, device)
    val_ats = sadl_helpers.get_ats(model, val_loader, device)
    bin_amounts = 10

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    castings_root = os.path.join(project_root, "Castings")

    lsa_results = LSA.fetch_lsa(train_ats, val_ats)
    save_named_results("lsa_results_with_filenames.csv", val_dataset.image_files, lsa_results)

    paths_in_bin = pack_bins(val_dataset.image_files, lsa_results, bin_amounts, prefix='lsa_')
    compute_mAP_for_bins(paths_in_bin, model_path, val_path, val_labels_path, prefix='lsa_')

    return {
        "image_files": val_dataset.image_files,
        "lsa_scores": lsa_results,
    }


def score_folder_lsa(model_path, train_path, target_path):
    model = torch.load(model_path, weights_only=False)
    if isinstance(model, dict) and "model" in model:
        model = model["model"]

    train_dataset = ImageDataset(train_path)
    target_dataset = ImageDataset(target_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    target_loader = DataLoader(target_dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ats = sadl_helpers.get_ats(model, train_loader, device)
    target_ats = sadl_helpers.get_ats(model, target_loader, device)

    lsa_scores = LSA.fetch_lsa(train_ats, target_ats)
    target_image_paths = [os.path.join(target_path, name) for name in target_dataset.image_files]
    folder_scores = score_folder(target_image_paths, lsa_scores, aggregation="median")
    return folder_scores, dict(zip(target_dataset.image_files, lsa_scores))


def score_folder_dsa(model_path, train_path, target_path):
    # Backward compatibility alias: this now uses LSA scoring.
    return score_folder_lsa(model_path, train_path, target_path)


def score_folder(image_paths, values, aggregation="median"):
    # Simple average of image scores as folder score.
    # Also infer source folder under Castings for each image and print per-folder mean.
    if len(image_paths) != len(values):
        raise ValueError("image_paths and values must have the same length.")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    castings_root = os.path.join(project_root, "Castings")

    def infer_castings_folder(image_path):
        # Fast path for names produced by the splitter, e.g. C0001_C0001_0001.png.
        image_name = os.path.basename(image_path)
        candidate = image_name.split("_", 1)[0]
        if candidate and os.path.isdir(os.path.join(castings_root, candidate)):
            return candidate

        # Fallback: search actual Castings subfolders for this image file.
        for folder in os.listdir(castings_root):
            folder_path = os.path.join(castings_root, folder)
            if not os.path.isdir(folder_path):
                continue
            if os.path.isfile(os.path.join(folder_path, image_name)):
                return folder
        return None

    folder_scores = []
    for image_path, value in zip(image_paths, values):
        folder = infer_castings_folder(image_path)
        folder_key = folder if folder is not None else "UNKNOWN"
        folder_scores.append((folder_key, float(value)))
        print(f"{folder_key}/{os.path.basename(image_path)}: {value}")

    all_folders = set(k for k, _ in folder_scores)
    avg_folder_scores = {}
    for folder in all_folders:
        folder_values = [v for k, v in folder_scores if k == folder]
        if aggregation == "mean":
            score = float(np.mean(folder_values))
        else:
            # Median is less sensitive to outliers in small data regimes.
            score = float(np.median(folder_values))
        avg_folder_scores[folder] = score

    return avg_folder_scores



if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "strat_10_unbiased_models", "fold_1", "weights", "best.pt")
    )
    train_path = os.path.join(project_root,  "Folds_strat_10","fold_1","images", "train")
    val_path = os.path.join(project_root,  "Folds_strat_10","fold_1","images", "val")
    train_labels_path = os.path.join(project_root,  "Folds_strat_10","fold_1","labels", "train")
    val_labels_path = os.path.join(project_root,  "Folds_strat_10","fold_1","labels", "val")
    run_sadl(model_path, train_path, val_path, train_labels_path, val_labels_path)