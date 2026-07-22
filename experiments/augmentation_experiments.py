import os
import sys
import shutil
import json
from pathlib import Path
import tempfile
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stages import model_training
from stages.train_model import train_yolo
from helpers.k_fold import augment_training_set_with_selected_transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _copy_dataset(src_root, dst_root):
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)


def _make_yaml(data_root, yaml_path):
    class_names = ["0"]
    model_training.create_yaml(data_root, yaml_path, class_names)


def run_experiments(
    source_dataset_root,
    output_root="experiments/augmentation_results",
    weights="yolo11n.pt",
    epochs=10,
    device="auto",
):
    os.makedirs(output_root, exist_ok=True)

    # Define augmentation configurations to test
    configs = [
        ("baseline", {"fliplr": 0.0, "flipud": 0.0, "rots": []}),
        ("hflip", {"fliplr": 0.5, "flipud": 0.0, "rots": []}),
        ("vflip", {"fliplr": 0.0, "flipud": 0.5, "rots": []}),
        ("both_flips", {"fliplr": 0.5, "flipud": 0.5, "rots": []}),
        ("rot90", {"fliplr": 0.0, "flipud": 0.0, "rots": ["rot90"]}),
        ("rot180", {"fliplr": 0.0, "flipud": 0.0, "rots": ["rot180"]}),
        ("rot270", {"fliplr": 0.0, "flipud": 0.0, "rots": ["rot270"]}),
        ("all_rots", {"fliplr": 0.0, "flipud": 0.0, "rots": ["rot90", "rot180", "rot270"]}),
    ]

    rows = []

    for name, opts in configs:
        print(f"Running experiment: {name}")
        work_dir = os.path.join(output_root, name)
        # copy dataset
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        shutil.copytree(source_dataset_root, work_dir)

        # apply rotations if requested (only affects training set)
        if opts["rots"]:
            train_img_dir = os.path.join(work_dir, "images", "train")
            train_lbl_dir = os.path.join(work_dir, "labels", "train")
            augment_training_set_with_selected_transforms(
                train_img_dir,
                train_lbl_dir,
                include_horizontal_flip=False,
                include_vertical_flip=False,
                include_rot90=("rot90" in opts["rots"]),
                include_rot180=("rot180" in opts["rots"]),
                include_rot270=("rot270" in opts["rots"]),
            )

        # create yaml
        yaml_path = os.path.join(work_dir, "data.yaml")
        _make_yaml(work_dir, yaml_path)

        # prepare model info json string
        model_info = {
            "name": f"augment_{name}",
            "model": weights,
            "date_time_trained": "",
            "total_training_time": "",
            "number_of_images": sum(
                1 for f in os.listdir(os.path.join(work_dir, "images", "train")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ),
            "epoch": str(epochs),
        }

        model_dir = os.path.join(os.path.abspath(output_root), f"model_{name}")

        # call train_yolo directly to avoid subprocess complexity
        try:
            trained_dir = train_yolo(
                data_yaml=yaml_path,
                model_info=json.dumps(model_info),
                training_start="now",
                model_dir=model_dir,
                weights=weights,
                img_size="640",
                batch_size="8",
                epochs=str(epochs),
                device=device,
                fliplr=str(opts["fliplr"]),
                flipud=str(opts["flipud"]),
            )
        except Exception as e:
            print(f"Training failed for {name}: {e}")
            rows.append({"config": name, "map50": "error"})
            continue

        # Read info.json saved by train_yolo
        info_path = os.path.join(trained_dir, "info.json")
        map50 = None
        if os.path.isfile(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                map50 = info.get("mAP_50")

        rows.append({"config": name, "map50": map50})

    # save CSV
    csv_path = os.path.join(output_root, "augmentation_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["config", "map50"])
        writer.writeheader()
        writer.writerows(rows)

    # plot
    configs = [r["config"] for r in rows if r["map50"] not in (None, "error")]
    values = [float(r["map50"]) for r in rows if r["map50"] not in (None, "error")]
    if configs and values:
        plt.figure(figsize=(10, 5))
        plt.bar(configs, values)
        plt.ylabel("mAP@0.50")
        plt.title("Augmentation experiment results")
        plt.savefig(os.path.join(output_root, "augmentation_results.png"))
        print(f"Saved plot to: {os.path.join(output_root, 'augmentation_results.png')}")

    print(f"Saved CSV to: {csv_path}")
    return csv_path


if __name__ == "__main__":
    # Default source dataset root expected to be in YOLO format:
    # <source_root>/images/train, images/val and <source_root>/labels/train, labels/val
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    run_experiments(src)
