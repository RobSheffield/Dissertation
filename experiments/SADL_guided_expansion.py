import os
import random
import shutil
import heapq
import sys
import json
import csv
import secrets
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.format_converter import convert_gt_to_yolo
from testing.SADL import run_SADL
from testing.SADL.map_eval import evaluate_map50_on_image_subset

from stages import train_model
from stages.model_training import create_yaml
from ultralytics import YOLO
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _evaluate_test_subset(model_weights, image_names, images_dir, labels_dir, bin_root):
    image_count, map50, f1 = evaluate_map50_on_image_subset(
        yolo_model=YOLO(model_weights),
        image_names=image_names,
        images_dir=images_dir,
        labels_dir=labels_dir,
        bin_root=bin_root,
    )
    return {
        "images": int(image_count),
        "map50": float(map50),
        "f1": float(f1),
    }


def _save_experiment_results(rows, output_csv):
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "run",
        "seed",
        "model",
        "test_images",
        "map50",
        "f1",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_aggregate_results(rows, output_csv):
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "runs", "mean_map50", "mean_f1"])
        for model_name in sorted(grouped.keys()):
            model_rows = grouped[model_name]
            n = len(model_rows)
            mean_map50 = sum(r["map50"] for r in model_rows) / n if n else float("nan")
            mean_f1 = sum(r["f1"] for r in model_rows) / n if n else float("nan")
            writer.writerow([model_name, n, mean_map50, mean_f1])


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


def run_guided_experiment(
    castings_dir="Castings",
    output_root="testing",
    portion_1=0.4,
    portion_2=0.4,
    portion_3=0.2,
    n_runs=10,
    base_model_path="yolo11n.pt",
    epochs=150,
    image_size=1280,
    seed=None,
):
    run_seeds = [seed] * n_runs if seed is not None else [secrets.randbelow(2**31 - 1) + 1 for _ in range(n_runs)]
    results_rows = []

    for run_idx in range(n_runs):
        run_num = run_idx + 1
        run_seed = run_seeds[run_idx]

        print(f"\n=== Guided SADL run {run_num}/{n_runs} (seed={run_seed}) ===")

        try:
            guide_output = f"{output_root}/sadl_guided_run_{run_num}"
            rand_output = f"{output_root}/sadl_random_run_{run_num}"
            guide2_output = f"{output_root}/sadl_guided_alt_run_{run_num}"
            baseline_model_dir = os.path.join(PROJECT_ROOT, f"{output_root}/sadl_baseline_model_run_{run_num}")
            guided_model_dir = os.path.join(PROJECT_ROOT, f"{output_root}/sadl_guided_model_run_{run_num}")
            random_model_dir = os.path.join(PROJECT_ROOT, f"{output_root}/sadl_random_model_run_{run_num}")

            print(f"[{run_num}] building partitions")
            build_partition(
                castings_dir=castings_dir,
                output_dir=guide_output,
                seed=run_seed,
                portion_1=portion_1,
                portion_2=portion_2,
                portion_3=portion_3,
            )
            build_partition(
                castings_dir=castings_dir,
                output_dir=rand_output,
                seed=run_seed,
                portion_1=portion_1,
                portion_2=portion_2,
                portion_3=portion_3,
            )
            build_partition(
                castings_dir=castings_dir,
                output_dir=guide2_output,
                seed=run_seed,
                portion_1=portion_1,
                portion_2=portion_2,
                portion_3=portion_3,
            )

            guide_root = os.path.join(PROJECT_ROOT, guide_output)
            rand_root = os.path.join(PROJECT_ROOT, rand_output)
            guide2_root = os.path.join(PROJECT_ROOT, guide2_output)

            print(f"[{run_num}] baseline training")
            _train_minimal(guide_root, baseline_model_dir, epochs=epochs, base_model_path=base_model_path, image_size=image_size)
            baseline_weights = os.path.join(baseline_model_dir, "weights", "best.pt")

            print(f"[{run_num}] baseline SADL scoring")
            run_SADL.run_sadl(
                model_path=baseline_weights,
                train_path=os.path.join(guide_root, "images", "train"),
                val_path=os.path.join(guide_root, "images", "val"),
                train_labels_path=os.path.join(guide_root, "labels", "train"),
                val_labels_path=os.path.join(guide_root, "labels", "val"),
            )

            test_image_dir = os.path.join(guide_root, "images", "test")
            test_label_dir = os.path.join(guide_root, "labels", "test")
            test_image_names = [
                f for f in os.listdir(test_image_dir)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ]

            baseline_metrics = _evaluate_test_subset(
                model_weights=baseline_weights,
                image_names=test_image_names,
                images_dir=test_image_dir,
                labels_dir=test_label_dir,
                bin_root=os.path.join(PROJECT_ROOT, f"binned_results/_temp_eval_guide_test_run_{run_num}_baseline"),
            )
            results_rows.append(
                {
                    "run": run_num,
                    "seed": run_seed,
                    "model": "baseline",
                    "test_images": baseline_metrics["images"],
                    "map50": baseline_metrics["map50"],
                    "f1": baseline_metrics["f1"],
                }
            )

            print(f"[{run_num}] ranking folders for guided selection")
            val_image_dir = os.path.join(guide_root, "images", "val")
            folders, lsa_image_scores = run_SADL.score_folder_lsa(
                model_path=baseline_weights,
                train_path=os.path.join(guide_root, "images", "train"),
                target_path=val_image_dir,
            )
            folders = sorted(folders.items(), key=lambda x: x[1], reverse=True)
            rng = random.Random(run_seed)
            top_half_count = int(len(folders) * 0.5)
            top_half_folders = [folder for folder, _ in folders[:top_half_count]]

            guided_k = int(len(top_half_folders) * 0.5)
            selected_folders = rng.sample(top_half_folders, k=guided_k) if guided_k > 0 else []

            all_candidate_folders = [folder for folder, _ in folders]
            random_folders = rng.sample(all_candidate_folders, k=guided_k) if guided_k > 0 else []

            move_folders_to_train(selected_folders, guide_root, source_splits=("val",))
            move_folders_to_train(random_folders, rand_root, source_splits=("val",))

            n_folders = len(folders)
            if n_folders > 0:
                low_idx = int(n_folders * 0.25)
                high_idx = int(n_folders * 0.75)
                middle_pool = [folder for folder, _ in folders[low_idx:high_idx]]
            else:
                middle_pool = []

            guided2_k = int(len(middle_pool) * 0.5)
            selected_folders2 = rng.sample(middle_pool, k=guided2_k) if guided2_k > 0 else []
            move_folders_to_train(selected_folders2, guide2_root, source_splits=("val",))

            print(f"[{run_num}] guided training")
            _train_minimal(guide_root, guided_model_dir, epochs=epochs, base_model_path=base_model_path, image_size=image_size)
            print(f"[{run_num}] random training")
            _train_minimal(rand_root, random_model_dir, epochs=epochs, base_model_path=base_model_path, image_size=image_size)

            guided_weights = os.path.join(guided_model_dir, "weights", "best.pt")
            random_weights = os.path.join(random_model_dir, "weights", "best.pt")
            guided2_model_dir = os.path.join(PROJECT_ROOT, f"{output_root}/sadl_guided_alt_model_run_{run_num}")

            guided_metrics = _evaluate_test_subset(
                model_weights=guided_weights,
                image_names=test_image_names,
                images_dir=test_image_dir,
                labels_dir=test_label_dir,
                bin_root=os.path.join(PROJECT_ROOT, f"binned_results/_temp_eval_guide_test_run_{run_num}_guided"),
            )
            results_rows.append(
                {
                    "run": run_num,
                    "seed": run_seed,
                    "model": "guided",
                    "test_images": guided_metrics["images"],
                    "map50": guided_metrics["map50"],
                    "f1": guided_metrics["f1"],
                }
            )

            print(f"[{run_num}] guided-alt training")
            _train_minimal(guide2_root, guided2_model_dir, epochs=epochs, base_model_path=base_model_path, image_size=image_size)
            guided2_weights = os.path.join(guided2_model_dir, "weights", "best.pt")
            guided2_metrics = _evaluate_test_subset(
                model_weights=guided2_weights,
                image_names=test_image_names,
                images_dir=test_image_dir,
                labels_dir=test_label_dir,
                bin_root=os.path.join(PROJECT_ROOT, f"binned_results/_temp_eval_guide2_test_run_{run_num}_guided2"),
            )
            results_rows.append(
                {
                    "run": run_num,
                    "seed": run_seed,
                    "model": "guided2",
                    "test_images": guided2_metrics["images"],
                    "map50": guided2_metrics["map50"],
                    "f1": guided2_metrics["f1"],
                }
            )

            random_metrics = _evaluate_test_subset(
                model_weights=random_weights,
                image_names=test_image_names,
                images_dir=test_image_dir,
                labels_dir=test_label_dir,
                bin_root=os.path.join(PROJECT_ROOT, f"binned_results/_temp_eval_rand_test_run_{run_num}_random"),
            )
            results_rows.append(
                {
                    "run": run_num,
                    "seed": run_seed,
                    "model": "random",
                    "test_images": random_metrics["images"],
                    "map50": random_metrics["map50"],
                    "f1": random_metrics["f1"],
                }
            )

            print(
                f"Run {run_num}/{n_runs} (seed={run_seed}) test metrics | "
                f"baseline F1={baseline_metrics['f1']:.4f}, "
                f"guided F1={guided_metrics['f1']:.4f}, "
                f"random F1={random_metrics['f1']:.4f}"
            )
        except Exception as exc:
            print(f"Run {run_num} failed at a guided-expansion stage: {exc}")
            continue

    detailed_csv = os.path.join(PROJECT_ROOT, output_root, "SADL", "guided_sadl_test_results.csv")
    aggregate_csv = os.path.join(PROJECT_ROOT, output_root, "SADL", "guided_sadl_test_results_summary.csv")
    _save_experiment_results(results_rows, detailed_csv)
    _save_aggregate_results(results_rows, aggregate_csv)

    print(f"Saved detailed test results: {detailed_csv}")
    print(f"Saved aggregate test results: {aggregate_csv}")
    return detailed_csv, aggregate_csv

def move_folders_to_train(folders, output_root, source_splits=("val",)):
    for folder in folders:
        for split in source_splits:
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


def _train_minimal(data_dir, model_dir, epochs=50, base_model_path="yolo11n.pt", image_size=1280):
    data_dir = os.path.abspath(data_dir)
    model_dir = os.path.abspath(model_dir)
    data_yaml = os.path.join(data_dir, "data.yaml")
    create_yaml(data_dir, data_yaml, ["defect"])

    train_img_dir = os.path.join(data_dir, "images", "train")
    train_img_count = len([f for f in os.listdir(train_img_dir) if f.lower().endswith(IMAGE_EXTENSIONS)])
    model_info = json.dumps(
        {
            "name": os.path.basename(model_dir),
            "model": base_model_path,
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
        weights=base_model_path,
        img_size=str(image_size),
        epochs=str(epochs),
        device="auto",
        flips = True
    )

def main(castings_dir="Castings", output_root="testing", portion_1=0.4, portion_2=0.4, portion_3=0.2, n_runs=10, base_model_path="yolo11n.pt", epochs=150, image_size=1280, seed=None):
    run_guided_experiment(
        castings_dir=castings_dir,
        output_root=output_root,
        portion_1=portion_1,
        portion_2=portion_2,
        portion_3=portion_3,
        n_runs=n_runs,
        base_model_path=base_model_path,
        epochs=epochs,
        image_size=image_size,
        seed=seed,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run the guided SADL retraining experiment.")
    parser.add_argument("--castings-dir", default="Castings")
    parser.add_argument("--output-root", default="testing")
    parser.add_argument("--portion-1", type=float, default=0.4)
    parser.add_argument("--portion-2", type=float, default=0.4)
    parser.add_argument("--portion-3", type=float, default=0.2)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--base-model-path", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--image-size", type=int, default=1280, choices=(640, 1280))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    main(
        castings_dir=args.castings_dir,
        output_root=args.output_root,
        portion_1=args.portion_1,
        portion_2=args.portion_2,
        portion_3=args.portion_3,
        n_runs=args.n_runs,
        base_model_path=args.base_model_path,
        epochs=args.epochs,
        image_size=args.image_size,
        seed=args.seed,
    )