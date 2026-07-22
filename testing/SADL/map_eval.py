import csv
import os
import shutil
import yaml
from ultralytics import YOLO


def evaluate_map50_on_image_subset(yolo_model, image_names, images_dir, labels_dir, bin_root, imgsz=1280, batch=16):
    """
    Build a temporary YOLO dataset from a subset of image names and return
    (image_count, map50, f1), where f1 is derived from detection precision/recall.
    """
    bin_images = os.path.join(bin_root, "images")
    bin_labels = os.path.join(bin_root, "labels")
    os.makedirs(bin_images, exist_ok=True)
    os.makedirs(bin_labels, exist_ok=True)

    copied_images = 0
    for image_name in image_names:
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
            open(dst_lbl, "w", encoding="utf-8").close()

    if copied_images == 0:
        return 0, float("nan"), float("nan")

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
        workers=0,
        plots=False,
        verbose=False,
    )
    map50 = float(getattr(metrics.box, "map50", float("nan")))
    precision = float(metrics.results_dict.get("metrics/precision(B)", float("nan")))
    recall = float(metrics.results_dict.get("metrics/recall(B)", float("nan")))
    denom = precision + recall
    f1 = float(2.0 * precision * recall / denom) if denom > 0 else 0.0
    return copied_images, map50, f1


def compute_mAP_for_bins(paths_in_bin, model_path, images_dir, labels_dir, prefix="", imgsz=1280, batch=16):
    """
    Compute mAP50 and stricter detection-level F1 per bin, then write a CSV summary under binned_results.
    """
    yolo_model = YOLO(model_path)

    output_root = "binned_results"
    temp_root = os.path.join(output_root, f"_temp_eval_{prefix.strip('_') or 'bin'}")
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    os.makedirs(temp_root, exist_ok=True)

    summary_rows = []

    for i, bin_paths in enumerate(paths_in_bin):
        print(f"Computing F1 for bin {i} with {len(bin_paths)} images...")

        bin_root = os.path.join(temp_root, f"bin_{i}")
        copied_images, map50, f1 = evaluate_map50_on_image_subset(
            yolo_model=yolo_model,
            image_names=bin_paths,
            images_dir=images_dir,
            labels_dir=labels_dir,
            bin_root=bin_root,
            imgsz=imgsz,
            batch=batch,
        )

        summary_rows.append((i, copied_images, map50, f1))
        if copied_images == 0:
            print(f"Bin {i} F1: NaN")
        else:
            print(f"Bin {i} F1: {f1}")

    metrics_summary_path = os.path.join(output_root, f"{prefix}bin_metrics.csv")
    with open(metrics_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin", "images", "map50", "f1"])
        writer.writerows(summary_rows)

    summary_path = os.path.join(output_root, f"{prefix}bin_map50.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin", "images", "map50"])
        writer.writerows((row[0], row[1], row[2]) for row in summary_rows)

    print(f"Saved per-bin F1 summary: {metrics_summary_path}")
    return summary_rows
