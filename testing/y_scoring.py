import helpers.image_analysis
import os
import shutil
from pathlib import Path
from ultralytics import YOLO


def make_y_segments(images_dir, labels_dir, output_root, n_segments=5):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_root = Path(output_root)

    segment_dirs = []
    for i in range(n_segments):
        seg_img = output_root / f"segment_{i}" / "images"
        seg_lbl = output_root / f"segment_{i}" / "labels"
        seg_img.mkdir(parents=True, exist_ok=True)
        seg_lbl.mkdir(parents=True, exist_ok=True)
        segment_dirs.append((seg_img, seg_lbl))

    for label_file in labels_dir.glob("*.txt"):
        kept = {i: [] for i in range(n_segments)}

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                y = float(parts[2])
                seg = min(int(y * n_segments), n_segments - 1) #getting index (0 indexed)
                kept[seg].append(line.strip())

        stem = label_file.stem
        image_file = f"{stem}.png"

        for seg, lines in kept.items():
            if not lines:
                continue

            seg_img, seg_lbl = segment_dirs[seg]
            shutil.copy2(image_file, seg_img / image_file.name)

            with open(seg_lbl / label_file.name, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")


def write_yaml(yaml_path, val_images_path):
    yaml_text = f"""names:
- '0'
nc: 1
val: {Path(val_images_path).as_posix()}
"""
    Path(yaml_path).write_text(yaml_text, encoding="utf-8")


def evaluate_segments(model_path, segmented_root, n_segments=5):
    model = YOLO(model_path)
    results = {}
    #calculate the mAP50 for each segment group
    for i in range(n_segments):
        val_path = Path(segmented_root) / f"segment_{i}" / "images"
        yaml_path = Path(segmented_root) / f"segment_{i}" / "dataset.yaml"
        write_yaml(yaml_path, val_path)
        metrics = model.val(data=str(yaml_path), split="val")
        results[i] = float(metrics.box.map50)
    return results


make_y_segments(
    images_dir="fold_paths/fold_1/images",
    labels_dir="fold_paths/fold_1/labels",
    output_root="temp_y_segments",
    n_segments=5
)

scores = evaluate_segments(
    model_path="fold_paths/fold_1/weights/best.pt",
    segmented_root="temp_y_segments",
    n_segments=5
)

for seg, map50 in scores.items():
    print(f"Segment {seg}: mAP50={map50:.4f}")


scores = evaluate_segments(
    model_path="fold_paths/fold_1/weights/best.pt",
    segmented_root="temp_y_segments",
    n_segments=5
)
for seg, map50 in scores.items():
    print(f"Segment {seg}: mAP50={map50:.4f}")
