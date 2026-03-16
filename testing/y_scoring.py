import sys
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
sys.path.append(str(Path(__file__).resolve().parent.parent))

import helpers.image_analysis

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
        image_name = f"{stem}.png"  # Renamed to clarify this is just a filename
        source_image_path = images_dir / image_name # The actual path to the image

        for seg, lines in kept.items():
            if not lines:
                continue

            seg_img, seg_lbl = segment_dirs[seg]
            
            # Check if source exists before copying
            if source_image_path.exists():
                shutil.copy2(source_image_path, seg_img / image_name)
            else:
                print(f"Warning: Image not found at {source_image_path}")

            with open(seg_lbl / label_file.name, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")


def write_yaml(yaml_path, val_images_path):
    # Ultralytics requires 'train' even for validation-only runs
    yaml_text = f"""
path: {Path(val_images_path).parent.resolve().as_posix()}
train: images
val: images
names:
  0: defect
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
        # Added workers=0 to help avoid multiprocessing issues on Windows if needed
        metrics = model.val(data=str(yaml_path), split="val", workers=0) 
        results[i] = float(metrics.box.map50)
    return results

if __name__ == '__main__':
    make_y_segments(
        images_dir="data/images/val",
        labels_dir="data/labels/val",
        output_root="temp_y_segments",
        n_segments=5
    )

    scores = evaluate_segments(
        model_path="flipers/fold_1_noFLIP/weights/best.pt",
        segmented_root="temp_y_segments",
        n_segments=5
    )

    for seg, map50 in scores.items():
        print(f"Segment {seg}: mAP50={map50:.4f}")



    scores = evaluate_segments(
        model_path="flipers/fold_1/weights/best.pt",
        segmented_root="temp_y_segments",
        n_segments=5
    )

    for seg, map50 in scores.items():
        print(f"Segment {seg}: mAP50={map50:.4f}")