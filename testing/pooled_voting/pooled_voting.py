import os
import yaml
from typing import List, Tuple
from ultralytics import YOLO

from helpers.k_fold import _resolve_path


def _resolve_path(path):
    """Resolve a path to an absolute path."""
    return os.path.abspath(os.path.expanduser(path))



def mAP_on_fold(test_dir, model_dir):
    test_dir = _resolve_path(test_dir)
    model_dir = _resolve_path(model_dir)
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



model_dir = 'strat_5_unbiased_models'
dataset = 'start_5_dataset'
fold_paths = [f'{dataset}/fold_{i}' for i in range(5)]
model_paths = [f'{model_dir}/fold_{i}' for i in range(5)]
for fold_dir,model_path in zip(fold_paths, model_paths):
    valid_model_paths = [model_path for model_path in model_paths if model_path != model_path]
    scores = []
    for model in valid_model_paths:
        print(f"Evaluating {model} on {fold_dir}")
        # Here you would load the model and evaluate it on the fold_dir dataset
        scores.append(mAP_on_fold(model, fold_dir))
        
