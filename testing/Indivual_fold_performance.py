import argparse
import csv
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _fold_key(name):
	try:
		return int(name.split("_", 1)[1])
	except (IndexError, ValueError):
		return 10**9


def _resolve_default_roots():
	project_root = Path(__file__).resolve().parents[1]
	final_datas = project_root / "testing" / "final_datas"

	unbiased_candidates = [
		final_datas / "unbiased",
		final_datas / "strat_10_unbiased_models",
	]
	models_root = next((p for p in unbiased_candidates if p.is_dir()), unbiased_candidates[-1])
	folds_root = final_datas / "Folds_strat_10"
	out_root = final_datas / "folder_f1_outputs"
	return models_root, folds_root, out_root


def _folder_from_image_name(image_name):
	return image_name.split("_", 1)[0]


def _build_temp_subset_dataset(images_dir, labels_dir, image_names, temp_root):
	images_out = Path(temp_root) / "images" / "val"
	labels_out = Path(temp_root) / "labels" / "val"
	images_out.mkdir(parents=True, exist_ok=True)
	labels_out.mkdir(parents=True, exist_ok=True)

	for image_name in image_names:
		source_image = Path(images_dir) / image_name
		if not source_image.is_file():
			continue

		shutil.copy2(source_image, images_out / image_name)

		label_name = source_image.stem + ".txt"
		source_label = Path(labels_dir) / label_name
		destination_label = labels_out / label_name
		if source_label.is_file():
			shutil.copy2(source_label, destination_label)
		else:
			destination_label.write_text("", encoding="utf-8")

	yaml_path = Path(temp_root) / "data.yaml"
	yaml_path.write_text(
		"path: " + Path(temp_root).resolve().as_posix() + "\n"
		"train: images/val\n"
		"val: images/val\n"
		"test: images/val\n"
		"nc: 1\n"
		"names:\n"
		"  - defect\n",
		encoding="utf-8",
	)
	return yaml_path


def _evaluate_subset_with_yolo(model, images_dir, labels_dir, image_names, temp_root):
	yaml_path = _build_temp_subset_dataset(images_dir, labels_dir, image_names, temp_root)
	metrics = model.val(
		data=str(yaml_path),
		split="val",
		imgsz=640,
		batch=16,
		workers=0,
		plots=False,
		verbose=False,
	)
	precision = float(getattr(metrics.box, "mp", float("nan")))
	recall = float(getattr(metrics.box, "mr", float("nan")))
	map50 = float(getattr(metrics.box, "map50", float("nan")))
	map5095 = float(getattr(metrics.box, "map", float("nan")))
	denom = precision + recall
	f1 = float(2.0 * precision * recall / denom) if denom > 0 else 0.0
	return {
		"precision": precision,
		"recall": recall,
		"map50": map50,
		"map5095": map5095,
		"f1": f1,
	}


def evaluate_fold_folder_metrics(model, images_dir, labels_dir):
	image_names = sorted(
		name for name in os.listdir(images_dir)
		if name.lower().endswith(IMAGE_EXTENSIONS)
	)

	per_folder = {}
	for image_name in image_names:
		folder = _folder_from_image_name(image_name)
		if folder not in per_folder:
			per_folder[folder] = {"images": 0, "image_names": []}
		per_folder[folder]["images"] += 1
		per_folder[folder]["image_names"].append(image_name)

	rows = []
	for folder, counts in sorted(per_folder.items()):
		with tempfile.TemporaryDirectory(prefix=f"folder_eval_{folder}_") as temp_root:
			metrics = _evaluate_subset_with_yolo(
				model=model,
				images_dir=images_dir,
				labels_dir=labels_dir,
				image_names=counts["image_names"],
				temp_root=temp_root,
			)

		rows.append(
			{
				"folder": folder,
				"images": counts["images"],
				"precision": metrics["precision"],
				"recall": metrics["recall"],
				"map50": metrics["map50"],
				"map5095": metrics["map5095"],
				"f1": metrics["f1"],
			}
		)

	return rows


def _draw_bar_chart(rows, overall_f1, out_png):
	if not rows:
		return None

	rows = sorted(rows, key=lambda r: (r["model_fold"], r["folder"]))
	labels = [f"{r['model_fold']} | {r['folder']}" for r in rows]
	values = [float(r["f1"]) for r in rows]
	colors = []
	palette = {
		"fold_1": "#4e79a7",
		"fold_2": "#f28e2b",
		"fold_3": "#e15759",
		"fold_4": "#76b7b2",
		"fold_5": "#59a14f",
		"fold_6": "#edc948",
		"fold_7": "#b07aa1",
		"fold_8": "#ff9da7",
		"fold_9": "#9c755f",
		"fold_10": "#bab0ab",
	}
	for row in rows:
		colors.append(palette.get(row["model_fold"], "#7f7f7f"))

	fig_w = max(10.5, 0.55 * len(rows) + 4.5)
	fig, ax = plt.subplots(figsize=(fig_w, 6.0))
	x = np.arange(len(rows))
	ax.axhline(overall_f1, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.55, zorder=0, label=f"Global F1 = {overall_f1:.3f}")
	ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4, zorder=2)

	ax.set_xticks(x)
	ax.set_xticklabels(labels, rotation=45, ha="right")
	ax.set_ylim(0.0, 1.06)
	ax.set_ylabel("F1 Score")
	ax.set_title("F1 by Model Fold")
	ax.grid(axis="y", alpha=0.25)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, borderaxespad=0.0)

	for idx, value in enumerate(values):
		if value >= 0.95:
			text_y = value - 0.025
			va = "top"
			text_color = "white"
		else:
			text_y = value + 0.02
			va = "bottom"
			text_color = "black"

		ax.text(
			idx,
			text_y,
			f"{value:.3f}",
			ha="center",
			va=va,
			fontsize=8,
			color=text_color,
			zorder=3,
		)

	fig.tight_layout(rect=(0, 0, 0.86, 1))
	out_png = Path(out_png)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
	plt.close(fig)
	return out_png


def run(models_root, folds_root, output_dir, split="val"):
	models_root = Path(models_root)
	folds_root = Path(folds_root)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	if not models_root.is_dir():
		raise FileNotFoundError(f"Models root not found: {models_root}")
	if not folds_root.is_dir():
		raise FileNotFoundError(f"Folds root not found: {folds_root}")

	model_folds = {d.name for d in models_root.iterdir() if d.is_dir() and d.name.startswith("fold_")}
	data_folds = {d.name for d in folds_root.iterdir() if d.is_dir() and d.name.startswith("fold_")}
	folds = sorted(model_folds & data_folds, key=_fold_key)

	if not folds:
		raise RuntimeError("No matching fold_* directories between models and folds.")

	all_rows = []
	all_model_images = []
	all_model_labels = []

	for fold in folds:
		model_path = models_root / fold / "weights" / "best.pt"
		image_dir = folds_root / fold / "images" / split
		label_dir = folds_root / fold / "labels" / split

		if not model_path.is_file():
			print(f"Skipping {fold}: missing model file at {model_path}")
			continue
		if not image_dir.is_dir() or not label_dir.is_dir():
			print(f"Skipping {fold}: missing split dirs in {folds_root / fold}")
			continue

		print(f"Evaluating {fold}")
		model = YOLO(str(model_path))
		fold_rows = evaluate_fold_folder_metrics(model=model, images_dir=image_dir, labels_dir=label_dir)

		for row in fold_rows:
			row["model_fold"] = fold
			all_rows.append(row)

		fold_image_names = sorted(name for name in os.listdir(image_dir) if name.lower().endswith(IMAGE_EXTENSIONS))
		all_model_images.extend([(model, image_dir, label_dir, fold_image_names)])

	csv_path = output_dir / "folder_f1_scores.csv"
	with open(csv_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=["model_fold", "folder", "images", "precision", "recall", "map50", "map5095", "f1"],
		)
		writer.writeheader()
		writer.writerows(all_rows)

	# Overall F1 from YOLO built-in validation on the full split for each evaluated model, then averaged across models.
	overall_f1_values = []
	for fold in folds:
		model_path = models_root / fold / "weights" / "best.pt"
		image_dir = folds_root / fold / "images" / split
		label_dir = folds_root / fold / "labels" / split
		if not model_path.is_file() or not image_dir.is_dir() or not label_dir.is_dir():
			continue
		with tempfile.TemporaryDirectory(prefix=f"overall_eval_{fold}_") as temp_root:
			model = YOLO(str(model_path))
			metrics = _evaluate_subset_with_yolo(
				model=model,
				images_dir=image_dir,
				labels_dir=label_dir,
				image_names=[name for name in os.listdir(image_dir) if name.lower().endswith(IMAGE_EXTENSIONS)],
				temp_root=temp_root,
			)
			overall_f1_values.append(metrics["f1"])

	overall_f1 = float(np.mean(overall_f1_values)) if overall_f1_values else 0.0
	chart_path = output_dir / "folder_f1_bar_chart.png"
	_draw_bar_chart(all_rows, overall_f1, chart_path)

	print(f"Saved scores: {csv_path}")
	print(f"Saved chart:  {chart_path}")
	return csv_path, chart_path


def main():
	default_models, default_folds, default_out = _resolve_default_roots()

	parser = argparse.ArgumentParser(description="Per-folder F1 scoring for each fold model.")
	parser.add_argument("--models-root", default=str(default_models))
	parser.add_argument("--folds-root", default=str(default_folds))
	parser.add_argument("--output-dir", default=str(default_out))
	parser.add_argument("--split", default="val", choices=["val", "train", "test"])
	args = parser.parse_args()

	run(
		models_root=args.models_root,
		folds_root=args.folds_root,
		output_dir=args.output_dir,
		split=args.split,
	)


if __name__ == "__main__":
	main()