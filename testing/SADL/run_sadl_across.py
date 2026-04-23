import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import DSA
import LSA
import sadl_helpers


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


class ImageDataset(Dataset):
	def __init__(self, image_dir, image_size=640):
		self.image_dir = image_dir
		self.image_size = image_size
		self.image_files = sorted(
			[
				name
				for name in os.listdir(image_dir)
				if name.lower().endswith(IMAGE_EXTS)
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
		image = cv2.resize(
			image,
			(self.image_size, self.image_size),
			interpolation=cv2.INTER_LINEAR,
		)
		image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
		return image


def _load_model(model_path):
	model = torch.load(model_path, weights_only=False)
	if isinstance(model, dict) and "model" in model:
		model = model["model"]
	return model


def _extract_ats(model, image_dir, device, batch_size=8, image_size=640):
	dataset = ImageDataset(image_dir=image_dir, image_size=image_size)
	if len(dataset) == 0:
		raise ValueError(f"No images found in '{image_dir}'")

	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
	ats = sadl_helpers.get_ats(model, loader, device)
	return dataset.image_files, ats


def _median(values):
	return float(np.median(np.asarray(values, dtype=np.float64)))


def _a12(train_scores, val_scores):
	"""
	Probability that a random val score is greater than a random train score.
	Includes 0.5 credit for ties and is robust to unequal sample sizes.
	"""
	train = np.sort(np.asarray(train_scores, dtype=np.float64))
	val = np.asarray(val_scores, dtype=np.float64)

	if train.size == 0 or val.size == 0:
		return float("nan")

	less_counts = np.searchsorted(train, val, side="left")
	leq_counts = np.searchsorted(train, val, side="right")
	greater_probs = (less_counts + 0.5 * (leq_counts - less_counts)) / train.size
	return float(np.mean(greater_probs))


def _write_per_image_scores(path, fold_name, split, image_names, scores, metric):
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["fold", "split", "filename", metric])
		for name, score in zip(image_names, scores):
			writer.writerow([fold_name, split, name, float(score)])


def _fold_key(name):
	try:
		return int(name.split("_", 1)[1])
	except (IndexError, ValueError):
		return 10**9


def _build_fold_surprise_heatmap(fold_to_scores, output_dir, metric, split="val", n_bins=30):
	"""
	Build a final heatmap of val surprise distributions against fold.
	Each fold column is normalized to a probability distribution so folds with
	different image counts remain directly comparable.
	"""
	folds = sorted(fold_to_scores.keys(), key=_fold_key)
	if not folds:
		return None

	all_scores = np.concatenate([
		np.asarray(fold_to_scores[fold], dtype=np.float64)
		for fold in folds
		if len(fold_to_scores[fold]) > 0
	])
	if all_scores.size == 0:
		return None

	# Robust clipping for readable contrast when extreme outliers exist.
	lo = float(np.quantile(all_scores, 0.01))
	hi = float(np.quantile(all_scores, 0.99))
	if hi <= lo:
		lo = float(np.min(all_scores))
		hi = float(np.max(all_scores))
	if hi <= lo:
		hi = lo + 1e-9

	bins = np.linspace(lo, hi, int(n_bins) + 1)
	heat = np.zeros((int(n_bins), len(folds)), dtype=np.float64)

	for col, fold in enumerate(folds):
		scores = np.asarray(fold_to_scores[fold], dtype=np.float64)
		if scores.size == 0:
			continue
		clipped = np.clip(scores, lo, hi)
		hist, _ = np.histogram(clipped, bins=bins)
		den = hist.sum()
		if den > 0:
			heat[:, col] = hist / den

	# Save numeric matrix for reproducibility.
	csv_path = Path(output_dir) / f"{metric}_{split}_fold_heatmap_matrix.csv"
	with open(csv_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		header = ["bin_low", "bin_high"] + folds
		writer.writerow(header)
		for i in range(heat.shape[0]):
			writer.writerow([float(bins[i]), float(bins[i + 1]), *heat[i, :].tolist()])

	# Render PNG with OpenCV colormap to avoid extra plotting deps.
	heat_img = heat[::-1, :]  # high surprise at top visually
	if np.max(heat_img) > 0:
		norm = (heat_img / np.max(heat_img) * 255.0).astype(np.uint8)
	else:
		norm = np.zeros_like(heat_img, dtype=np.uint8)

	color = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
	scale_y = 16
	scale_x = 56
	resized = cv2.resize(
		color,
		(color.shape[1] * scale_x, color.shape[0] * scale_y),
		interpolation=cv2.INTER_NEAREST,
	)

	margin_top = 56
	margin_bottom = 72
	margin_left = 90
	margin_right = 24
	canvas_h = resized.shape[0] + margin_top + margin_bottom
	canvas_w = resized.shape[1] + margin_left + margin_right
	canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
	canvas[margin_top:margin_top + resized.shape[0], margin_left:margin_left + resized.shape[1]] = resized

	cv2.putText(
		canvas,
		f"{metric.upper()} Heatmap: {split.title()} Surprise Distribution by Fold",
		(18, 30),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.75,
		(20, 20, 20),
		2,
		cv2.LINE_AA,
	)

	for idx, fold in enumerate(folds):
		x = margin_left + idx * scale_x + 6
		cv2.putText(
			canvas,
			fold.replace("fold_", "F"),
			(x, canvas_h - 24),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(25, 25, 25),
			1,
			cv2.LINE_AA,
		)

	for y_frac, label in ((0.0, f"{hi:.3f}"), (0.5, f"{(lo + hi) / 2.0:.3f}"), (1.0, f"{lo:.3f}")):
		y = int(margin_top + y_frac * resized.shape[0])
		cv2.putText(
			canvas,
			label,
			(10, max(12, min(canvas_h - 8, y))),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(25, 25, 25),
			1,
			cv2.LINE_AA,
		)

	png_path = Path(output_dir) / f"{metric}_{split}_fold_heatmap.png"
	cv2.imwrite(str(png_path), canvas)
	return png_path


def _load_saved_scores(output_dir, split="val", metric="lsa"):
	output_dir = Path(output_dir)
	fold_to_scores = {}
	pattern = f"fold_*_{split}_{metric}.csv"

	for csv_path in sorted(output_dir.glob(pattern), key=lambda p: _fold_key(p.stem.split("_", 2)[0] + "_" + p.stem.split("_", 2)[1])):
		with open(csv_path, "r", encoding="utf-8", newline="") as f:
			reader = csv.DictReader(f)
			scores = [float(row[metric]) for row in reader if row.get(metric) not in (None, "")]
		if not scores:
			continue
		parts = csv_path.stem.split("_")
		fold_name = f"{parts[0]}_{parts[1]}"
		fold_to_scores[fold_name] = scores

	return dict(sorted(fold_to_scores.items(), key=lambda kv: _fold_key(kv[0])))


def _build_ecdf_panel(fold_to_scores, output_dir, metric="lsa", split="val"):
	folds = sorted(fold_to_scores.keys(), key=_fold_key)
	if not folds:
		return None

	all_scores = np.concatenate([
		np.asarray(fold_to_scores[f], dtype=np.float64)
		for f in folds
		if len(fold_to_scores[f]) > 0
	])
	if all_scores.size == 0:
		return None

	x_min = float(np.min(all_scores))
	x_max = float(np.max(all_scores))
	if x_max <= x_min:
		x_max = x_min + 1e-9

	cols = min(5, len(folds))
	rows = int(np.ceil(len(folds) / cols))
	panel_w = 360
	panel_h = 240
	pad = 24
	top_title = 56

	canvas_h = top_title + rows * panel_h + (rows + 1) * pad
	canvas_w = cols * panel_w + (cols + 1) * pad
	canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

	cv2.putText(
		canvas,
		f"ECDF Panel ({split.upper()} {metric.upper()}) by Fold",
		(18, 34),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.9,
		(20, 20, 20),
		2,
		cv2.LINE_AA,
	)

	for i, fold in enumerate(folds):
		r = i // cols
		c = i % cols
		x0 = pad + c * (panel_w + pad)
		y0 = top_title + pad + r * (panel_h + pad)

		left = x0 + 54
		right = x0 + panel_w - 20
		top = y0 + 20
		bottom = y0 + panel_h - 36

		# axes
		cv2.rectangle(canvas, (left, top), (right, bottom), (210, 210, 210), 1)
		cv2.line(canvas, (left, bottom), (right, bottom), (100, 100, 100), 1)
		cv2.line(canvas, (left, top), (left, bottom), (100, 100, 100), 1)

		scores = np.sort(np.asarray(fold_to_scores[fold], dtype=np.float64))
		n = scores.size
		y_vals = np.arange(1, n + 1, dtype=np.float64) / n

		# Draw ECDF as stair line.
		prev_x = left
		prev_y = bottom
		for sx, sy in zip(scores, y_vals):
			x_norm = (sx - x_min) / (x_max - x_min)
			x = int(left + x_norm * (right - left))
			y = int(bottom - sy * (bottom - top))
			cv2.line(canvas, (prev_x, prev_y), (x, prev_y), (31, 119, 180), 2)
			cv2.line(canvas, (x, prev_y), (x, y), (31, 119, 180), 2)
			prev_x, prev_y = x, y

		# small labels
		cv2.putText(canvas, fold.replace("fold_", "Fold "), (x0 + 10, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
		cv2.putText(canvas, f"n={n}", (right - 58, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
		cv2.putText(canvas, "0", (left - 16, bottom + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
		cv2.putText(canvas, "1", (left - 16, top + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)

	# Shared x-axis range note.
	cv2.putText(
		canvas,
		f"Shared {metric.upper()} range: [{x_min:.3f}, {x_max:.3f}]",
		(18, canvas_h - 10),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.55,
		(40, 40, 40),
		1,
		cv2.LINE_AA,
	)

	png_path = Path(output_dir) / f"{metric}_{split}_ecdf_panel.png"
	cv2.imwrite(str(png_path), canvas)
	return png_path


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
	return 1 if len(non_empty_lines) > 1 else 0


def _build_folder_defect_labels(image_files, castings_root):
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


def run_across(models_root, folds_root, output_dir, image_size=640, batch_size=8, var_threshold=1e-5, metric="lsa", castings_root=None):
	models_root = Path(models_root)
	folds_root = Path(folds_root)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	metric = metric.lower().strip()
	if metric not in {"lsa", "dsa"}:
		raise ValueError("metric must be one of: lsa, dsa")

	if metric == "dsa":
		if castings_root is None:
			project_root = Path(__file__).resolve().parents[2]
			castings_root = project_root / "Castings"
		castings_root = Path(castings_root)
		if not castings_root.is_dir():
			raise FileNotFoundError(f"Castings directory not found: {castings_root}")

	if not models_root.is_dir():
		raise FileNotFoundError(f"Models directory not found: {models_root}")
	if not folds_root.is_dir():
		raise FileNotFoundError(f"Folds directory not found: {folds_root}")

	model_folds = {
		d.name
		for d in models_root.iterdir()
		if d.is_dir() and d.name.startswith("fold_")
	}
	data_folds = {
		d.name
		for d in folds_root.iterdir()
		if d.is_dir() and d.name.startswith("fold_")
	}
	folds = sorted(model_folds & data_folds, key=_fold_key)

	if not folds:
		raise RuntimeError(
			f"No overlapping fold_* folders found between '{models_root}' and '{folds_root}'"
		)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	summary_rows = []
	val_scores_by_fold = {}

	for fold in folds:
		model_path = models_root / fold / "weights" / "best.pt"
		train_dir = folds_root / fold / "images" / "train"
		val_dir = folds_root / fold / "images" / "val"

		if not model_path.is_file():
			print(f"Skipping {fold}: missing model at {model_path}")
			continue
		if not train_dir.is_dir() or not val_dir.is_dir():
			print(f"Skipping {fold}: missing train/val dirs under {folds_root / fold}")
			continue

		print(f"Processing {fold}")
		model = _load_model(str(model_path))

		train_files, train_ats = _extract_ats(
			model,
			str(train_dir),
			device=device,
			batch_size=batch_size,
			image_size=image_size,
		)
		val_files, val_ats = _extract_ats(
			model,
			str(val_dir),
			device=device,
			batch_size=batch_size,
			image_size=image_size,
		)

		if metric == "lsa":
			train_scores = LSA.fetch_lsa(train_ats, train_ats, var_threshold=var_threshold)
			val_scores = LSA.fetch_lsa(train_ats, val_ats, var_threshold=var_threshold)
		else:
			train_labels = _build_folder_defect_labels(train_files, str(castings_root))
			val_labels = _build_folder_defect_labels(val_files, str(castings_root))
			train_scores = DSA.fetch_dsa(train_ats, train_labels, train_ats, train_labels, k=3, eps=1e-12)
			val_scores = DSA.fetch_dsa(train_ats, train_labels, val_ats, val_labels, k=3, eps=1e-12)

		val_scores_by_fold[fold] = val_scores

		train_median = _median(train_scores)
		val_median = _median(val_scores)
		median_gap = val_median - train_median
		a12 = _a12(train_scores, val_scores)

		_write_per_image_scores(
			output_dir / f"{fold}_train_{metric}.csv",
			fold,
			"train",
			train_files,
			train_scores,
			metric,
		)
		_write_per_image_scores(
			output_dir / f"{fold}_val_{metric}.csv",
			fold,
			"val",
			val_files,
			val_scores,
			metric,
		)

		summary_rows.append(
			{
				"fold": fold,
				"metric": metric,
				"n_train": len(train_scores),
				"n_val": len(val_scores),
				"train_score_median": train_median,
				"val_score_median": val_median,
				"median_gap_val_minus_train": median_gap,
				"a12_val_greater_than_train": a12,
			}
		)

	heatmap_path = _build_fold_surprise_heatmap(val_scores_by_fold, output_dir, metric=metric, split="val")
	ecdf_path = _build_ecdf_panel(val_scores_by_fold, output_dir, metric=metric, split="val")
	summary_path = output_dir / f"{metric}_across_folds_summary.csv"
	with open(summary_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"fold",
				"metric",
				"n_train",
				"n_val",
				"train_score_median",
				"val_score_median",
				"median_gap_val_minus_train",
				"a12_val_greater_than_train",
			],
		)
		writer.writeheader()
		writer.writerows(summary_rows)

	if heatmap_path is not None:
		print(f"Saved heatmap: {heatmap_path}")
	if ecdf_path is not None:
		print(f"Saved ECDF panel: {ecdf_path}")
	print(f"Saved summary: {summary_path}")
	return summary_path


def _default_paths():
	script_dir = Path(__file__).resolve().parent
	candidate_dirs = [
		script_dir.parent / "final_datas",
		script_dir.parent.parent / "final_datas",
	]
	final_datas = next((p for p in candidate_dirs if p.is_dir()), candidate_dirs[0])
	return (
		final_datas / "strat_10_unbiased_models",
		final_datas / "Folds_strat_10",
		final_datas / "sadl_across_outputs",
	)


def main():
	default_models, default_folds, default_out = _default_paths()

	parser = argparse.ArgumentParser(
		description=(
			"Compute LSA across all fold models by comparing train-vs-val "
			"surprise distributions for each fold."
		)
	)
	parser.add_argument("--models-root", default=str(default_models))
	parser.add_argument("--folds-root", default=str(default_folds))
	parser.add_argument("--output-dir", default=str(default_out))
	parser.add_argument("--image-size", type=int, default=640)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--var-threshold", type=float, default=1e-5)
	parser.add_argument("--metric", choices=["lsa", "dsa"], default="lsa")
	parser.add_argument("--castings-root", default=None)
	parser.add_argument(
		"--ecdf-only",
		action="store_true",
		help="Build ECDF panel from saved fold CSVs in output-dir without recomputing LSA.",
	)
	parser.add_argument(
		"--ecdf-split",
		choices=["val", "train"],
		default="val",
		help="Which saved split CSVs to use when --ecdf-only is set.",
	)
	args = parser.parse_args()

	if args.ecdf_only:
		fold_to_scores = _load_saved_scores(args.output_dir, split=args.ecdf_split, metric=args.metric)
		if not fold_to_scores:
			raise RuntimeError(
				f"No saved fold score CSVs found in '{args.output_dir}' for split='{args.ecdf_split}', metric='{args.metric}'."
			)
		ecdf_path = _build_ecdf_panel(fold_to_scores, args.output_dir, metric=args.metric, split=args.ecdf_split)
		if ecdf_path is None:
			raise RuntimeError("Could not build ECDF panel from saved scores.")
		print(f"Saved ECDF panel: {ecdf_path}")
		return

	run_across(
		models_root=args.models_root,
		folds_root=args.folds_root,
		output_dir=args.output_dir,
		image_size=args.image_size,
		batch_size=args.batch_size,
		var_threshold=args.var_threshold,
		metric=args.metric,
		castings_root=args.castings_root,
	)


if __name__ == "__main__":
	main()
