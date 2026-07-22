from __future__ import annotations


MODEL_WEIGHT_OPTIONS = {
	"YOLOv11n": "yolo11n.pt",
	"YOLOv5mu": "yolov5m.pt",
}


def resolve_training_weights(model_name_or_path: str) -> str:
	"""Resolve a UI model choice or direct path to a YOLO weights file."""
	selected = (model_name_or_path or "").strip()
	if not selected:
		return MODEL_WEIGHT_OPTIONS["YOLOv11n"]

	if selected.endswith(".pt"):
		return selected

	return MODEL_WEIGHT_OPTIONS.get(selected, selected)