from __future__ import annotations

import csv
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "testing" / "model_test_reports"
REPORT_PATH = REPORT_DIR / "existing_model_test_results.csv"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _is_target_model_dir(model_dir: Path) -> bool:
    parts = [part.lower() for part in model_dir.parts]
    return any(part.startswith("fold_") or "first_60_guide" in part for part in parts)


def _discover_model_dirs() -> list[Path]:
    model_dirs: list[Path] = []
    seen: set[Path] = set()

    for weight_path in PROJECT_ROOT.rglob("best.pt"):
        if weight_path.parent.name != "weights":
            continue

        model_dir = weight_path.parent.parent
        if not _is_target_model_dir(model_dir):
            continue

        resolved = model_dir.resolve()
        if resolved in seen:
            continue

        seen.add(resolved)
        model_dirs.append(model_dir)

    return sorted(model_dirs)


def _read_saved_data_path(args_path: Path) -> str | None:
    if not args_path.is_file():
        return None

    for line in args_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue

        value = stripped.split("data:", 1)[1].strip().strip('"').strip("'")
        return value or None

    return None


def _resolve_saved_path(saved_path: str) -> Path | None:
    candidate = Path(saved_path)
    if candidate.is_file():
        return candidate

    parts = candidate.parts
    if "Dissertation" in parts:
        relative = Path(*parts[parts.index("Dissertation") + 1 :])
        relative_suffix = relative.as_posix()

        for local_candidate in PROJECT_ROOT.rglob(relative.name):
            if local_candidate.as_posix().endswith(relative_suffix):
                return local_candidate

    for local_candidate in PROJECT_ROOT.rglob(candidate.name):
        if local_candidate.name == candidate.name:
            return local_candidate

    return None


def _find_data_yaml(model_dir: Path) -> Path | None:
    local_args = model_dir / "args.yaml"
    saved_data_path = _read_saved_data_path(local_args)
    if saved_data_path:
        resolved = _resolve_saved_path(saved_data_path)
        if resolved is not None:
            return resolved

    for fallback in (
        model_dir.parent / "data.yaml",
        model_dir.parent.parent / "data.yaml",
    ):
        if fallback.is_file():
            return fallback

    return None


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0

    return sum(1 for item in directory.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)


def _evaluate_model(model_path: Path, data_yaml: Path) -> dict[str, object]:
    model = YOLO(str(model_path))
    last_error: Exception | None = None

    for split in ("test", "val"):
        try:
            metrics = model.val(
                data=str(data_yaml),
                split=split,
                imgsz=1280,
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
                "status": "ok",
                "split": split,
                "images": _count_images(data_yaml.parent / "images" / split),
                "precision": precision,
                "recall": recall,
                "map50": map50,
                "map5095": map5095,
                "f1": f1,
                "error": "",
            }
        except Exception as exc:
            last_error = exc

    return {
        "status": "error",
        "split": "",
        "images": 0,
        "precision": float("nan"),
        "recall": float("nan"),
        "map50": float("nan"),
        "map5095": float("nan"),
        "f1": float("nan"),
        "error": str(last_error) if last_error is not None else "Unknown evaluation error",
    }


def main() -> int:
    model_dirs = _discover_model_dirs()
    if not model_dirs:
        print("No matching model directories found.")
        return 0

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for model_dir in model_dirs:
        weight_path = model_dir / "weights" / "best.pt"
        data_yaml = _find_data_yaml(model_dir)

        if not weight_path.is_file():
            rows.append(
                {
                    "model_dir": str(model_dir),
                    "weights": str(weight_path),
                    "data_yaml": "",
                    "status": "missing_weights",
                    "split": "",
                    "images": 0,
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "map50": float("nan"),
                    "map5095": float("nan"),
                    "f1": float("nan"),
                    "error": "weights/best.pt not found",
                }
            )
            print(f"Skipped {model_dir} (missing weights)")
            continue

        if data_yaml is None or not data_yaml.is_file():
            rows.append(
                {
                    "model_dir": str(model_dir),
                    "weights": str(weight_path),
                    "data_yaml": "",
                    "status": "missing_data",
                    "split": "",
                    "images": 0,
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "map50": float("nan"),
                    "map5095": float("nan"),
                    "f1": float("nan"),
                    "error": "dataset YAML could not be resolved from args.yaml or local fallbacks",
                }
            )
            print(f"Skipped {model_dir} (missing dataset YAML)")
            continue

        print(f"Evaluating {weight_path} using {data_yaml}")
        metrics = _evaluate_model(weight_path, data_yaml)
        rows.append(
            {
                "model_dir": str(model_dir),
                "weights": str(weight_path),
                "data_yaml": str(data_yaml),
                **metrics,
            }
        )

    fieldnames = [
        "model_dir",
        "weights",
        "data_yaml",
        "status",
        "split",
        "images",
        "precision",
        "recall",
        "map50",
        "map5095",
        "f1",
        "error",
    ]

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {REPORT_PATH}")
    print(f"Processed {len(rows)} discovered model directories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())