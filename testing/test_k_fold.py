import cv2
import numpy as np

from helpers import k_fold
from helpers.k_fold import _write_image_safely


def test_write_image_safely_creates_decodable_image(tmp_path):
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    output_path = tmp_path / "augmented.png"

    _write_image_safely(str(output_path), image)

    written_image = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    assert written_image is not None
    assert written_image.shape == image.shape
    assert not (tmp_path / "augmented.tmp.png").exists()


def test_train_all_can_group_training_output_by_fold(tmp_path, monkeypatch):
    folds_root = tmp_path / "folds"
    train_images = folds_root / "fold_1" / "images" / "train"
    train_images.mkdir(parents=True)
    (train_images / "sample.png").touch()
    captured = {}

    def fake_train_yolo(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(k_fold, "train_yolo", fake_train_yolo)

    output_root = tmp_path / "results"
    k_fold.train_all(
        str(folds_root),
        model_dir=str(output_root),
        epochs=1,
        group_outputs_by_fold=True,
    )

    assert captured["model_dir"] == str(output_root / "fold_1" / "train")


def test_map_outputs_validation_under_matching_fold(tmp_path, monkeypatch):
    test_root = tmp_path / "test"
    (test_root / "images").mkdir(parents=True)
    (test_root / "labels").mkdir()

    results_root = tmp_path / "results"
    weights_path = results_root / "fold_1" / "train" / "weights" / "best.pt"
    weights_path.parent.mkdir(parents=True)
    weights_path.touch()

    captured = {}

    class FakeBox:
        map50 = 0.5
        map = 0.25
        mp = 0.75
        mr = 0.6

    class FakeMetrics:
        box = FakeBox()

    class FakeModel:
        def __init__(self, model_path):
            captured["model_path"] = model_path

        def val(self, **kwargs):
            captured["validation_args"] = kwargs
            return FakeMetrics()

    monkeypatch.setattr(k_fold, "YOLO", FakeModel)

    report_path = k_fold.mAP_on_test_set(
        str(test_root),
        str(results_root),
        imgsz=640,
        validation_output_root=str(results_root),
    )

    assert captured["model_path"] == str(weights_path)
    assert captured["validation_args"]["project"] == str(results_root / "fold_1")
    assert captured["validation_args"]["name"] == "val"
    assert captured["validation_args"]["exist_ok"] is True
    assert report_path == str(results_root / "test_map_results.txt")
