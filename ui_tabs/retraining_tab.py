import random
import shutil
from pathlib import Path
import cv2
from datetime import datetime
from PySide6.QtWidgets import (
    QVBoxLayout, QWidget, QLabel, QPushButton, QMessageBox,
    QComboBox, QSpinBox, QHBoxLayout
)
from PySide6.QtCore import Qt

from stages.train_model import train_yolo
from data_classes.model_info import ModelInfo

TEMP_BLUR_DIR = Path("data/temp_blur_images")

def select_random_images_from_dir(img_dir: Path, num_to_select: int):
    img_list = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))
    if len(img_list) < num_to_select:
        return []
    return random.sample(img_list, num_to_select)

def blur_and_save_images(img_paths, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is not None:
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            dest_path = dest_dir / img_path.name
            cv2.imwrite(str(dest_path), blurred)

class RetrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.info_label = QLabel("Retrain model with blurred images")
        layout.addWidget(self.info_label)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Select model to retrain:"))
        self.model_combo = QComboBox()
        self.populate_model_combo()
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # Image count selection
        img_count_layout = QHBoxLayout()
        img_count_layout.addWidget(QLabel("Number of images to blur:"))
        self.img_count_spin = QSpinBox()
        self.img_count_spin.setRange(1, 100)
        self.img_count_spin.setValue(5)
        img_count_layout.addWidget(self.img_count_spin)
        layout.addLayout(img_count_layout)

        self.blur_button = QPushButton("Blur Random Images and Retrain")
        self.blur_button.clicked.connect(self.blur_and_retrain)
        layout.addWidget(self.blur_button)
        self.setLayout(layout)

    def populate_model_combo(self):
        self.model_combo.clear()
        models_path = Path("trained_models")
        if models_path.exists():
            for d in models_path.iterdir():
                if d.is_dir() and (d / "info.json").exists():
                    self.model_combo.addItem(d.name)

    def blur_and_retrain(self):
        img_dir = Path("data/images/train")
        num_to_select = self.img_count_spin.value()
        selected_model_name = self.model_combo.currentText()
        if not selected_model_name:
            QMessageBox.warning(self, "No Model Selected", "Please select a model to retrain.")
            return

        # Clean temp dir if exists
        if TEMP_BLUR_DIR.exists():
            shutil.rmtree(TEMP_BLUR_DIR)

        selected_imgs = select_random_images_from_dir(img_dir, num_to_select)
        if not selected_imgs:
            QMessageBox.warning(self, "No Images", f"Not enough images to select {num_to_select} from {img_dir}.")
            return

        blur_and_save_images(selected_imgs, TEMP_BLUR_DIR)
        print(f"Blurred and saved {num_to_select} images to {TEMP_BLUR_DIR}")

        #create a temp dataset yaml for retraining
        dataset_yaml_path = Path("data/dataset_yaml/retraining_temp.yaml")
        
        dataset_yaml_content = f"""names:
                                - '0'
                                nc: 1
                                train: {TEMP_BLUR_DIR.as_posix()}
                                val: data/images/val
                                """
        dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dataset_yaml_path, "w") as f:
            f.write(dataset_yaml_content)

        # Prepare retraining arguments
        data_yaml = "data/dataset_yaml/retraining_temp.yaml"
        model_dir = f"trained_models/retrained_blur_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        weights = f"trained_models/{selected_model_name}/weights/best.pt"
        img_size = "640"
        batch_size = "16"
        epochs = "10"
        flips = False

        # Load model_info from selected model
        info_path = Path("trained_models") / selected_model_name / "info.json"
        if not info_path.exists():
            QMessageBox.warning(self, "Model Info Missing", f"Could not find info.json for {selected_model_name}.")
            return
        with open(info_path, "r") as f:
            model_info_json = f.read()
        model_info = model_info_json
        training_start = datetime.now().isoformat()

        train_yolo(
            data_yaml=data_yaml,
            model_info=model_info,
            training_start=training_start,
            model_dir=model_dir,
            weights=weights,
            img_size=img_size,
            batch_size=batch_size,
            epochs=epochs,
            flips=flips
        )

        QMessageBox.information(self, "Retraining Complete", "Retraining with blurred images is complete.")
