import datetime
import gc
import os
import shutil
import stat
import subprocess
import time
from PySide6.QtCore import Signal, QThread
import sys
import torch
from helpers import file_helpers
from helpers.console_output import CaptureConsoleOutputThread
from helpers.k_fold import augment_training_set_with_selected_transforms
from helpers.model_weights import resolve_training_weights
import stages.model_training
from data_classes.model_info import ModelInfo
from stages.test_model import TestModelStage


class MainTrainPipeline(QThread):
    """
    Creates the training pipeline thread.
    It augments the data, trains a model and evaluates/tests the produced model.
    """
    pipeline_started = Signal(bool)
    pipeline_finished = Signal()
    data_augmentation_text = Signal(str)
    data_augmentation_progress_bar = Signal(int)
    model_training_text = Signal(str)
    model_training_progress_bar = Signal(int)
    model_testing_text = Signal(str)
    model_testing_progress_bar = Signal(int)

    def __init__(self, parent, model_info: ModelInfo, augmentation_options=None, image_size=640):
        QThread.__init__(self, parent)
        self._is_running = True
        self.model_info = model_info
        self.augmentation_options = augmentation_options or {}
        self.image_size = image_size
        self.is_cleaned_up = False
        self.data_dir = os.path.abspath("data")  # Path to the dataset directory
        self.train_data_dir = os.path.join(self.data_dir, "images")
        self.train_labels_dir = os.path.join(self.data_dir, "labels")

        stored_training_images_dir = os.path.abspath("stored_training_images")
        self.dataset_config_dir = os.path.join(stored_training_images_dir, "datasets")
        self.image_dir = os.path.join(stored_training_images_dir, "images", "raw")
        self.annotation_dir = os.path.join(stored_training_images_dir, "labels", "raw")
        self.model_dir = os.path.abspath("trained_models")
        runs_dir = os.path.join("runs", "detect")

        if os.path.exists(runs_dir):
            # Removes each file in the directory. If the file is access blocked, it changes the permissions to read only
            # then deletes it. This prevents previous processes locking resources when they should have exited.
            shutil.rmtree(
                runs_dir,
                onexc=lambda func, path, exc_info: (os.chmod(path, stat.S_IWRITE), func(path))
            )

    def run(self):
        self.pipeline_flow()

    def stop(self):
        self._is_running = False

    def pipeline_flow(self):
        """
        Defines the pipeline using flow. Each stage is executed sequentially.
        Using results from previous stages.
        """

        self.pipeline_started.emit(True)
        if self._is_running:
            self.prepare_data()
            result = self.train_model()

        if result and self._is_running:
            self.test_model(result)

        self.pipeline_finished.emit()

    def prepare_data(self):
        """Augments the data to improve the performance of the model training."""
        self.data_augmentation_text.emit("Resetting Data...")
        file_helpers.reset_training_data(self.train_data_dir, self.train_labels_dir)
        self.data_augmentation_progress_bar.emit(10)

        self.data_augmentation_text.emit("Loading Data...")
        all_img_path = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', '.PNG', '.JPG'))]
        self.data_augmentation_progress_bar.emit(20)

        if (self.model_info.dataset_config != "All Images"):
            self.data_augmentation_text.emit("Config Found, Loading...")
            config_path = os.path.join(self.dataset_config_dir, self.model_info.dataset_config + ".txt")
            selected_files = file_helpers.read_config_file(config_path)
            filtered_img_list = [
                img for img in all_img_path
                if any(img.lower() in filename.lower()
                       for filename in selected_files)
            ]
            all_img_path = filtered_img_list

        self.data_augmentation_progress_bar.emit(80)
        self.data_augmentation_text.emit("Initialising Training Values...")
        file_helpers.load_training_images(all_img_path, self.image_dir, self.annotation_dir, self.data_dir)

        rotation_options = self.augmentation_options.get("rotations", {})
        if any(rotation_options.values()):
            self.data_augmentation_text.emit("Applying selected training rotations...")
            augment_training_set_with_selected_transforms(
                os.path.join(self.data_dir, "images", "train"),
                os.path.join(self.data_dir, "labels", "train"),
                include_horizontal_flip=False,
                include_vertical_flip=False,
                include_rot90=rotation_options.get("rot90", False),
                include_rot180=rotation_options.get("rot180", False),
                include_rot270=rotation_options.get("rot270", False),
            )

        self.data_augmentation_progress_bar.emit(100)
        self.data_augmentation_text.emit("Data Loaded!")

    def train_model(self):
        """
        Uses the augmented data to create a trained model based on the parameters.
        """

        class_names = ["0"]  # List of class names, only 1 flaw, so only one class

        path = os.path.join("data", "dataset_yaml", "dataset.yaml")
        output_yaml = os.path.abspath(path)  # Path to save the YAML file

        if self.model_info.starting_model == "":
            weights = resolve_training_weights(self.model_info.model)
        else:
            weights = self.model_info.starting_model

        horizontal_mirror = self.augmentation_options.get("horizontal_mirror", True)
        vertical_mirror = self.augmentation_options.get("vertical_mirror", False)
        img_size = int(self.image_size)  # Image size for training
        batch_size = 8 if img_size >= 1280 else 16  # Batch size for training
        epochs = int(self.model_info.epoch)  # Number of epochs for training

        # Create YAML file for YOLO to use
        stages.model_training.create_yaml(self.data_dir, output_yaml, class_names)

        training_start = datetime.datetime.now()
        timestamp = training_start.strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(os.path.abspath("trained_models"), f"model_{timestamp}")

        train_yolo_path = os.path.join(os.path.abspath("stages"), "train_model.py")
        training_start = training_start.strftime("%Y-%m-%d__%H:%M:%S")
        command = [
            "pythonw", train_yolo_path,
            "--data_yaml", output_yaml,
            "--weights", weights,
            "--training_start", training_start,
            "--img_size", str(img_size),
            "--model_dir", model_dir,
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--model_info", str(self.model_info.to_json()),
            "--fliplr", str(0.5 if horizontal_mirror else 0.0),
            "--flipud", str(0.5 if vertical_mirror else 0.0),
        ]

        self.process = subprocess.Popen(command,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        bufsize=1,
                                        universal_newlines=True,
                                        encoding='utf-8',
                                        errors='replace')

        self.console_thread = CaptureConsoleOutputThread(self.process)
        self.console_thread.output_written.connect(self.model_training_text)
        self.console_thread.start()

        while self.process.poll() is None:
            if self._is_running is False:
                self.process.terminate()
                self.process.wait()
            else:
                time.sleep(100)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        self.cleanup()

        return model_dir

    def test_model(self, model_info_dir):
        """Tests the trained model against previous iterations, to produce a performance score."""
        model_info_path = os.path.join(model_info_dir, "info.json")

        model_info = ModelInfo.fromPath(model_info_path)

        self.testing_thread = TestModelStage(model_info, self.train_data_dir, self.annotation_dir,
                                             self.image_dir, self.model_dir)
        self.testing_thread.model_testing_text_signal.connect(self.model_testing_text)
        self.testing_thread.model_testing_progress_bar_signal.connect(self.model_testing_progress_bar)
        self.testing_thread.run()
        self.testing_thread.wait()

    def cleanup(self):
        if hasattr(self, 'console_thread'):
            self.console_thread.stop()
            self.console_thread.quit()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
