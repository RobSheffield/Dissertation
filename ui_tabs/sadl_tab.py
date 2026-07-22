from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _RunWorker(QThread):
    output = Signal(str)
    success = Signal(int)
    failure = Signal(str)

    def __init__(self, command: list[str], cwd: str):
        super().__init__()
        self.command = command
        self.cwd = cwd

    def run(self):
        try:
            process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                self.output.emit(line.rstrip())
            return_code = process.wait()
            if return_code == 0:
                self.success.emit(return_code)
            else:
                self.failure.emit(f"Process exited with code {return_code}.")
        except Exception as exc:  # pragma: no cover - defensive GUI boundary
            self.failure.emit(str(exc))


class _PathRow(QWidget):
    def __init__(self, select_folder: bool, default_value: str = "", placeholder_text: str = ""):
        super().__init__()
        self.select_folder = select_folder
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(default_value)
        if placeholder_text:
            self.edit.setPlaceholderText(placeholder_text)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        layout.addWidget(self.edit)
        layout.addWidget(browse)

    def _browse(self):
        if self.select_folder:
            value = QFileDialog.getExistingDirectory(self, "Select Folder", str(PROJECT_ROOT))
        else:
            value, _ = QFileDialog.getOpenFileName(self, "Select File", str(PROJECT_ROOT), "All Files (*)")
        if value:
            self.edit.setText(value)

    def text(self) -> str:
        return self.edit.text().strip()


class SADLTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker: _RunWorker | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("SADL TOOLKIT")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        augmentation_frame = QWidget()
        augmentation_layout = QFormLayout(augmentation_frame)
        self.horizontal_mirror_checkbox = QCheckBox("Horizontal mirroring (YOLO default)")
        self.horizontal_mirror_checkbox.setChecked(True)
        self.vertical_mirror_checkbox = QCheckBox("Vertical mirroring")
        self.vertical_mirror_checkbox.setChecked(False)
        self.rotate_90_checkbox = QCheckBox("Rotate 90°")
        self.rotate_180_checkbox = QCheckBox("Rotate 180°")
        self.rotate_270_checkbox = QCheckBox("Rotate 270°")
        augmentation_layout.addRow(QLabel("Training Augmentations"))
        augmentation_layout.addRow(self.horizontal_mirror_checkbox)
        augmentation_layout.addRow(self.vertical_mirror_checkbox)
        augmentation_layout.addRow(self.rotate_90_checkbox)
        augmentation_layout.addRow(self.rotate_180_checkbox)
        augmentation_layout.addRow(self.rotate_270_checkbox)
        layout.addWidget(augmentation_frame)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_single_tab(), "Single Run")
        self.tabs.addTab(self._build_across_tab(), "Across Folds")
        layout.addWidget(self.tabs)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

    def _build_single_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)
        self.single_model = _PathRow(False, placeholder_text="Select model file")
        self.single_train = _PathRow(True, placeholder_text="Select train images folder")
        self.single_val = _PathRow(True, placeholder_text="Select val images folder")
        self.single_train_labels = _PathRow(True, placeholder_text="Select train labels folder")
        self.single_val_labels = _PathRow(True, placeholder_text="Select val labels folder")
        self.single_image_size = QComboBox()
        self.single_image_size.addItems(["640", "1280"])
        self.single_image_size.setCurrentText("640")
        self.single_epochs = QSpinBox()
        self.single_epochs.setRange(1, 1000)
        self.single_epochs.setValue(50)
        self.single_bins = QSpinBox()
        self.single_bins.setRange(2, 100)
        self.single_bins.setValue(10)

        form.addRow("Model", self.single_model)
        form.addRow("Train images", self.single_train)
        form.addRow("Val images", self.single_val)
        form.addRow("Train labels", self.single_train_labels)
        form.addRow("Val labels", self.single_val_labels)
        form.addRow("Resolution size", self.single_image_size)
        form.addRow("Epochs used", self.single_epochs)
        form.addRow("Bins", self.single_bins)

        button = QPushButton("Run SADL")
        button.clicked.connect(self.run_single)
        form.addRow(button)
        return tab

    def _build_across_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)
        self.across_models_root = _PathRow(True, placeholder_text="Select models root")
        self.across_folds_root = _PathRow(True, placeholder_text="Select folds root")
        self.across_output = _PathRow(True, placeholder_text="Select output folder")
        self.across_metric = QComboBox()
        self.across_metric.addItems(["lsa", "dsa"])
        self.across_castings_root = _PathRow(True, placeholder_text="Select castings root")
        self.across_image_size = QComboBox()
        self.across_image_size.addItems(["640", "1280"])
        self.across_image_size.setCurrentText("640")
        self.across_epochs = QSpinBox()
        self.across_epochs.setRange(1, 1000)
        self.across_epochs.setValue(50)
        self.across_batch_size = QSpinBox()
        self.across_batch_size.setRange(1, 256)
        self.across_batch_size.setValue(8)
        self.across_var_threshold = QDoubleSpinBox()
        self.across_var_threshold.setDecimals(8)
        self.across_var_threshold.setRange(0.0, 1.0)
        self.across_var_threshold.setValue(1e-5)

        form.addRow("Models root", self.across_models_root)
        form.addRow("Folds root", self.across_folds_root)
        form.addRow("Output dir", self.across_output)
        form.addRow("Metric", self.across_metric)
        form.addRow("Castings root", self.across_castings_root)
        form.addRow("Resolution size", self.across_image_size)
        form.addRow("Epochs used", self.across_epochs)
        form.addRow("Batch size", self.across_batch_size)
        form.addRow("Variance threshold", self.across_var_threshold)

        button = QPushButton("Run Across Folds")
        button.clicked.connect(self.run_across)
        form.addRow(button)
        return tab

    def _start_worker(self, command: list[str]):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "SADL in Progress", "A SADL task is already running.")
            return

        self.log_output.clear()
        self.progress.setRange(0, 0)
        self.worker = _RunWorker(command, str(PROJECT_ROOT))
        self.worker.output.connect(self.log_output.append)
        self.worker.success.connect(self._on_success)
        self.worker.failure.connect(self._on_failure)
        self.worker.start()

    def _on_success(self, exit_code: int):
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.log_output.append(f"Finished successfully with exit code {exit_code}.")

    def _on_failure(self, message: str):
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.log_output.append(message)
        QMessageBox.critical(self, "SADL Error", message)

    def run_single(self):
        model_path = self.single_model.text() or str(PROJECT_ROOT / "final_datas" / "strat_10_unbiased_models" / "fold_1" / "weights" / "best.pt")
        train_path = self.single_train.text() or str(PROJECT_ROOT / "final_datas" / "Folds_strat_10" / "fold_1" / "images" / "train")
        val_path = self.single_val.text() or str(PROJECT_ROOT / "final_datas" / "Folds_strat_10" / "fold_1" / "images" / "val")
        train_labels_path = self.single_train_labels.text() or str(PROJECT_ROOT / "final_datas" / "Folds_strat_10" / "fold_1" / "labels" / "train")
        val_labels_path = self.single_val_labels.text() or str(PROJECT_ROOT / "final_datas" / "Folds_strat_10" / "fold_1" / "labels" / "val")
        command = [
            sys.executable,
            "-m",
            "testing.SADL.run_SADL",
            "--model-path",
            model_path,
            "--train-path",
            train_path,
            "--val-path",
            val_path,
            "--train-labels-path",
            train_labels_path,
            "--val-labels-path",
            val_labels_path,
            "--image-size",
            self.single_image_size.currentText(),
            "--epochs",
            str(self.single_epochs.value()),
            "--bin-amounts",
            str(self.single_bins.value()),
            "--horizontal-mirror" if self.horizontal_mirror_checkbox.isChecked() else "--no-horizontal-mirror",
            "--vertical-mirror" if self.vertical_mirror_checkbox.isChecked() else "",
            "--rotate-90" if self.rotate_90_checkbox.isChecked() else "",
            "--rotate-180" if self.rotate_180_checkbox.isChecked() else "",
            "--rotate-270" if self.rotate_270_checkbox.isChecked() else "",
        ]
        self._start_worker([arg for arg in command if arg != ""])

    def run_across(self):
        models_root = self.across_models_root.text() or str(PROJECT_ROOT / "final_datas" / "strat_10_unbiased_models")
        folds_root = self.across_folds_root.text() or str(PROJECT_ROOT / "final_datas" / "Folds_strat_10")
        output_dir = self.across_output.text() or str(PROJECT_ROOT / "final_datas" / "sadl_across_outputs")
        castings_root = self.across_castings_root.text() or str(PROJECT_ROOT / "Castings")
        command = [
            sys.executable,
            "-m",
            "testing.SADL.run_sadl_across",
            "--models-root",
            models_root,
            "--folds-root",
            folds_root,
            "--output-dir",
            output_dir,
            "--image-size",
            self.across_image_size.currentText(),
            "--epochs",
            str(self.across_epochs.value()),
            "--batch-size",
            str(self.across_batch_size.value()),
            "--var-threshold",
            str(self.across_var_threshold.value()),
            "--metric",
            self.across_metric.currentText(),
            "--castings-root",
            castings_root,
            "--horizontal-mirror" if self.horizontal_mirror_checkbox.isChecked() else "--no-horizontal-mirror",
            "--vertical-mirror" if self.vertical_mirror_checkbox.isChecked() else "",
            "--rotate-90" if self.rotate_90_checkbox.isChecked() else "",
            "--rotate-180" if self.rotate_180_checkbox.isChecked() else "",
            "--rotate-270" if self.rotate_270_checkbox.isChecked() else "",
        ]
        self._start_worker([arg for arg in command if arg != ""])
