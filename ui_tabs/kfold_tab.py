from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from helpers.model_weights import MODEL_WEIGHT_OPTIONS


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


class KFoldTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker: _RunWorker | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("K-FOLDS TESTING")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        form = QFormLayout()
        self.data_root = _PathRow(True, placeholder_text="Select data root (GDXray format)")
        self.output_path = _PathRow(True, placeholder_text="Select output folder or CSV")
        self.k_value = QSpinBox()
        self.k_value.setRange(2, 20)
        self.k_value.setValue(5)
        self.epochs_value = QSpinBox()
        self.epochs_value.setRange(1, 1000)
        self.epochs_value.setValue(150)
        self.batch_size_value = QSpinBox()
        self.batch_size_value.setRange(1, 256)
        self.batch_size_value.setValue(12)
        self.resolution_size_value = QComboBox()
        self.resolution_size_value.addItems(["640", "1280"])
        self.resolution_size_value.setCurrentText("640")
        self.image_wise_partitioning = QCheckBox("Use image-level partitioning")
        self.image_wise_partitioning.setChecked(False)
        self.weights_value = QComboBox()
        self.weights_value.addItems(list(MODEL_WEIGHT_OPTIONS.keys()))
        self.weights_value.setCurrentIndex(0)

        self.horizontal_mirror_checkbox = QCheckBox("Horizontal mirroring (YOLO default)")
        self.horizontal_mirror_checkbox.setChecked(True)
        self.vertical_mirror_checkbox = QCheckBox("Vertical mirroring")
        self.vertical_mirror_checkbox.setChecked(False)
        self.rotate_90_checkbox = QCheckBox("Rotate 90°")
        self.rotate_180_checkbox = QCheckBox("Rotate 180°")
        self.rotate_270_checkbox = QCheckBox("Rotate 270°")

        form.addRow("Data root", self.data_root)
        form.addRow("Output CSV", self.output_path)
        form.addRow("K folds", self.k_value)
        form.addRow("Epochs", self.epochs_value)
        form.addRow("Batch size", self.batch_size_value)
        form.addRow("Resolution size", self.resolution_size_value)
        form.addRow("Base model", self.weights_value)
        form.addRow(self.image_wise_partitioning)
        form.addRow(QLabel("Training Augmentations"))
        form.addRow(self.horizontal_mirror_checkbox)
        form.addRow(self.vertical_mirror_checkbox)
        form.addRow(self.rotate_90_checkbox)
        form.addRow(self.rotate_180_checkbox)
        form.addRow(self.rotate_270_checkbox)

        run_button = QPushButton("Run K-Folds Experiment")
        run_button.clicked.connect(self.run_kfold)
        form.addRow(run_button)
        layout.addLayout(form)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("K-fold experiment logs will appear here...")
        layout.addWidget(self.log_output)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

    def _start_worker(self, command: list[str]):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "K-Folds In Progress", "A K-fold experiment is already running.")
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
        QMessageBox.critical(self, "K-Folds Error", message)

    def run_kfold(self):
        data_root = self.data_root.text() or str(PROJECT_ROOT / "Castings")
        output_path = self.output_path.text() or str(PROJECT_ROOT / "experiments" / "lca_kfold_intervals_report.csv")
        command = [
            sys.executable,
            "-m",
            "testing.lca_kfold_intervals",
            "--data-root",
            data_root,
            "--out",
            output_path,
            "--k",
            str(self.k_value.value()),
            "--epochs",
            str(self.epochs_value.value()),
            "--batch-size",
            str(self.batch_size_value.value()),
            "--img-size",
            self.resolution_size_value.currentText(),
            "--weights",
            MODEL_WEIGHT_OPTIONS[self.weights_value.currentText()],
            "--apply-augmentations",
            "--horizontal-mirror" if self.horizontal_mirror_checkbox.isChecked() else "--no-horizontal-mirror",
            "--vertical-mirror" if self.vertical_mirror_checkbox.isChecked() else "",
            "--rotate-90" if self.rotate_90_checkbox.isChecked() else "",
            "--rotate-180" if self.rotate_180_checkbox.isChecked() else "",
            "--rotate-270" if self.rotate_270_checkbox.isChecked() else "",
            "--partition-mode",
            "image" if self.image_wise_partitioning.isChecked() else "folder",
        ]
        self._start_worker([arg for arg in command if arg != ""])

    def _normalize_output_path(self) -> str:
        selected = Path(self.output_path.text())
        if selected.suffix.lower() == ".csv":
            return str(selected)
        if selected.is_dir():
            return str(selected / "lca_kfold_intervals_report.csv")
        if selected.exists() and not selected.is_dir():
            return str(selected)
        return str(PROJECT_ROOT / "experiments" / "lca_kfold_intervals_report.csv")
