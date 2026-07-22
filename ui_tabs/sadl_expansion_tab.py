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
    QDoubleSpinBox,
    QSpinBox,
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


class SADLExpansionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker: _RunWorker | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("SADL EXPANSION EXPERIMENT")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        form = QFormLayout()
        self.data_root = _PathRow(True, placeholder_text="Select data root (GDXray format)")
        self.output_path = _PathRow(True, placeholder_text="Select output folder")
        self.k_value = QSpinBox()
        self.k_value.setRange(2, 20)
        self.k_value.setValue(5)
        self.epochs_value = QSpinBox()
        self.epochs_value.setRange(1, 1000)
        self.epochs_value.setValue(150)
        self.resolution_size_value = QComboBox()
        self.resolution_size_value.addItems(["640", "1280"])
        self.resolution_size_value.setCurrentText("1280")
        self.initial_train_fraction = QDoubleSpinBox()
        self.initial_train_fraction.setDecimals(2)
        self.initial_train_fraction.setRange(0.01, 0.99)
        self.initial_train_fraction.setSingleStep(0.05)
        self.initial_train_fraction.setValue(0.50)
        self.expansion_step_fraction = QDoubleSpinBox()
        self.expansion_step_fraction.setDecimals(2)
        self.expansion_step_fraction.setRange(0.01, 0.99)
        self.expansion_step_fraction.setSingleStep(0.05)
        self.expansion_step_fraction.setValue(0.25)
        self.holdout_fraction = QDoubleSpinBox()
        self.holdout_fraction.setDecimals(2)
        self.holdout_fraction.setRange(0.00, 0.99)
        self.holdout_fraction.setSingleStep(0.05)
        self.holdout_fraction.setValue(0.25)
        self.image_wise_partitioning = QCheckBox("Use deterministic seed (42)")
        self.image_wise_partitioning.setChecked(False)

        form.addRow("Data root", self.data_root)
        form.addRow("Output root", self.output_path)
        form.addRow("Runs", self.k_value)
        form.addRow("Epochs used", self.epochs_value)
        form.addRow("Resolution size", self.resolution_size_value)
        form.addRow("Portion 1", self.initial_train_fraction)
        form.addRow("Portion 2", self.expansion_step_fraction)
        form.addRow("Portion 3", self.holdout_fraction)
        form.addRow(self.image_wise_partitioning)

        run_button = QPushButton("Run Expansion Experiment")
        run_button.clicked.connect(self.run_expansion)
        form.addRow(run_button)
        layout.addLayout(form)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Expansion experiment logs will appear here...")
        layout.addWidget(self.log_output)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

    def _start_worker(self, command: list[str]):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Expansion In Progress", "An expansion experiment is already running.")
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
        QMessageBox.critical(self, "Expansion Error", message)

    def run_expansion(self):
        data_root = self.data_root.text() or str(PROJECT_ROOT / "Castings")
        output_root = self._normalize_output_path()
        command = [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "SADL_guided_expansion.py"),
            "--castings-dir",
            data_root,
            "--output-root",
            output_root,
            "--epochs",
            str(self.epochs_value.value()),
            "--image-size",
            self.resolution_size_value.currentText(),
            "--n-runs",
            str(self.k_value.value()),
            "--portion-1",
            str(self.initial_train_fraction.value()),
            "--portion-2",
            str(self.expansion_step_fraction.value()),
            "--portion-3",
            str(self.holdout_fraction.value()),
        ]
        if self.image_wise_partitioning.isChecked():
            command.extend(["--seed", "42"])
        self._start_worker(command)

    def _normalize_output_path(self) -> str:
        selected = Path(self.output_path.text())
        if selected.exists():
            return str(selected)
        if selected.suffix:
            return str(selected.parent)
        return str(PROJECT_ROOT / "testing")
