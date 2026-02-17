import os
import cv2
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QGridLayout, 
                               QPushButton, QComboBox, QFrame, QMessageBox, 
                               QProgressBar, QTextEdit, QDialog, QHBoxLayout,
                               QSpinBox, QTabWidget, QDoubleSpinBox, QListWidget,
                               QScrollArea)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from data_classes.model_info import ModelInfo
from stages.test_model import TestModelStage
from ui_tabs.graphing import MetamorphicTestGraphing, GraphingCanvas
import numpy as np


class ResultsWindow(QDialog):
    """Window to display metamorphic testing results with visual graphs."""
    def __init__(self, model_info, num_test_images, image_results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Metamorphic Testing Analysis")
        self.resize(1400, 700)
        self.image_results = image_results or {}
        
        layout = QVBoxLayout(self)
        
        main_content = QHBoxLayout()
        
        # Sidebar with image results list
        sidebar = QVBoxLayout()
        sidebar_label = QLabel("Test Results")
        sidebar_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        sidebar.addWidget(sidebar_label)
        
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.on_result_selected)
        
        if self.image_results:
            for key in sorted(self.image_results.keys()):
                self.results_list.addItem(str(key))
        else:
            self.results_list.addItem("No results available")
        
        sidebar.addWidget(self.results_list)
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setMaximumWidth(300)
        main_content.addWidget(sidebar_widget)
        
        # Main results area
        right_panel = QVBoxLayout()
        
        # Image display area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.image_display = QLabel("Select a test result from the list to view the comparison.")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("color: #777; font-size: 14px;")
        scroll.setWidget(self.image_display)
        right_panel.addWidget(scroll, 3)
        
        # Graph area
        self.graph_tabs = QTabWidget()
        
        # Text results tab
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("font-family: 'Consolas', monospace; font-size: 13px;")
        
        raw_results = getattr(model_info, 'metamorphic_test_result', 'No results found')
        content = f"METAMORPHIC TEST RESULTS: {getattr(model_info, 'folder_name', 'N/A')}\n"
        content += f"Tested on {num_test_images} random samples.\n"
        content += "=" * 50 + "\n\n"
        content += raw_results.replace(" | ", "\n")
        self.results_text.setText(content)
        self.graph_tabs.addTab(self.results_text, "Text Results")
        
        # Graphs tab
        graph_widget = QWidget()
        self.graph_layout = QVBoxLayout(graph_widget)
        self.display_result_graphs(raw_results)
        self.graph_tabs.addTab(graph_widget, "Graphs")
        
        right_panel.addWidget(self.graph_tabs, 2)
        
        main_content.addLayout(right_panel, 3)
        layout.addLayout(main_content)
        
        close_btn = QPushButton("Close Analysis")
        close_btn.setFixedHeight(40)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        # Auto-select first result
        if self.image_results:
            self.results_list.setCurrentRow(0)
            self.on_result_selected(self.results_list.item(0))
    
    def on_result_selected(self, item):
        """Load and display selected visual comparison image."""
        if not item or item.text() == "No results available":
            return
        
        result_key = item.text()
        img_bgr = self.image_results.get(result_key)
        
        if img_bgr is None:
            self.image_display.setText(f"No image data for: {result_key}")
            return
        
        # Convert BGR numpy array to QPixmap
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit the display area while keeping aspect ratio
        scaled = pixmap.scaledToWidth(
            min(960, pixmap.width()), Qt.SmoothTransformation
        )
        self.image_display.setPixmap(scaled)
    
    def display_result_graphs(self, raw_results):
        """Parse result data and display graphs."""
        try:
            perturbation_data = {}
            standard_results = {}
            
            for item in raw_results.split(" | "):
                if ":" in item:
                    full_name, acc_str = item.split(":")
                    acc_val = float(acc_str.strip().replace('%', ''))
                    
                    if full_name.endswith(")"):
                        parts = full_name.rsplit("(", 1)
                        if len(parts) > 1:
                            base_name = parts[0].strip()
                            level_str = parts[1].replace(")", "").strip()
                            try:
                                level = float(level_str)
                                if base_name not in perturbation_data:
                                    perturbation_data[base_name] = []
                                perturbation_data[base_name].append((level, acc_val))
                                continue
                            except ValueError:
                                pass
                    
                    standard_results[full_name.strip()] = [acc_val]
            
            if perturbation_data:
                trend_canvas = GraphingCanvas()
                trend_fig = MetamorphicTestGraphing.plot_perturbation_impact(perturbation_data)
                trend_canvas.display_figure(trend_fig)
                self.graph_layout.addWidget(trend_canvas)
            
            if standard_results:
                comp_canvas = GraphingCanvas()
                comp_fig = MetamorphicTestGraphing.plot_relation_comparison(standard_results)
                comp_canvas.display_figure(comp_fig)
                self.graph_layout.addWidget(comp_canvas)
        except Exception as e:
            print(f"Graphing display error: {e}")


class TestingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        title = QLabel("METAMORPHIC RELATION TESTING")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Top Frame for Global Settings (Model & Data)
        global_frame = QFrame()
        global_grid = QGridLayout(global_frame)
        self.model_combo = QComboBox()
        self.data_combo = QComboBox()
        self.img_count_label = QLabel("0")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Even Selection", "Dynamic Selection"])
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 1000)
        self.sample_spin.setValue(10)

        global_grid.addWidget(QLabel("Target Model:"), 0, 0)
        global_grid.addWidget(self.model_combo, 0, 1)
        global_grid.addWidget(QLabel("Image Source:"), 0, 2)
        global_grid.addWidget(self.data_combo, 0, 3)
        global_grid.addWidget(QLabel("Strategy:"), 1, 0)
        global_grid.addWidget(self.strategy_combo, 1, 1)
        global_grid.addWidget(QLabel("Samples:"), 1, 2)
        global_grid.addWidget(self.sample_spin, 1, 3)
        global_grid.addWidget(QLabel("Available:"), 1, 4)
        global_grid.addWidget(self.img_count_label, 1, 5)
        layout.addWidget(global_frame)

        # Tab Widget for Testing Modes
        self.test_tabs = QTabWidget()
        
        # Tab 1: Quick Batch
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)
        batch_info = QLabel("Runs a standard suite of MR tests with predefined values.\n"
                           "Includes: Rotations, Mirrors, Noise, Blur, Gamma, etc.")
        batch_info.setStyleSheet("color: #777; font-style: italic;")
        batch_layout.addWidget(batch_info)
        self.start_batch_btn = QPushButton("START STANDARD BATCH TEST")
        self.start_batch_btn.setFixedHeight(40)
        self.start_batch_btn.clicked.connect(lambda: self.start_testing(mode="batch"))
        batch_layout.addStretch()
        batch_layout.addWidget(self.start_batch_btn)
        self.test_tabs.addTab(batch_tab, "Standard Batch")

        # Tab 2: Individual MR
        adv_tab = QWidget()
        adv_grid = QGridLayout(adv_tab)
        
        self.mr_type_combo = QComboBox()
        # Mapping names to the module functions
        self.mr_options = {
            "Gaussian Noise": "noise_addition_gaussian",
            "Salt & Pepper": "noise_addition_salt_and_pepper",
            "Gamma Correction": "gamma_correction",
            "Brightness": "brightness_adjustment",
            "Contrast": "contrast_adjustment",
            "Blur": "blur"
        }
        self.mr_type_combo.addItems(list(self.mr_options.keys()))
        
        self.start_val = QDoubleSpinBox()
        self.start_val.setRange(-255, 255)
        self.end_val = QDoubleSpinBox()
        self.end_val.setRange(-255, 255)
        self.end_val.setValue(1.0)
        self.step_val = QDoubleSpinBox()
        self.step_val.setRange(0.001, 100)
        self.step_val.setValue(0.1)

        adv_grid.addWidget(QLabel("Select MR:"), 0, 0)
        adv_grid.addWidget(self.mr_type_combo, 0, 1, 1, 2)
        adv_grid.addWidget(QLabel("Start Value:"), 1, 0)
        adv_grid.addWidget(self.start_val, 1, 1)
        adv_grid.addWidget(QLabel("End Value:"), 1, 2)
        adv_grid.addWidget(self.end_val, 1, 3)
        adv_grid.addWidget(QLabel("Step:"), 1, 4)
        adv_grid.addWidget(self.step_val, 1, 5)
        
        self.start_adv_btn = QPushButton("RUN CUSTOM PARAMETER TEST")
        self.start_adv_btn.setFixedHeight(40)
        self.start_adv_btn.clicked.connect(lambda: self.start_testing(mode="custom"))
        adv_grid.addWidget(self.start_adv_btn, 2, 0, 1, 6)
        
        self.test_tabs.addTab(adv_tab, "Individual MR Analysis")
        layout.addWidget(self.test_tabs)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Logs will appear here...")
        layout.addWidget(self.log_output)
        
        self.overall_progress = QProgressBar()
        layout.addWidget(self.overall_progress)
        
        # Connect signal
        self.data_combo.currentIndexChanged.connect(self.update_image_count)
        self.mr_type_combo.currentTextChanged.connect(self.update_adv_defaults)
        
        self.refresh_lists()
        self.update_adv_defaults(self.mr_type_combo.currentText())

    def update_adv_defaults(self, mr_name):
        """Sets sensible defaults for the selected MR."""
        if mr_name == "Salt & Pepper":
            self.start_val.setValue(0.001)
            self.end_val.setValue(0.02)
            self.step_val.setValue(0.001)
            self.start_val.setSingleStep(0.001)
        elif mr_name == "Gaussian Noise":
            self.start_val.setValue(1)
            self.end_val.setValue(16)
            self.step_val.setValue(3)
        elif mr_name == "Blur":
            self.start_val.setValue(3)
            self.end_val.setValue(21)
            self.step_val.setValue(4)
        elif mr_name == "Brightness":
            self.start_val.setValue(-100)
            self.end_val.setValue(100)
            self.step_val.setValue(50)
        else:
            self.start_val.setValue(0.5)
            self.end_val.setValue(2.0)
            self.step_val.setValue(0.5)

    def update_image_count(self):
        index = self.data_combo.currentIndex()
        count = 0
        if index <= 0:
            path = os.path.join("stored_training_images", "images", "raw")
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        else:
            dataset_file = self.data_combo.currentText()
            path = os.path.join("stored_training_images", "datasets", dataset_file)
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        count = len([line for line in f.readlines() if line.strip()])
                except Exception:
                    count = 0
        self.img_count_label.setText(str(count))

    def refresh_lists(self):
        self.model_combo.clear()
        models_path = "trained_models"
        if os.path.exists(models_path):
            models = [d for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]
            self.model_combo.addItems(models)
        
        self.data_combo.clear()
        self.data_combo.addItem("All Raw Images")
        datasets_path = os.path.join("stored_training_images", "datasets")
        if os.path.exists(datasets_path):
            datasets = [f for f in os.listdir(datasets_path) if f.endswith('.txt')]
            self.data_combo.addItems(datasets)
        
        self.update_image_count()

    def start_testing(self, mode="batch"):
        if hasattr(self, 'test_worker') and self.test_worker.isRunning():
            QMessageBox.warning(self, "Test in Progress", "A test is already running. Please wait for it to finish.")
            return

        model_name = self.model_combo.currentText()
        if not model_name:
            QMessageBox.warning(self, "Selection Error", "Please select a model.")
            return

        # Disable UI elements
        self.start_batch_btn.setEnabled(False)
        self.start_adv_btn.setEnabled(False)
        self.test_tabs.setEnabled(False)

        model_dir = os.path.abspath(os.path.join("trained_models", model_name))
        model_json = os.path.join(model_dir, "info.json")
        model_info = ModelInfo.fromPath(model_json) if os.path.exists(model_json) else \
                     ModelInfo(model_name, "YOLOv5", "N/A", "N/A", "N/A", path=model_dir, folder_name=model_name)

        if self.data_combo.currentIndex() <= 0:
            img_p = os.path.join("stored_training_images", "images", "raw")
            lab_p = os.path.join("stored_training_images", "labels", "raw")
        else:
            dataset_name = self.data_combo.currentText()
            img_p = os.path.join("stored_training_images", "images", "raw")
            lab_p = os.path.join("stored_training_images", "datasets", dataset_name)

        custom_mrs = None
        if mode == "custom":
            mr_name = self.mr_type_combo.currentText()
            func_name = self.mr_options[mr_name]
            from testing.metamorphic_relations import metamorphic_relations
            func = getattr(metamorphic_relations, func_name)
            
            import numpy as np
            levels = list(np.arange(self.start_val.value(), self.end_val.value() + self.step_val.value(), self.step_val.value()))
            levels = [round(float(l), 4) for l in levels if l <= self.end_val.value() or np.isclose(l, self.end_val.value())]
            custom_mrs = [(mr_name, func, levels[:30])]

        self.log_output.clear()
        self.overall_progress.setValue(0)
        
        num_samples = self.sample_spin.value()
        strategy = self.strategy_combo.currentText()

        self.test_worker = TestModelStage(
            model_info, img_p, lab_p, "data/images", "trained_models",
            num_samples=num_samples, strategy=strategy, custom_mrs=custom_mrs
        )
        
        self.test_worker.model_testing_text_signal.connect(self.log_output.append)
        self.test_worker.model_testing_progress_bar_signal.connect(self.overall_progress.setValue)
        self.test_worker.finished.connect(self.on_test_finished)
        self.test_worker.start()

    def on_test_finished(self):
        """Re-enable UI and show results."""
        self.start_batch_btn.setEnabled(True)
        self.start_adv_btn.setEnabled(True)
        self.test_tabs.setEnabled(True)
        self.show_results()
        

    def show_results(self):
        """Displays the results window. Graphing is handled within ResultsWindow logic."""
        self.results_win = ResultsWindow(self.test_worker.model_info, self.sample_spin.value(), self.test_worker.image_results)
        self.results_win.exec()

    def runGradCAM(self):

        pass