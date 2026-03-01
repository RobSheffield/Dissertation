import os
import signal
import sys
import ctypes
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PySide6.QtGui import QIcon

from ui_tabs.analyse_image_tab import AnalyseImageTab
from ui_tabs.dataset_config_tab import DatasetConfigTab
from ui_tabs.documentation_viewer import DocumentationViewerTab
from ui_tabs.train_ai_tab import TrainAiTab
from ui_tabs.view_models_tab import ViewModelsTab
from ui_tabs.testing_tab import TestingTab
from ui_tabs.retraining_tab import RetrainingTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("X-Ray Image Analysis")
        self.setGeometry(200, 200, 800, 400)

        self.analyse_image_tab = AnalyseImageTab()
        self.view_models_tab = ViewModelsTab()
        self.train_ai_tab = TrainAiTab()
        self.training_data_tab = DatasetConfigTab()
        self.documentation_tab = DocumentationViewerTab()
        self.testing_tab = TestingTab()
        self.retraining_tab = RetrainingTab()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.analyse_image_tab, "Analyse Image")
        self.tabs.addTab(self.view_models_tab, "View Models")
        self.tabs.addTab(self.train_ai_tab, "Train AI")
        self.tabs.addTab(self.training_data_tab, "Training Data")
        self.tabs.addTab(self.documentation_tab, "Documentation Viewer")
        self.tabs.addTab(self.testing_tab, "Model Testing")
        self.tabs.addTab(self.retraining_tab, "Retraining")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.analyse_image_tab.new_image_signal.connect(self.load_new_image)

        # central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # styling
        self.apply_styles()

    def on_tab_changed(self, index):
        if index == 1:
            self.view_models_tab.update_models()
        elif index == 2:
            self.train_ai_tab.load_dataset_configs()

    def resizeEvent(self, event):
        """Adjust the column count dynamically when the window resizes."""
        new_columns = max(1, self.width() // 200)  # Adjust column width threshold (200px per column)
        self.training_data_tab.set_column_count(new_columns)
        return super().resizeEvent(event)

    def apply_styles(self):
        """Load QSS stylesheet and apply it to the app."""
        style_path = os.path.join("helpers", "style.qss")
        with open(style_path, "r") as file:
            self.setStyleSheet(file.read())

    def load_new_image(self, img_path, txt_path, file_name):
        self.training_data_tab.add_image(safe_image_path=img_path,
                                         txt_destination_path=txt_path,
                                         safe_image_file_name=file_name)

        self.tabs.setCurrentWidget(self.training_data_tab)

    def closeEvent(self, event):
        """Ensure threads are stopped before closing."""
        print("Closing application, cleaning up threads...")
        if hasattr(self.testing_tab, 'test_worker') and self.testing_tab.test_worker.isRunning():
            self.testing_tab.test_worker.terminate()
            self.testing_tab.test_worker.wait()
        event.accept()


# TODO: Need to fix functionality for closing window. Ensure all threads terminate correctly.
def handle_sigint(signal_received, frame):
    """ Handle Ctrl+C """
    print("Ctrl+C detected, exiting...")
    app.quit()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)  # CTRL C exit condition
    os.environ["QT_IMAGEIO_MAXALLOC"] = str(1024 * 1024 * 1024 * 3)  # 1 GB
    app = QApplication(sys.argv)

    # Logo Setting
    app.setWindowIcon(QIcon("Logo.ico"))
    my_app_id = 'amrc.x-ray-image-analysis'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_app_id)

    window = MainWindow()
    window.showNormal()
    sys.exit(app.exec())
