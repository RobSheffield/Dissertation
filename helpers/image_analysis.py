from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO


class ImageAnalysisWorker(QThread):
    """A worker class to analyse a provided image using YOLO."""
    finished_signal = Signal(object)

    def __init__(self, model_path, image_path):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path

    def run(self):
        model = YOLO(self.model_path)
        results = model(self.image_path)
        self.finished_signal.emit(results)
