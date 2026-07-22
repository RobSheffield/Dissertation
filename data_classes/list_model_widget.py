from PySide6.QtWidgets import QWidget, QListWidget, QVBoxLayout
from PySide6.QtCore import Signal
import os

from data_classes.model_info import ModelInfo
from helpers import file_helpers


class ListModelWidget(QWidget):
    """List Widget to Select an Model. It utilises ModelInfo to load information about each model."""

    item_selected = Signal(str, ModelInfo)

    def __init__(self):
        super().__init__()

        self.list_widget = QListWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.list_widget)
        self.setLayout(layout)

        self.load_configs()
        self.populate_list()

        self.list_widget.currentItemChanged.connect(self.on_item_selected)

    def load_configs(self):
        """Reads JSON files into ModelInfo objects."""
        self.model_info_json = {}

        models_dir = os.path.abspath("trained_models")
        if not os.path.exists(models_dir):
            print("ERROR: TRAINED_MODELS_FOLDER_DOES_NOT_EXIST")
            return

        for folder in os.listdir(models_dir):
            if os.path.isdir(os.path.join(models_dir, folder)):
                folder_path = os.path.join(models_dir, folder)
                info_path = os.path.join(folder_path, "info.json")
                try:
                    self.model_info_json[folder] = ModelInfo.fromPath(info_path)
                except FileNotFoundError:
                    print(f"{folder} model is not valid, removing")
                    file_helpers.delete_folder(folder_path)

    def populate_list(self):
        """Populates the list with config names."""
        self.list_widget.clear()
        for _, model_data in self.model_info_json.items():
            display_name = model_data.name + " - " + model_data.date_time_trained
            self.list_widget.insertItem(0, display_name)

    def update_list(self):
        """Reloads call configs"""
        self.load_configs()
        self.populate_list()

    def on_item_selected(self):
        """Handles item selection and emits signal."""
        if self.list_widget.currentItem() is not None:
            selected_name = self.list_widget.currentItem().text()

            name_data = None
            for _, data in self.model_info_json.items():
                if (data.name + " - " + data.date_time_trained) == selected_name:
                    name_data = data
                    break

            if name_data is not None:
                self.item_selected.emit(selected_name, name_data)
