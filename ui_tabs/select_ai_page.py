from PySide6.QtWidgets import QWidget, QGridLayout, QPushButton
from PySide6.QtCore import Qt, Signal

from data_classes.model_info import ModelInfo
from data_classes.list_model_widget import ListModelWidget


class SelectAiPage(QWidget):
    model_selected_signal = Signal(str, str)
    switch_view = Signal()

    def __init__(self):
        """ Allows the user to select an AI from the list of saved models. """
        super().__init__()
        self.layout = QGridLayout()

        self.cancel_button = QPushButton("Cancel")
        self.layout.addWidget(self.cancel_button, 0, 0, Qt.AlignmentFlag.AlignCenter)
        self.cancel_button.pressed.connect(self.close_view)

        self.config_list_widget = ListModelWidget()  # Use modular widget
        self.config_list_widget.setMinimumWidth(300)
        self.layout.addWidget(self.config_list_widget, 1, 0, Qt.AlignmentFlag.AlignCenter)
        self.config_list_widget.item_selected.connect(self.model_clicked)

        self.submit_button = QPushButton("Select")
        self.layout.addWidget(self.submit_button, 2, 0, Qt.AlignmentFlag.AlignCenter)
        self.submit_button.clicked.connect(self.submit_button_pressed)
        self.selected_model = ""

        self.setLayout(self.layout)

    def model_clicked(self, config_name, model: ModelInfo):
        """ Updates the selected model with the newly selected model. """
        self.selected_model = model

    def submit_button_pressed(self):
        """ Selected model is returned. """
        if self.selected_model != "":
            self.model_selected_signal.emit(self.selected_model.name, self.selected_model.get_best_pt_path())

    def close_view(self):
        """ Window is closed, without returning a model. """
        self.switch_view.emit()
