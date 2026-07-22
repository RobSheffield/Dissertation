from PySide6.QtWidgets import (QWidget, QGridLayout, QPushButton)
from PySide6.QtCore import Signal

from data_classes.image_item_container import ImageItemContainer


class EditConfigItemContainer(QWidget):
    """Adds additional functionality to a provided ImageItemContainer,
    related to adding and removing from Configuration Files."""
    item_added = Signal(str, QWidget)
    item_removed = Signal(str, QWidget)

    def __init__(self, image_item_container: ImageItemContainer, is_in_current_config):
        super().__init__()
        self.image_item_container = image_item_container
        self.layout = QGridLayout()
        self.is_in_current_config = is_in_current_config

        # Buttons
        self.add_button = QPushButton("Add")
        self.add_button.setStyleSheet("background-color: green; color: white;")
        self.add_button.pressed.connect(self.toggle_visible)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setStyleSheet("background-color: red; color: white;")
        self.remove_button.pressed.connect(self.toggle_visible)

        if self.is_in_current_config:
            self.add_button.setHidden(True)
        else:
            self.remove_button.setHidden(True)

        self.layout.addWidget(self.image_item_container, 0, 0)
        self.layout.addWidget(self.add_button, 1, 0)
        self.layout.addWidget(self.remove_button, 1, 0)

        self.setLayout(self.layout)

    def toggle_visible(self):
        """Changes which button, add or remove, is visible. Also updates internal value of "is_in_current_config."""
        self.add_button.setVisible(not self.add_button.isVisible())
        self.remove_button.setVisible(not self.remove_button.isVisible())

        if self.is_in_current_config:
            self.item_removed.emit(self.image_item_container.get_label_text, self)
        else:
            self.item_added.emit(self.image_item_container.get_label_text, self)

        self.is_in_current_config = not self.is_in_current_config

    def to_string(self) -> str:
        """Returns image name."""
        return self.image_item_container.img_file

    def save_to_config(self) -> str:
        """Returns image name if it exists in the current config"""
        if self.is_in_current_config:
            return self.to_string() + "\n"
        else:
            return ""

    def get_label_text(self) -> str:
        """Returns the label text"""
        return self.image_item_container.get_label_text()
