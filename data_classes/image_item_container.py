from PySide6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QFrame)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import os


class ImageItemContainer(QWidget):
    """Displays the image and its file name in a Widget.
    This is designed to be used in a grid to allow the user to click on an image to open a more detailed view."""
    def __init__(self, image_dir, img_file, open_detail_view):
        super().__init__()

        self.image_dir = image_dir
        self.img_file = img_file
        self.open_detail_view = open_detail_view

        img_path = os.path.join(image_dir, img_file)

        # Create a container widget for image + label
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignCenter)

        # Thumbnail
        self.thumb_label = QLabel()
        self.thumb_label.setPixmap(QPixmap(img_path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.thumb_label.setFixedSize(110, 110)  # Keeps all boxes the same size
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setFrameStyle(QFrame.Box)  # Adds a border for consistency

        # Image name
        self.name_label = QLabel(img_file)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setFixedWidth(110)  # Keeps labels aligned

        # Add to layout
        self.layout.addWidget(self.thumb_label)
        self.layout.addWidget(self.name_label)
        self.setLayout(self.layout)

        self.thumb_label.mousePressEvent = lambda e, path=img_path, file_name=img_file: open_detail_view(path,
                                                                                                         file_name)

    def get_label_text(self) -> str:
        """Returns the label text, which is the file name."""
        return self.name_label.text()
