from typing import List
from PySide6.QtWidgets import (QWidget, QGridLayout,
                               QScrollArea, QStackedLayout, QPushButton,
                               QLineEdit)
from PySide6.QtCore import Signal
from data_classes.image_item_container import ImageItemContainer
from helpers import file_helpers


class ViewDataset(QWidget):
    add_image_signal = Signal(str)

    def __init__(self, image_item_containers: List[ImageItemContainer], editable=True, annotation_dir=None,
                 dataset_file_path=None, column_count=3):
        """ View to browse and edit YOLO dataset """

        super().__init__()
        self.stacked_layout = QStackedLayout()

        # Init values
        self.annotation_dir = annotation_dir
        self.list_of_item_containers = image_item_containers
        self.column_count = column_count

        # Dataset filtering
        self.dataset_file_path = dataset_file_path

        # Scrollable Thumbnail Grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)  # Ensures the content resizes properly
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.scroll_area.setWidget(self.grid_container)

        # Search Function
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search...")
        self.search_bar.textChanged.connect(self.filter_images)

        # Buttons
        self.add_new_image_button = QPushButton("Add New Image")
        self.refresh_button = QPushButton("Refresh Images")
        self.add_new_image_button.clicked.connect(self.add_image)
        self.refresh_button.clicked.connect(self.refresh)

        # Main Layout
        self.images_grid_layout = QGridLayout()
        self.images_grid_layout.addWidget(self.search_bar, 0, 0)
        self.images_grid_layout.addWidget(self.add_new_image_button, 0, 1)
        self.images_grid_layout.addWidget(self.refresh_button, 0, 2)
        self.images_grid_layout.addWidget(self.scroll_area, 1, 0, 1, 3)
        self.images_grid_layout_wrapper = QWidget()
        self.images_grid_layout_wrapper.setLayout(self.images_grid_layout)

        self.setLayout(self.stacked_layout)

        self.stacked_layout.addWidget(self.images_grid_layout_wrapper)
        self.stacked_layout.setCurrentIndex(0)

        # Checks if config is able to be edited
        if not editable:
            self.add_new_image_button.setVisible(False)

        # If a config has been provided, it is loaded.
        if self.dataset_file_path is not None:
            self.all_containers = self.list_of_item_containers
            self.filter_images_for_config()
        self.filter_images()

    def filter_images_for_config(self):
        """ If a config has been provided, it is loaded and used to filter the loaded images. """
        if self.dataset_file_path is not None:
            self.selected_files = file_helpers.read_config_file(self.dataset_file_path)
            self.filtered_list = [
                img for img in self.all_containers
                if any(self.compare_name_to_search(img, filename)
                       for filename in self.selected_files)
            ]

            self.list_of_item_containers = self.filtered_list

    def filter_images(self):
        """Filters the images based on the contents of the search box"""
        self.filtered_list = []

        self.filtered_list = [img for img in self.list_of_item_containers
                              if self.compare_name_to_search(img, self.search_bar.text())]
        self.update_grid()

    def compare_name_to_search(self, img, search_string):
        """Check if the last item is a QLabel and contains the provided string."""
        img_label = img.get_label_text()
        return search_string.lower() in img_label.lower()  # Check contains the search string

    def set_column_count(self, count):
        """Update column count and refresh the grid."""
        if count != self.column_count:
            self.column_count = max(1, count)
            self.update_grid()

    def update_grid(self):
        """Clear and repopulate the grid with images based on current column count."""
        # Clear previous items
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)

        # Re-add images based on new column count
        for index, widget in enumerate(self.filtered_list):
            row = index // self.column_count
            col = index % self.column_count
            self.grid_layout.addWidget(widget, row, col)

    def add_image(self):
        """ Emits the add image signal """
        self.add_image_signal.emit(self.dataset_file_path)

    def refresh(self):
        """ Refreshes the images on the page. """
        self.search_bar.setText("")
        if self.dataset_file_path is not None:
            self.filter_images_for_config()
        self.filter_images()

    def update_images(self, list_of_item_containers):
        """ Updates the images on the page with the provided images. """
        self.all_containers = list_of_item_containers
        self.filter_images_for_config()
        self.filter_images()
