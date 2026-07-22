import os
from PySide6.QtWidgets import QWidget, QLabel, QGridLayout, QPushButton, QListWidget
from PySide6.QtCore import Qt, Signal

from data_classes.ScrollableViewResults import ScrollableViewResults
from helpers import file_helpers


class ViewResultsPage(QWidget):
    switch_view = Signal()
    new_image_signal = Signal(str, str, str)

    def __init__(self, image_path, ai_result):
        """ Displays the results of the analysed image.
            Providing information on if a fault has been detected or not. """
        super().__init__()

        self.image_path = image_path
        self.ai_result = ai_result

        layout = QGridLayout()

        result_label = QLabel("")
        if len(self.ai_result[0].boxes.cls) > 0:
            result_label.setText("Result: Fault Detected")
        else:
            result_label.setText("No Fault Detected")

        layout.addWidget(result_label, 0, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)

        # Annotation List
        self.annotation_list = QListWidget()

        # Load Image and Annotations
        self.image_view = ScrollableViewResults(self.image_path, ai_result, self.annotation_list)
        self.annotation_list.itemClicked.connect(self.image_view.on_annotation_selected)

        layout.addWidget(self.image_view, 1, 0)
        layout.addWidget(self.annotation_list, 1, 1)

        zoom_in_button = QPushButton("Zoom In")
        zoom_out_button = QPushButton("Zoom Out")

        zoom_in_button.clicked.connect(self.image_view.zoom_in)
        zoom_out_button.clicked.connect(self.image_view.zoom_out)

        layout.addWidget(zoom_in_button, 2, 0)
        layout.addWidget(zoom_out_button, 2, 1)

        correct_button = QPushButton("AI Predicted Correctly")
        self.incorrect_button = QPushButton("AI Was Incorrect (Relabel and Add to Dataset)")

        correct_button.pressed.connect(self.close_view)
        layout.addWidget(correct_button, 3, 0)

        self.incorrect_button.pressed.connect(self.create_new_image)
        layout.addWidget(self.incorrect_button, 3, 1)

        self.setLayout(layout)

    def close_view(self):
        """ Switches the view back to the previous view. """
        self.switch_view.emit()

    def create_new_image(self):
        """ Creates a new image using the values provided from the results of the analysed image. """
        self.incorrect_button.setDisabled(True)
        new_img_folder = os.path.join("stored_training_images", "images", "raw")
        new_txt_folder = os.path.join("stored_training_images", "labels", "raw")
        safe_image_path, safe_image_file_name = file_helpers.create_valid_data_file_name(self.image_path,
                                                                                         new_img_folder,
                                                                                         new_txt_folder)

        file_helpers.move_file(self.image_path, safe_image_path)

        # Annotation file created
        file_name_without_ext = (os.path.splitext(safe_image_file_name)[0]) + ".txt"
        txt_destination_path = os.path.join(new_txt_folder, file_name_without_ext)

        file_helpers.create_file_empty_txt(txt_destination_path)
        new_txt = self.image_view.save_annotations()
        with open(txt_destination_path, 'w') as file:
            file.write(new_txt)

        # To differentiate AI annotations and manual annotations, the colour annotation file is created.
        # It contains duplicates of the contents of the annotations file, but can only be deleted from,
        # not appended to separate the manual changes from the original changes.
        colour_name = (os.path.splitext(safe_image_file_name)[0]) + "_colour.txt"
        annotation_colour_path = os.path.join(new_txt_folder, colour_name)

        file_helpers.create_file_empty_txt(annotation_colour_path)
        new_txt = self.image_view.save_annotations()
        with open(annotation_colour_path, 'w') as file:
            file.write(new_txt)

        self.new_image_signal.emit(safe_image_path, txt_destination_path, safe_image_file_name)
