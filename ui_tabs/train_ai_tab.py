import re
import threading
from PySide6.QtWidgets import (QWidget, QVBoxLayout,
                               QLabel, QGridLayout, QPushButton, QLineEdit, QComboBox,
                               QStackedLayout, QFrame, QMessageBox, QProgressBar, QTextEdit,
                               QCheckBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
from data_classes.model_info import ModelInfo
from stages.main_train_pipeline import MainTrainPipeline
from helpers import file_helpers
from helpers.model_weights import MODEL_WEIGHT_OPTIONS
import os

from ui_tabs.select_ai_page import SelectAiPage


class TrainAiTab(QWidget):
    def __init__(self):
        """ Provides the functionality to train an AI, and view an AI training in progress """
        super().__init__()
        self.train_in_progress = False
        data_dir = os.path.abspath("stored_training_images")
        self.dataset_config_dir = os.path.join(data_dir, "datasets")
        self.image_dir = os.path.join(data_dir, "images", "raw")
        self.image_count = "0"
        self.selected_starting_model = ""

        self.main_layout = QGridLayout()

        # Adds button to be able to switch to previous view when training is in progress
        switch_view_layout = QGridLayout()
        self.control_view_push_button = QPushButton()
        self.control_view_push_button.setText("Switch View")
        self.control_view_push_button.pressed.connect(self.switch_views)

        switch_view_layout.addWidget(QLabel("Start Train AI View"), 0, 0)
        switch_view_layout.addWidget(self.control_view_push_button, 0, 1)

        # Divider between elements
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        switch_view_layout.addWidget(separator, 1, 0, 1, 2)

        self.switch_view_layout_wrapper_widget = QWidget()
        self.switch_view_layout_wrapper_widget.setLayout(switch_view_layout)
        self.switch_view_layout_wrapper_widget.setHidden(True)

        # Provides the layout for the contents of the page, either start train AI,
        # select model to train from or view AI training progress.
        self.stacked_layout = QStackedLayout()

        self.main_layout.addWidget(self.switch_view_layout_wrapper_widget, 0, 0, 2, 2)
        self.main_layout.addLayout(self.stacked_layout, 2, 0, 1, 2)
        self.setLayout(self.main_layout)

        self.stacked_layout.addWidget(self.create_start_train_ai_gui())
        self.stacked_layout.addWidget(self.create_ai_train_gui())
        self.stacked_layout.addWidget(self.create_select_model())

        self.stacked_layout.setCurrentIndex(0)

    def create_start_train_ai_gui(self) -> QWidget:
        """ Creates the GUI for starting to training the AI. Separated for readability. """
        # Labels
        self.image_count_label = QLabel(self.image_count)
        self.selected_model_label = QLabel("None")

        # Comboboxes
        self.model_selected_combobox = QComboBox()
        self.model_selected_combobox.addItems(list(MODEL_WEIGHT_OPTIONS.keys()))
        self.model_selected_combobox.setCurrentIndex(0)

        self.resolution_size_combobox = QComboBox()
        self.resolution_size_combobox.addItems(["640", "1280"])
        self.resolution_size_combobox.setCurrentText("640")

        self.config_selector_combobox = QComboBox()
        self.load_dataset_configs()
        self.config_selector_combobox.currentIndexChanged.connect(self.combobox_index_changed)

        # Buttons
        self.select_model_button = QPushButton("Select Model")
        self.select_model_button.pressed.connect(self.select_model)

        self.reset_model_selected_button = QPushButton("Reset Model")
        self.reset_model_selected_button.setHidden(True)
        self.reset_model_selected_button.pressed.connect(self.reset_model_selection)

        self.train_button = QPushButton("Train")
        self.train_button.pressed.connect(self.start_ai_train)

        self.cancel_button = QPushButton("Cancel Training")
        self.cancel_button.pressed.connect(self.confirm_cancel_training)
        self.cancel_button.setHidden(True)

        # Line Edits
        self.ai_name_input = QLineEdit()
        self.ai_name_input.setText("New AI")

        self.epoch_num_input = QLineEdit()
        self.epoch_num_input.setPlaceholderText("Enter Number of Epoch")
        self.epoch_num_input.setValidator(QIntValidator())  # Only allows integers
        self.epoch_num_input.setText("50")

        # Layouts
        new_ai_model_layout = QGridLayout()
        start_train_ai_layout = QGridLayout()
        selected_model_layout = QGridLayout()

        selected_model_layout.addWidget(self.selected_model_label, 0, 0)
        selected_model_layout.addWidget(self.select_model_button, 0, 1)
        selected_model_layout.addWidget(self.reset_model_selected_button, 0, 1)

        selected_model_layout_wrapper_widget = QWidget()
        selected_model_layout_wrapper_widget.setLayout(selected_model_layout)

        new_ai_model_layout.addWidget(QLabel("Name:"), 0, 0)
        new_ai_model_layout.addWidget(self.ai_name_input, 0, 1)

        new_ai_model_layout.addWidget(QLabel("Selected Dataset"), 1, 0)
        new_ai_model_layout.addWidget(self.config_selector_combobox, 1, 1)

        new_ai_model_layout.addWidget(QLabel("Image Count:"), 2, 0)
        new_ai_model_layout.addWidget(self.image_count_label, 2, 1)

        new_ai_model_layout.addWidget(QLabel("Base Model:"), 3, 0)
        new_ai_model_layout.addWidget(self.model_selected_combobox, 3, 1)

        new_ai_model_layout.addWidget(QLabel("Resolution Size:"), 4, 0)
        new_ai_model_layout.addWidget(self.resolution_size_combobox, 4, 1)

        new_ai_model_layout.addWidget(QLabel("Continue Training From:"), 5, 0)
        new_ai_model_layout.addWidget(selected_model_layout_wrapper_widget, 5, 1)

        new_ai_model_layout.addWidget(QLabel("Epoch:"), 6, 0)
        new_ai_model_layout.addWidget(self.epoch_num_input, 6, 1)

        # Augmentations
        augmentation_frame = QFrame()
        augmentation_frame.setFrameShape(QFrame.Shape.Box)
        augmentation_frame.setLineWidth(1)
        augmentation_layout = QGridLayout()

        augmentation_title = QLabel("Training Augmentations")
        augmentation_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        augmentation_layout.addWidget(augmentation_title, 0, 0, 1, 2)

        self.horizontal_mirror_checkbox = QCheckBox("Horizontal mirroring (YOLO default)")
        self.horizontal_mirror_checkbox.setChecked(True)
        self.vertical_mirror_checkbox = QCheckBox("Vertical mirroring")
        self.vertical_mirror_checkbox.setChecked(False)
        self.rotate_90_checkbox = QCheckBox("Rotate 90°")
        self.rotate_180_checkbox = QCheckBox("Rotate 180°")
        self.rotate_270_checkbox = QCheckBox("Rotate 270°")

        augmentation_layout.addWidget(self.horizontal_mirror_checkbox, 1, 0, 1, 2)
        augmentation_layout.addWidget(self.vertical_mirror_checkbox, 2, 0, 1, 2)
        augmentation_layout.addWidget(self.rotate_90_checkbox, 3, 0)
        augmentation_layout.addWidget(self.rotate_180_checkbox, 3, 1)
        augmentation_layout.addWidget(self.rotate_270_checkbox, 4, 0)

        augmentation_frame.setLayout(augmentation_layout)

        # Init main layout
        start_train_ai_layout.addWidget(QLabel("Train New AI"), 0, 0, 1, 2, Qt.AlignmentFlag.AlignHCenter)
        start_train_ai_layout.addLayout(new_ai_model_layout, 1, 0, new_ai_model_layout.rowCount(), 2)
        start_train_ai_layout.addWidget(augmentation_frame, new_ai_model_layout.rowCount() + 1, 0, 1, 2)
        start_train_ai_layout.addWidget(self.train_button, new_ai_model_layout.rowCount() + 2, 0, 1, 2)
        start_train_ai_layout.addWidget(self.cancel_button, new_ai_model_layout.rowCount() + 2, 0, 1, 2)

        start_train_ai_widget = QWidget()
        start_train_ai_widget.setLayout(start_train_ai_layout)
        return start_train_ai_widget

    def create_ai_train_gui(self):
        """ Creates the GUI for viewing the AI training progress. Separated for readability. """
        ai_training_in_progress_layout = QGridLayout()

        # Main Label
        self.ai_name_input_in_progress = QLabel(self.ai_name_input.text())
        ai_training_in_progress_layout.addWidget(self.ai_name_input_in_progress, 0, 1, 1, 1,
                                                 Qt.AlignmentFlag.AlignCenter)
        ai_training_in_progress_layout.setSpacing(20)

        # Data Augmentation Layout
        data_augmentation_column = QFrame()
        data_augmentation_column.setFrameShape(QFrame.Box)
        data_augmentation_column.setLineWidth(1)
        data_augmentation_column.setMinimumWidth(200)

        data_augmentation_label = QLabel("Data Preparation Stage")
        data_augmentation_label.setAlignment(Qt.AlignCenter)

        self.data_augmentation_text_box = QTextEdit()
        self.data_augmentation_text_box.setReadOnly(True)

        self.data_augmentation_progress_bar = QProgressBar()
        self.data_augmentation_progress_bar.setValue(0)

        data_augmentation_layout = QVBoxLayout()
        data_augmentation_layout.addWidget(data_augmentation_label)
        data_augmentation_layout.addWidget(self.data_augmentation_text_box, Qt.AlignmentFlag.AlignCenter)
        data_augmentation_layout.addWidget(self.data_augmentation_progress_bar, Qt.AlignmentFlag.AlignBottom)

        data_augmentation_column.setLayout(data_augmentation_layout)

        # Training Layout
        training_stage_column = QFrame()
        training_stage_column.setFrameShape(QFrame.Box)
        training_stage_column.setLineWidth(1)
        training_stage_column.setMinimumWidth(200)

        training_label = QLabel("Training Stage")
        training_label.setAlignment(Qt.AlignCenter)

        self.model_training_text_box = QTextEdit()
        self.model_training_text_box.setReadOnly(True)

        self.training_progress_bar = QProgressBar()
        self.training_progress_bar.setValue(0)

        training_layout = QVBoxLayout()
        training_layout.addWidget(training_label)
        training_layout.addWidget(self.model_training_text_box, Qt.AlignmentFlag.AlignCenter)
        training_layout.addWidget(self.training_progress_bar, Qt.AlignmentFlag.AlignBottom)

        training_stage_column.setLayout(training_layout)

        # Testing Layout
        testing_stage_column = QFrame()
        testing_stage_column.setFrameShape(QFrame.Box)
        testing_stage_column.setLineWidth(1)
        testing_stage_column.setMinimumWidth(200)

        testing_label = QLabel("Testing Stage")
        testing_label.setAlignment(Qt.AlignCenter)

        self.testing_stage_text_box = QTextEdit()
        self.testing_stage_text_box.setReadOnly(True)

        self.testing_progress_bar = QProgressBar()
        self.testing_progress_bar.setValue(0)

        testing_layout = QVBoxLayout()
        testing_layout.addWidget(testing_label)
        testing_layout.addWidget(self.testing_stage_text_box, Qt.AlignmentFlag.AlignCenter)
        testing_layout.addWidget(self.testing_progress_bar, Qt.AlignmentFlag.AlignBottom)

        testing_stage_column.setLayout(testing_layout)

        # Adding layouts to main training layout
        ai_training_in_progress_layout.addWidget(data_augmentation_column, 1, 0)
        ai_training_in_progress_layout.addWidget(training_stage_column, 1, 1)
        ai_training_in_progress_layout.addWidget(testing_stage_column, 1, 2)

        ai_training_in_progress_layout.setColumnStretch(0, 1)
        ai_training_in_progress_layout.setColumnStretch(1, 1)
        ai_training_in_progress_layout.setColumnStretch(2, 1)
        ai_training_in_progress_widget = QWidget()
        ai_training_in_progress_widget.setLayout(ai_training_in_progress_layout)

        return ai_training_in_progress_widget

    def select_model(self):
        """ Displays the select model layout. """
        self.stacked_layout.setCurrentIndex(2)

    def reset_model_selection(self):
        """ Removes the currently, selected model. """
        self.selected_starting_model = ""
        self.selected_model_label.setText("None")
        self.select_model_button.setHidden(False)
        self.reset_model_selected_button.setHidden(True)

    def create_select_model(self):
        """ Creates the select model layout. """
        self.select_ai_widget = SelectAiPage()
        self.select_ai_widget.model_selected_signal.connect(self.update_model_selected)
        self.select_ai_widget.switch_view.connect(self.switch_views)
        return self.select_ai_widget

    def update_model_selected(self, name, path):
        """ Triggered when a model is selected. The selected model value is updated. """
        self.selected_starting_model = path
        self.selected_model_label.setText(name)
        self.select_model_button.setHidden(True)
        self.reset_model_selected_button.setHidden(False)
        self.switch_views()

    def load_dataset_configs(self):
        """ Retrieves the list of .txt files in the directory """
        if self.config_selector_combobox.count() != 0:
            self.config_selector_combobox.clear()

        datasets = [os.path.splitext(f)[0] for f in os.listdir(self.dataset_config_dir) if f.endswith(".txt")]
        self.config_combobox_items = ["All Images"]
        self.config_combobox_items.extend(datasets)
        self.config_selector_combobox.addItems(self.config_combobox_items)
        self.config_selector_combobox.setCurrentIndex(0)
        self.combobox_index_changed(0)

    def combobox_index_changed(self, index):
        """ Updates the selected config when the index changes. """
        if index == 0:
            self.image_count = str(file_helpers.count_image_files_in_directory(self.image_dir))
            self.image_count_label.setText(self.image_count)
        else:
            current_config = self.config_combobox_items[index]
            config_path = os.path.join(self.dataset_config_dir, current_config + ".txt")
            self.image_count = str(file_helpers.count_lines_in_file(config_path))
            self.image_count_label.setText(self.image_count)

    def switch_views(self):
        """ Returns the stacked layout, back to the previous view. """
        if (self.train_in_progress):
            if (self.stacked_layout.currentIndex() == 0):
                self.stacked_layout.setCurrentIndex(1)
            else:
                self.stacked_layout.setCurrentIndex(0)
        else:
            if (self.stacked_layout.currentIndex() == 0):
                self.stacked_layout.setCurrentIndex(2)
            else:
                self.stacked_layout.setCurrentIndex(0)

    def start_ai_train(self):
        """ Starts the AI training process. """
        # Check for valid parameters.
        if (self.ai_name_input.text() == "" or not
                self.epoch_num_input.text().isdigit() or int(self.epoch_num_input.text()) <= 0):

            error_message = QMessageBox()
            error_message.setIcon(QMessageBox.Critical)
            error_message.setWindowTitle("Error")
            error_message.setText("One or more parameters is incorrect.\nPlease check your inputs.")
            error_message.exec()
            return

        if (int(self.image_count_label.text()) <= 3):
            error_message = QMessageBox()
            error_message.setIcon(QMessageBox.Critical)
            error_message.setWindowTitle("Error")
            error_message.setText("Insufficient number of training images provided. Please add more images.")
            error_message.exec()
            return

        # Update the views.
        self.ai_name_input_in_progress.setText(self.ai_name_input.text())
        self.train_in_progress = True
        self.train_button.setVisible(False)
        self.cancel_button.setHidden(False)
        self.cancel_button.setText("Creating Pipeline...")
        self.cancel_button.setDisabled(True)

        # Create a new model info based on information provided.
        self.model_info = ModelInfo(self.ai_name_input.text(),
                                    self.model_selected_combobox.currentText(),
                                    "",
                                    "",
                                    self.image_count,
                                    epoch=self.epoch_num_input.text(),
                                    dataset_config=self.config_combobox_items[
                                        self.config_selector_combobox.currentIndex()],
                                    starting_model=self.selected_starting_model)

        augmentation_options = {
            "horizontal_mirror": self.horizontal_mirror_checkbox.isChecked(),
            "vertical_mirror": self.vertical_mirror_checkbox.isChecked(),
            "rotations": {
                "rot90": self.rotate_90_checkbox.isChecked(),
                "rot180": self.rotate_180_checkbox.isChecked(),
                "rot270": self.rotate_270_checkbox.isChecked(),
            },
        }

        # Create the training pipeline and its connections.
        self.pipeline = MainTrainPipeline(
            self,
            self.model_info,
            augmentation_options=augmentation_options,
            image_size=self.resolution_size_combobox.currentText(),
        )

        self.pipeline.pipeline_started.connect(self.pipeline_created)
        self.pipeline.data_augmentation_text.connect(self.update_data_augmentation_text)
        self.pipeline.data_augmentation_progress_bar.connect(self.update_data_augmentation_progress_bar)
        self.pipeline.model_training_text.connect(self.update_model_training_text)
        self.pipeline.model_training_progress_bar.connect(self.update_model_training_progress_bar)
        self.pipeline.model_testing_text.connect(self.update_testing_text)
        self.pipeline.model_testing_progress_bar.connect(self.update_testing_progress_bar)
        self.pipeline.pipeline_finished.connect(self.cleanup_training)

        # Start training the AI
        self.pipeline.start()

    def pipeline_created(self, val):
        """ Changes the view once the pipeline has been created and started. """
        self.switch_view_layout_wrapper_widget.setHidden(False)
        self.stacked_layout.setCurrentIndex(1)
        self.cancel_button.setText("Cancel Training")
        self.cancel_button.setDisabled(False)

    def confirm_cancel_training(self):
        """ Confirms the user wants to cancel the training. """
        confirm_message_box = QMessageBox()
        confirm_message_box.setIcon(QMessageBox.Warning)
        confirm_message_box.setWindowTitle("Cancel Confirmation")
        confirm_message_box.setText("Are you sure you want to cancel training?\nCancelling training can be unstable, "
                                    "you may need to restart the program after cancelling.")
        confirm_message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = confirm_message_box.exec()

        if (result == QMessageBox.Yes):
            thread = threading.Thread(target=self.cancel_training)
            thread.start()

    def cancel_training(self):
        """ Stops the training pipeline. """
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling...")
        if (self.pipeline and self.pipeline._is_running):
            self.pipeline.stop()
            self.pipeline.wait()
        self.cleanup_training()

    def cleanup_training(self):
        """ Cleans the training pipeline, ready for the next execution. """
        if (self.pipeline and self.pipeline.is_cleaned_up is False):
            self.data_augmentation_text_box.clear()
            self.data_augmentation_progress_bar.setValue(0)
            self.model_training_text_box.clear()
            self.training_progress_bar.setValue(0)
            self.testing_stage_text_box.clear()
            self.testing_progress_bar.setValue(0)

            self.train_in_progress = False
            self.cancel_button.setVisible(False)
            self.switch_view_layout_wrapper_widget.setVisible(False)
            self.cancel_button.setText("Cancel Training")
            self.stacked_layout.setCurrentIndex(0)
            self.train_button.setVisible(True)

            self.pipeline.is_cleaned_up = True

    def update_data_augmentation_progress_bar(self, progress):
        """ Updates the progress bar with the provided value """
        self.data_augmentation_progress_bar.setValue(progress)

    def update_data_augmentation_text(self, update):
        """ Amends the descriptive text box with the provided string. """
        self.data_augmentation_text_box.append(update)

    def update_model_training_progress_bar(self, progress):
        """ Updates the progress bar with the provided value """
        self.training_progress_bar.setValue(progress)

    def parse_update_for_progress(self, text):
        """ Parses the update text for progress on the training AI, based on its epochs,
            the current epoch and the progress in the current stage. This is used to update the progress bar. """
        # Regex to extract (epoch/total_epochs) and (percentage completed)
        pattern = r"(\d+)\/(\d+).*?(\d+)%\|"
        match = re.search(pattern, text)

        if match:
            epoch = int(match.group(1))         # Current epoch
            total_epochs = int(match.group(2))  # Total epochs
            stage_progress = int(match.group(3))  # % progress in the current stage

            # Overall progress
            self.training_progress_bar.setValue(
                ((epoch - 1) / total_epochs + (stage_progress / 100) / total_epochs) * 100
            )

    def update_model_training_text(self, update):
        """ Amends the descriptive text box with the provided string. """
        self.model_training_text_box.append(update)
        self.parse_update_for_progress(update)

    def update_testing_progress_bar(self, progress):
        """ Updates the progress bar with the provided value """
        self.testing_progress_bar.setValue(progress)

    def update_testing_text(self, update):
        """ Amends the descriptive text box with the provided string. """
        self.testing_stage_text_box.append(update)

    def closeEvent(self, event):
        """ Handle window close event """
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()  # Ensure thread exits before closing

        event.accept()  # Proceed with closing
