from PySide6.QtWidgets import QWidget, QLabel, QGridLayout, QPushButton, QMessageBox, QScrollArea, QSizePolicy
from PySide6.QtCore import Qt
from data_classes.model_info import ModelInfo
from data_classes.list_model_widget import ListModelWidget
import os
from helpers import file_helpers


class ViewModelsTab(QWidget):
    def __init__(self):
        """ Displays the trained models and their information in a tab. """
        super().__init__()

        self.layout = QGridLayout()
        self.list_model_widget = ListModelWidget()  # Use modular widget
        self.list_model_widget.setFixedWidth(300)
        self.selected_model = ""

        self.layout.addWidget(self.list_model_widget, 0, 0, Qt.AlignmentFlag.AlignLeft)
        self.list_model_widget.item_selected.connect(self.display_model_details)

        details_layout = QGridLayout()

        self.name_label = QLabel("")
        self.setup_details_grid(details_layout, "Name: ", self.name_label, 0)

        self.model_label = QLabel("")
        self.setup_details_grid(details_layout, "Model: ", self.model_label, 1)

        self.date_time_trained_label = QLabel("")
        self.setup_details_grid(details_layout, "Date Time Trained: ", self.date_time_trained_label, 2)

        self.total_training_time_label = QLabel("")
        self.setup_details_grid(details_layout, "Total Training Time: ", self.total_training_time_label, 3)

        self.number_of_images_label = QLabel("")
        self.setup_details_grid(details_layout, "Number of Images: ", self.number_of_images_label, 4)

        self.path_label = QLabel("")
        self.path_label.setWordWrap(True)
        self.setup_details_grid(details_layout, "Path: ", self.path_label, 5)

        self.epoch_label = QLabel("")
        self.setup_details_grid(details_layout, "Epoch: ", self.epoch_label, 6)

        self.box_loss_label = QLabel("")
        self.setup_details_grid(details_layout, "Box Loss: ", self.box_loss_label, 7)

        self.cls_loss_label = QLabel("")
        self.setup_details_grid(details_layout, "CLS Loss: ", self.cls_loss_label, 8)

        self.mAP_50_label = QLabel("")
        self.setup_details_grid(details_layout, "Mean Average Precision IoU(0.5): ", self.mAP_50_label, 9)

        self.mAP_50_95_label = QLabel("")
        self.setup_details_grid(details_layout, "Mean Average Precision IoU(0.5-0.95): ", self.mAP_50_95_label, 10)

        self.precision_label = QLabel("")
        self.setup_details_grid(details_layout, "Precision: ", self.precision_label, 11)

        self.recall_label = QLabel("")
        self.setup_details_grid(details_layout, "Recall: ", self.recall_label, 12)

        self.dataset_config_label = QLabel("")
        self.setup_details_grid(details_layout, "Dataset Config: ", self.dataset_config_label, 13)

        self.starting_model_label = QLabel("")
        self.setup_details_grid(details_layout, "Starting Model: ", self.starting_model_label, 14)

        self.metamorphic_test_result_label = QLabel("")
        self.setup_details_grid(details_layout, "Metamorphic Test Results: ", self.metamorphic_test_result_label, 15)

        self.differential_test_result_label = QLabel("")
        self.setup_details_grid(details_layout, "Differential Test Results: ", self.differential_test_result_label, 16)

        self.fuzzing_test_result_label = QLabel("")
        self.setup_details_grid(details_layout, "Fuzzing Test Results: ", self.fuzzing_test_result_label, 17)

        self.results_image = QLabel(self)
        details_layout.addWidget(QLabel("Results Image: "), 18, 0, 1, 2)
        details_layout.addWidget(self.results_image, 18, 0, 1, 2)

        self.delete_model_button = QPushButton("Delete Selected Model")
        self.delete_model_button.pressed.connect(self.delete_selected_model)
        details_layout.addWidget(self.delete_model_button, 19, 0, 1, 2)

        details_layout.setAlignment(Qt.AlignHCenter)

        details_layout_widget = QWidget()
        details_layout_widget.setLayout(details_layout)

        # Add widget to scroll area so content is not cutoff in smaller screens.
        scrollable_details_widget = QScrollArea()
        scrollable_details_widget.setWidgetResizable(True)
        scrollable_details_widget.setWidget(details_layout_widget)
        scrollable_details_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.layout.addWidget(scrollable_details_widget, 0, 1)

        self.layout.setColumnStretch(0, 0)
        self.layout.setColumnStretch(1, 1)

        self.setLayout(self.layout)

    def setup_details_grid(self, grid: QGridLayout, label_text: str, label_to_be_added: QLabel, row: int):
        """ Adds the provided elements and a new label to the provided grid layout at the desired row. """
        grid.addWidget(QLabel(label_text), row, 0)
        grid.addWidget(label_to_be_added, row, 1)

    def display_model_details(self, model_name, model: ModelInfo):
        """Displays the selected config's details."""
        self.name_label.setText(model.name)
        self.model_label.setText(model.model)
        self.date_time_trained_label.setText(model.date_time_trained)
        self.total_training_time_label.setText(model.total_training_time)
        self.number_of_images_label.setText(model.number_of_images)
        self.path_label.setText(model.path)
        self.epoch_label.setText(str(model.epoch))
        self.box_loss_label.setText(str(model.box_loss))
        self.cls_loss_label.setText(str(model.cls_loss))
        self.mAP_50_label.setText(str(model.mAP_50))
        self.mAP_50_95_label.setText(str(model.mAP_50_95))
        self.precision_label.setText(str(model.precision))
        self.recall_label.setText(str(model.recall))
        self.dataset_config_label.setText(str(model.dataset_config))
        self.starting_model_label.setText(str(model.starting_model))
        self.selected_model = model.folder_name
        self.metamorphic_test_result_label.setText(str(model.metamorphic_test_result))
        self.differential_test_result_label.setText(str(model.differential_test_result))
        self.fuzzing_test_result_label.setText(str(model.fuzzing_test_result))

        pixmap = model.get_results_png().scaled(800, 400)
        self.results_image.setPixmap(pixmap)

    def reset_layout(self):
        self.name_label.setText("")
        self.model_label.setText("")
        self.date_time_trained_label.setText("")
        self.total_training_time_label.setText("")
        self.number_of_images_label.setText("")
        self.path_label.setText("")
        self.epoch_label.setText("")
        self.box_loss_label.setText("")
        self.cls_loss_label.setText("")
        self.mAP_50_label.setText("")
        self.mAP_50_95_label.setText("")
        self.precision_label.setText("")
        self.recall_label.setText("")
        self.dataset_config_label.setText("")
        self.starting_model_label.setText("")
        self.metamorphic_test_result_label.setText("")
        self.differential_test_result_label.setText("")
        self.fuzzing_test_result_label.setText("")
        self.results_image.clear()

        self.selected_model = ""

    def update_models(self):
        """ Refreshes the models in the config list widget. """
        self.list_model_widget.update_list()

    def delete_selected_model(self):
        """ Deletes the currently selected model. """
        if self.selected_model != "":
            reply = QMessageBox.question(self,
                                         "Confirmation",
                                         "Are you sure you would like to delete this model? "
                                         "This operation cannot be undone!",
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.No)
            if reply == QMessageBox.No:
                return

            model_to_be_deleted = os.path.join(os.path.abspath("trained_models"), self.selected_model)
            file_helpers.delete_folder(model_to_be_deleted)

            self.reset_layout()
            self.update_models()
