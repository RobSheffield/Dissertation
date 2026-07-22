from PySide6.QtWidgets import (QWidget, QGridLayout,
                               QStackedLayout, QPushButton,
                               QMessageBox, QComboBox)
import os
from data_classes.image_item_container import ImageItemContainer
from helpers import file_helpers
from ui_tabs.create_dataset import CreateDatasetConfig
from ui_tabs.edit_config import EditConfig
from ui_tabs.view_dataset import ViewDataset
from ui_tabs.view_image import ImageViewer


class DatasetConfigTab(QWidget):
    def __init__(self):
        """Displays stored images, and their related annotations.
           Provides dataset creation and editing tools.
           Allows image deletion and addition."""
        super().__init__()
        self.main_stacked_layout = QStackedLayout()
        self.view_config_stacked_layout = QStackedLayout()

        # Paths
        data_dir = os.path.abspath("stored_training_images")
        self.dataset_config_dir = os.path.join(data_dir, "datasets")
        self.image_dir = os.path.join(data_dir, "images", "raw")
        self.annotation_dir = os.path.join(data_dir, "labels", "raw")

        # Init values
        self.image_files = []
        self.column_count = 3  # Used when changing the size of the display.
        self.previous_page_stack = []  # Previous page's indexes are stored in this stack for easy retrieval
        self.create_config_active = False  # Stops creating two create config windows when the button is held.

        # Buttons
        self.create_new_config_button = QPushButton("Create New Config")
        self.edit_config_button = QPushButton("Edit Config")
        self.delete_config_button = QPushButton("Delete Config")
        self.create_new_config_button.pressed.connect(self.create_config)
        self.edit_config_button.pressed.connect(self.edit_config)
        self.delete_config_button.pressed.connect(self.delete_config)

        # Select Image Config
        self.dataset_config_combobox = QComboBox()
        self.update_dataset_configs(self.dataset_config_dir)
        self.dataset_config_combobox.currentIndexChanged.connect(self.load_config)

        # Main Layout
        self.stacked_layout_widget_wrapper = QWidget()
        self.stacked_layout_widget_wrapper.setLayout(self.view_config_stacked_layout)

        self.dataset_config_grid_layout = QGridLayout()
        self.dataset_config_grid_layout.addWidget(self.dataset_config_combobox, 0, 0)
        self.dataset_config_grid_layout.addWidget(self.edit_config_button, 0, 1)
        self.dataset_config_grid_layout.addWidget(self.delete_config_button, 0, 2)
        self.dataset_config_grid_layout.addWidget(self.create_new_config_button, 0, 3)
        self.dataset_config_grid_layout.addWidget(self.stacked_layout_widget_wrapper, 1, 0, 1, 4)

        self.dataset_config_grid_layout_widget_container = QWidget()
        self.dataset_config_grid_layout_widget_container.setLayout(self.dataset_config_grid_layout)

        self.setLayout(self.main_stacked_layout)

        self.main_stacked_layout.addWidget(self.dataset_config_grid_layout_widget_container)

        self.load_images()

        self.load_config(0)  # Loads default view.

        self.view_config_stacked_layout.setCurrentIndex(0)
        self.main_stacked_layout.setCurrentIndex(0)

    def set_column_count(self, count):
        """ Updates the grid view to the current width of the window. """
        if count != self.column_count:
            self.column_count = max(1, count)
            if hasattr(self, "edit_config_page") and self.edit_config_page != "":
                self.edit_config_page.set_column_count(count // 2)
                self.current_config.column_count = count
            elif self.current_config is not None:
                self.current_config.set_column_count(count)

    def update_dataset_configs(self, dataset_dir):
        """ Retrieve list of .txt files in the directory """
        if self.dataset_config_combobox.count() != 0:
            self.dataset_config_combobox.clear()

        datasets = [os.path.splitext(f)[0] for f in os.listdir(dataset_dir) if f.endswith(".txt")]
        self.combobox_items = ["All Images"]
        self.combobox_items.extend(datasets)
        self.dataset_config_combobox.addItems(self.combobox_items)
        self.dataset_config_combobox.setCurrentIndex(0)

    def load_images(self):
        """ Loads the images to be added to the list of item containers """
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', '.PNG', '.JPG'))]

        self.list_of_item_containers = []
        self.filtered_list = []

        for i, img_file in enumerate(self.image_files):
            self.append_images(img_file)

    def append_images(self, img_file):
        """ Creates a new image container and adds it to the view. """
        item_container = ImageItemContainer(self.image_dir, img_file, self.open_detail_view)
        self.list_of_item_containers.append(item_container)

    def open_detail_view(self, img_path, file_name):
        """ Open selected image in a detailed view. """
        annotation_path = os.path.join(self.annotation_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")

        # Creates a layout to contain the detailed view, that facilitates returning to current view.
        self.view_image_layout = QGridLayout()
        backButton = QPushButton("Return To View Images")
        backButton.pressed.connect(self.reset_layout)
        self.view_image_layout.addWidget(backButton, 0, 0)
        self.img_viewer_widget = ImageViewer(img_path, annotation_path, file_name)
        self.img_viewer_widget.save_finished_signal.connect(self.reset_layout)
        self.img_viewer_widget.delete_image_signal.connect(self.delete_image)
        self.view_image_layout.addWidget(self.img_viewer_widget)

        self.view_image_layout_wrapper = QWidget()
        self.view_image_layout_wrapper.setLayout(self.view_image_layout)

        self.main_stacked_layout.addWidget(self.view_image_layout_wrapper)
        self.previous_page_stack.append(self.main_stacked_layout.currentIndex())
        self.main_stacked_layout.setCurrentWidget(self.view_image_layout_wrapper)

    def reset_layout(self):
        """ Resets the stacked layout to the previous page, while cleaning up previous pages. """
        if self.main_stacked_layout.count() > 1:
            if hasattr(self, "img_viewer_widget"):
                if self.img_viewer_widget.image_view.is_unsaved_changes():
                    reply = QMessageBox.question(self,
                                                 "Unsaved Changes",
                                                 "You have unsaved changes to the bounding boxes. "
                                                 "Do you want to exit without saving?",
                                                 QMessageBox.Yes | QMessageBox.No,
                                                 QMessageBox.No)
                    if reply == QMessageBox.No:
                        return

            if hasattr(self, "edit_config_page") and self.edit_config_page != "":
                if self.edit_config_page.delete:
                    self.edit_config_button.setDisabled(False)
                    self.current_config.set_column_count(self.current_config.column_count)
                    self.edit_config_page = ""
                    self.current_config.refresh()

            #  Previous pages are stored as a stack, so when an a page is visited on top of a page,
            #  the previous pages can be kept track.
            try:
                previous_page = self.previous_page_stack.pop()
            except IndexError:
                previous_page = 0

            current_page = self.main_stacked_layout.currentIndex()
            self.main_stacked_layout.setCurrentIndex(previous_page)

            self.main_stacked_layout.removeWidget(
                self.main_stacked_layout.itemAt(
                    current_page
                ).widget()
            )

            # The view has been reset, therefore it must no longer be on the create config view,
            # even if the previous window was not a create config window.
            self.create_config_active = False

    def delete_image(self, image_path, annotation_path, img_name, colour_path=""):
        """ Deletes the provided image and annotation path """
        for img in self.list_of_item_containers:
            if self.compare_name_to_search(img, img_name):
                self.reset_layout()
                self.list_of_item_containers.remove(img)
                file_helpers.delete_file(image_path)
                file_helpers.delete_file(annotation_path)
                if colour_path != "":
                    file_helpers.delete_file(colour_path)
                file_helpers.remove_filename_from_configs(img_name, self.dataset_config_dir)
                break

        if self.current_config:
            # Removes the deleted image from the list of images
            self.current_config.update_images(self.list_of_item_containers)

    def compare_name_to_search(self, img, search_string):
        """ Compares the provided search string, with the provided image's label. """
        img_label = img.get_label_text()
        return search_string.lower() in img_label.lower()  # Check contains the search string

    def create_config(self):
        """ Creates the view to create the config dataset. """
        if not self.create_config_active:  # Prevents create config button from being held, and creating two views.
            self.create_config_active = True
            self.create_config_page = CreateDatasetConfig(self.dataset_config_dir, self.image_dir)
            self.create_config_page.dataset_created_signal.connect(self.config_created)
            self.create_config_page.cancel_creation_signal.connect(self.reset_layout)
            self.main_stacked_layout.addWidget(self.create_config_page)
            self.previous_page_stack.append(self.main_stacked_layout.currentIndex())
            self.main_stacked_layout.setCurrentWidget(self.create_config_page)

    def config_created(self, new_file_name):
        """ Called when a config has been created. Allows user to add items to new config. """
        self.update_dataset_configs(self.dataset_config_dir)
        index_of_new_config = self.combobox_items.index(new_file_name)
        self.dataset_config_combobox.setCurrentIndex(index_of_new_config)
        self.load_config(index_of_new_config)
        self.reset_layout()

    def load_config(self, index):
        """ Changes the view to focus the displayed items to the items contained in the selected config. """
        if index >= 0:  # Ensure a valid selection
            # Reset from previous selected config
            if hasattr(self, "current_config"):
                self.view_config_stacked_layout.setCurrentIndex(0)
                self.view_config_stacked_layout.removeWidget(self.current_config)
                self.current_config = None

            # Display all items
            if index == 0:
                self.edit_config_button.setHidden(True)
                self.delete_config_button.setHidden(True)
                self.current_config = ViewDataset(self.list_of_item_containers, annotation_dir=self.annotation_dir)
            # Display items from selected config
            else:
                self.edit_config_button.setHidden(False)
                self.delete_config_button.setHidden(False)
                selected_file = self.combobox_items[index]
                config_path = os.path.join(self.dataset_config_dir, selected_file + ".txt")
                self.current_config = ViewDataset(self.list_of_item_containers,
                                                  annotation_dir=self.annotation_dir, dataset_file_path=config_path)

            self.current_config.set_column_count(self.column_count)
            self.view_config_stacked_layout.addWidget(self.current_config)
            self.view_config_stacked_layout.setCurrentWidget(self.current_config)

            self.current_config.add_image_signal.connect(self.add_image)

    def add_image(self, path_to_config="", safe_image_path="", txt_destination_path="", safe_image_file_name=""):
        """ Adds an image to the database. Images can be provided or selected.
            A config can be provided to add the new image to by default."""
        if safe_image_path == "":
            # Image prompt, user selects valid image
            result = file_helpers.browse_file(self, "", "Images (*.png *.jpg)")
            if result != "" and (result.endswith(".png") or result.endswith(".jpg") or result.endswith(".JPG") or result.endswith(".PNG")):
                self.selectedImage = result
            else:
                return

            new_img_folder = os.path.join("stored_training_images", "images", "raw")
            new_txt_folder = os.path.join("stored_training_images", "labels", "raw")
            safe_image_path, safe_image_file_name = file_helpers.create_valid_data_file_name(result,
                                                                                             new_img_folder,
                                                                                             new_txt_folder)

            file_helpers.move_file(result, safe_image_path)

        if txt_destination_path == "":
            # Annotation file created
            file_name_without_ext = (os.path.splitext(safe_image_file_name)[0]) + ".txt"
            txt_destination_path = os.path.join(new_txt_folder, file_name_without_ext)

            file_helpers.create_file_empty_txt(txt_destination_path)

        if path_to_config != "":
            file_helpers.add_new_img(path_to_config, file_name_without_ext)

        self.append_images(safe_image_file_name)
        self.current_config.update_images(self.list_of_item_containers)
        self.open_detail_view(safe_image_path, safe_image_file_name)

    def edit_config(self):
        """ Changes the view to display the config information and change images contained in it. """
        self.edit_config_button.setDisabled(True)
        config_name = self.combobox_items[self.dataset_config_combobox.currentIndex()]
        config_path = os.path.join(self.dataset_config_dir, config_name + ".txt")

        self.edit_config_page = EditConfig(config_path, self.list_of_item_containers,
                                           self.current_config.column_count // 2)

        self.edit_config_page.return_signal.connect(self.reset_layout)

        self.main_stacked_layout.addWidget(self.edit_config_page)
        self.previous_page_stack.append(self.main_stacked_layout.currentIndex())
        self.main_stacked_layout.setCurrentWidget(self.edit_config_page)

    def delete_config(self):
        """ Deletes the currently selected config. """
        combo_box_current_index = self.dataset_config_combobox.currentIndex()
        if combo_box_current_index != 0:
            reply = QMessageBox.question(self,
                                         "Confirmation",
                                         "Are you sure you would like to delete this config? "
                                         "This operation cannot be undone!",
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.No)
            if reply == QMessageBox.No:
                return

            config_to_be_delete_name = self.combobox_items[combo_box_current_index]
            config_to_be_delete_path = os.path.join(self.dataset_config_dir, config_to_be_delete_name + ".txt")
            file_helpers.delete_file(config_to_be_delete_path)

            self.update_dataset_configs(self.dataset_config_dir)
