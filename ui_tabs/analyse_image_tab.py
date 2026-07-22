from pathlib import Path
from PySide6.QtWidgets import QWidget, QLabel, QGridLayout, QPushButton, QMessageBox, QStackedLayout
from PySide6.QtCore import Signal

from helpers import file_helpers
from helpers.image_analysis import ImageAnalysisWorker
from ui_tabs.select_ai_page import SelectAiPage
from ui_tabs.view_results_page import ViewResultsPage


class AnalyseImageTab(QWidget):
    new_image_signal = Signal(str, str, str)

    def __init__(self):
        """ Provides the functionality for the user to analyse a provided image, with a selected model. """
        super().__init__()
        self.selectedImage = ""
        self.selectedAIModelPath = ""
        self.selectAIModelName = ""

        start_analyse_image_layout = QGridLayout()

        start_analyse_image_layout.addWidget(QLabel("Select Image of X-Ray"), 0, 0)
        self.selectedImageLabel = QLabel("")
        start_analyse_image_layout.addWidget(self.selectedImageLabel, 0, 1)
        selectedImageButton = QPushButton("Browse...")
        start_analyse_image_layout.addWidget(selectedImageButton, 0, 2)
        selectedImageButton.clicked.connect(self.update_selected_image)

        start_analyse_image_layout.addWidget(QLabel("Select AI Model"), 1, 0)
        selectAIModelButton = QPushButton("Select AI")
        self.selectedModelLabel = QLabel("")
        start_analyse_image_layout.addWidget(self.selectedModelLabel, 1, 1)
        start_analyse_image_layout.addWidget(selectAIModelButton, 1, 2)
        selectAIModelButton.clicked.connect(self.update_selected_model)

        self.analyseImageButton = QPushButton("Analyse")
        start_analyse_image_layout.addWidget(self.analyseImageButton, 2, 0, 1, 3)
        self.analyseImageButton.clicked.connect(self.start_image_analysis)

        self.stacked_layout = QStackedLayout()
        start_analyse_image_layout_wrapper_widget = QWidget()
        start_analyse_image_layout_wrapper_widget.setLayout(start_analyse_image_layout)

        self.setLayout(self.stacked_layout)
        self.stacked_layout.addWidget(start_analyse_image_layout_wrapper_widget)
        self.stacked_layout.setCurrentIndex(0)

    def update_selected_model(self):
        """ Changes the view to the SelectAIPage, where the user is able to chose an AI to analyse the image. """
        self.select_ai_widget = SelectAiPage()
        self.select_ai_widget.model_selected_signal.connect(self.update_model_selected)
        self.select_ai_widget.switch_view.connect(self.reset_view)
        self.stacked_layout.addWidget(self.select_ai_widget)
        self.stacked_layout.setCurrentWidget(self.select_ai_widget)

    def update_model_selected(self, name, path):
        """ Updates the selected model path and resets the view. """
        self.selectedAIModelPath = path
        self.selectedModelLabel.setText(name)
        self.reset_view()

    def update_selected_image(self):
        """ Creates a file browser, to allow the user to select an image to be analysed """
        result = file_helpers.browse_file(self, "", "Images (*.png *.jpg)")

        # Checks to see if selected image is valid
        if result != "" and (result.endswith(".png") or result.endswith(".jpg") or result.endswith(".JPG") or result.endswith(".PNG")):
            self.selectedImage = result
            self.selectedImageLabel.setText("..." + self.selectedImage[-20:])

    def start_image_analysis(self):
        """ Analyses the selected image using the selected model. """
        # Checks for valid AI path
        self.analyseImageButton.setDisabled(True)
        self.analyseImageButton.setText("Analysing...")

        if (self.selectedAIModelPath == "" or Path(self.selectedAIModelPath).suffix != ".pt"):
            errorMessage = QMessageBox()
            errorMessage.setIcon(QMessageBox.Critical)
            errorMessage.setWindowTitle("Error")
            errorMessage.setText("Please select a valid AI model.")
            errorMessage.exec()
            self.analyseImageButton.setDisabled(False)
            self.analyseImageButton.setText("Analyse")
            return

        # Checks for valid image path
        elif (self.selectedImage == "" or Path(self.selectedImage).suffix not in {".png", ".jpg", ".PNG", ".JPG"}):
            errorMessage = QMessageBox()
            errorMessage.setIcon(QMessageBox.Critical)
            errorMessage.setWindowTitle("Error")
            errorMessage.setText("Please select a valid image to analyse.")
            errorMessage.exec()
            self.analyseImageButton.setDisabled(False)
            self.analyseImageButton.setText("Analyse")
            return

        # Setup worker and thread
        self.analyse_image_thread = ImageAnalysisWorker(self.selectedAIModelPath, self.selectedImage)
        self.analyse_image_thread.finished_signal.connect(self.analysis_finished)
        self.analyse_image_thread.start()

    def analysis_finished(self, results):
        """ Displays the view results page. """
        self.analyseImageButton.setDisabled(False)
        self.analyseImageButton.setText("Analyse")
        self.results_widget = ViewResultsPage(self.selectedImage, results)
        self.results_widget.switch_view.connect(self.reset_view)
        self.results_widget.new_image_signal.connect(self.new_image)
        self.stacked_layout.addWidget(self.results_widget)
        self.stacked_layout.setCurrentWidget(self.results_widget)

    def reset_view(self):
        """ Resets the view to its original state. """
        self.stacked_layout.setCurrentIndex(0)

        if hasattr(self, 'results_widget') and self.results_widget is not None:
            self.stacked_layout.removeWidget(self.results_widget)
            self.results_widget = None

        elif hasattr(self, 'select_ai_widget') and self.select_ai_widget is not None:
            self.stacked_layout.removeWidget(self.select_ai_widget)
            self.select_ai_widget = None

    def new_image(self, img_path, txt_path, file_name):
        """ If the AI was incorrect and the image is added to the dataset,
            this function is called to view the new annotated image."""
        self.reset_view()
        self.new_image_signal.emit(img_path, txt_path, file_name)
