from PySide6.QtWidgets import (QWidget, QGridLayout,
                               QListWidget, QTextBrowser, QLabel)
import os
import markdown


class DocumentationViewerTab(QWidget):
    def __init__(self):
        """ Provides functionality on viewing the information about the documentation. """
        super().__init__()
        self.docs_dir = "docs"

        # UI elements
        self.file_list = QListWidget()
        self.file_list.setMaximumWidth(200)
        self.documentation_viewer = QTextBrowser()

        # Layout
        documentation_layout = QGridLayout()
        documentation_layout.addWidget(QLabel("Documentation"), 0, 0, 1, 2)
        documentation_layout.addWidget(self.file_list, 1, 0)
        documentation_layout.addWidget(self.documentation_viewer, 1, 1)

        documentation_layout.setColumnStretch(1, 5)

        self.setLayout(documentation_layout)

        self.markdown_files = [
            f for f in os.listdir(self.docs_dir) if f.endswith(".md")
        ]
        self.file_list.addItems(self.markdown_files)

        self.file_list.currentItemChanged.connect(self.load_selected_markdown)

        # Load first by default
        if self.markdown_files:
            self.file_list.setCurrentRow(0)

    def load_selected_markdown(self, current, previous):
        """ Loads the selected markdown file and displays it on the doc viewer. """
        if current:
            file_name = current.text()
            file_path = os.path.join(self.docs_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                md_text = f.read()

            # Converts markdown file to HTML so that Qt can parse it with the correct format.
            html = markdown.markdown(md_text, extension=['extra'])
            self.documentation_viewer.setHtml(html)
