from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene,
                               QGraphicsPixmapItem, QGraphicsRectItem)
from PySide6.QtGui import QPixmap, QPen, QWheelEvent
from PySide6.QtCore import Qt, QRectF, Signal
from data_classes.bounding_box_item import BoundingBoxItem
import os
import cv2


class ScrollableQGraphicsView(QGraphicsView):
    """Displays a provided image and its related annotations.
    Facilitates functionalities such as zoom, adding annotations, saving annotations and deletion."""
    annotation_added = Signal()

    def __init__(self, image_path, annotation_path, annotation_list, annotation_colour_path=""):
        super().__init__()

        self.scale_factor = 1.0  # To track the zoom scale
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.annotation_list = annotation_list
        self.annotation_colour_path = annotation_colour_path

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.scene.setBackgroundBrush(Qt.GlobalColor.black)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

        pixmap = QPixmap(self.image_path)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)

        self.img = cv2.imread(self.image_path)
        self.img_h, self.img_w, _ = self.img.shape

        self.bounding_boxes = []
        self.load_annotations()

        self.drawing_box = None
        self.start_point = None
        self.is_adding_box = False
        self.unsaved_changes = False

    def load_annotations(self):
        """ Load YOLO bounding boxes and convert to pixel coordinates """
        if os.path.exists(self.annotation_path):
            # Check to see if any annotations have been added by the image analysis AI
            colour_lines = set()
            if self.annotation_colour_path:
                with open(self.annotation_colour_path, "r") as cf:
                    colour_lines = set(line.strip() for line in cf if line.strip())

            with open(self.annotation_path, "r") as f:
                for line in f.readlines():
                    stripped_line = line.strip()
                    class_id, x_center, y_center, box_w, box_h = map(float, stripped_line.split())

                    # If the line is in the AI colour lines, it must be created by an AI
                    is_ai_colour = stripped_line in colour_lines

                    box = BoundingBoxItem(class_id, x_center, y_center, box_w, box_h, self.img_w, self.img_h,
                                          self.annotation_list, ai_colour=is_ai_colour)

                    self.scene.addItem(box)  # Adds bounding box to image scene
                    self.bounding_boxes.append(box)

    def save_annotations(self) -> str:
        """Converts all bounding box annotations to a string."""
        new_annotations_txt = ""
        length_of_list = len(self.bounding_boxes)
        if length_of_list != 0:
            for index, box in enumerate(self.bounding_boxes):
                new_annotations_txt += box.convert_to_string(self.img_w, self.img_h)
                if index != length_of_list - 1:
                    new_annotations_txt += "\n"

        self.unsaved_changes = False
        return new_annotations_txt

    def on_annotation_selected(self, item):
        """ Highlight the corresponding bounding box when an annotation is clicked """
        for box in self.bounding_boxes:
            if box.list_item == item:
                box.setPen(QPen(Qt.green, 3))
                box.setSelected(True)
            else:
                box.setPen(QPen(box.colour, 2))
                box.setSelected(False)

    def add_annotation(self):
        """Allows the user to draw on the image, to create a new annotation."""
        self.setCursor(Qt.CrossCursor)  # Change cursor to crosshair for drawing
        self.drawing_box = None
        self.start_point = None
        self.is_adding_box = True

    def cancel_add_annotation(self):
        """Cancels adding an annotation."""
        self.is_adding_box = False

    def delete_selected_annotation(self):
        """ Remove selected annotation from scene and list """
        selected_items = self.annotation_list.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            for box in self.bounding_boxes:
                if box.list_item == item:
                    self.scene.removeItem(box)
                    self.bounding_boxes.remove(box)
                    self.annotation_list.takeItem(self.annotation_list.row(item))
                    self.unsaved_changes = True
                    break

    def wheelEvent(self, event: QWheelEvent):
        """ Handle zoom in/out using the mouse wheel """
        zoom_in = event.angleDelta().y() > 0
        if zoom_in:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self):
        """ Zoom in the image by increasing scale factor """
        self.setTransform(self.transform().scale(1 / self.scale_factor, 1 / self.scale_factor))
        self.scale_factor *= 1.2
        self.apply_zoom()

    def zoom_out(self):
        """ Zoom out the image by decreasing scale factor """
        self.setTransform(self.transform().scale(1 / self.scale_factor, 1 / self.scale_factor))
        self.scale_factor /= 1.2
        self.apply_zoom()

    def apply_zoom(self):
        """ Apply the zoom to the entire scene (image and bounding boxes) """
        self.scene.setSceneRect(self.scene.itemsBoundingRect())  # Make sure the scene updates its bounding box
        self.setTransform(self.transform().scale(self.scale_factor, self.scale_factor))

    def mousePressEvent(self, event):
        """ Start drawing the bounding box when the user clicks the mouse """
        if self.is_adding_box and event.button() == Qt.LeftButton:
            self.start_point = self.mapToScene(event.pos())  # Get start point of the bounding box
            self.drawing_box = QGraphicsRectItem(QRectF(self.start_point, self.start_point))  # Create an empty rect
            self.drawing_box.setPen(QPen(Qt.red, 2, Qt.DashLine))  # Set a dashed line for the new bounding box
            self.scene.addItem(self.drawing_box)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """ Update the size of the bounding box as the user drags the mouse """
        if self.is_adding_box and self.drawing_box and self.start_point:
            end_point = self.mapToScene(event.pos())  # Get end point of the bounding box
            rect = QRectF(self.start_point, end_point).normalized()  # Normalize the rect to ensure correct orientation
            self.drawing_box.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """ Finalise the bounding box and add it to the annotation list when the mouse is released """
        if self.drawing_box and self.start_point:
            end_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, end_point).normalized()

            # Create a bounding box item and add it to the scene
            box = BoundingBoxItem.from_rect("0", rect, self.img_w, self.img_h, self.annotation_list)
            self.scene.addItem(box)
            self.bounding_boxes.append(box)

            # Reset drawing state
            self.scene.removeItem(self.drawing_box)
            self.drawing_box = None
            self.start_point = None

            self.setCursor(Qt.ArrowCursor)  # Reset cursor to normal

            self.is_adding_box = False
            self.unsaved_changes = True
            self.annotation_added.emit()
        else:
            super().mouseReleaseEvent(event)

    def is_unsaved_changes(self) -> bool:
        """Checks for unsaved changes when attempting to return to previous view."""
        return self.unsaved_changes
