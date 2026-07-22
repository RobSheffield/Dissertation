from PySide6.QtWidgets import (QGraphicsRectItem, QListWidgetItem)
from PySide6.QtGui import QPen
from PySide6.QtCore import Qt, QRectF


class BoundingBoxItem(QGraphicsRectItem):
    """ Custom class to handle bounding box selection/editing """
    def __init__(self, class_id, x_center, y_center, box_w, box_h, img_w,
                 img_h, list_widget, confidence_score="", ai_colour=False):

        self.class_id = int(class_id)
        self.x_center = x_center
        self.y_center = y_center
        self.box_w = box_w
        self.box_h = box_h
        self.img_w = img_w
        self.img_h = img_h
        self.list_widget = list_widget
        self.confidence_score = confidence_score

        if ai_colour is True:
            self.colour = Qt.blue
        else:
            self.colour = Qt.red

        # Calculates coordinates for drawing the bounding box
        x1 = (self.x_center - self.box_w / 2) * self.img_w
        y1 = (self.y_center - self.box_h / 2) * self.img_h
        rect = QRectF(x1, y1, self.box_w * self.img_w, self.box_h * self.img_h)
        super().__init__(rect)

        # Draws bounding box
        self.setPen(QPen(self.colour, 2))

        self.setFlag(QGraphicsRectItem.ItemIsMovable, False)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, False)

        if self.confidence_score == "":
            annotation_text = f"Class {self.class_id}: ({x1:.1f}, {y1:.1f})"
        else:
            annotation_text = f"({x1:.1f}, {y1:.1f}) - Confidence Score: {self.confidence_score}"

        list_item = QListWidgetItem(annotation_text)
        self.list_widget.addItem(list_item)

        self.annotation_text = annotation_text
        self.list_item = list_item

    @classmethod
    def from_rect(cls, class_id, rect: QRectF, img_w, img_h, list_widget):
        """Allows bounding box to be loaded from a rectangle drawn by the user,
        instead of from a YOLO annotation file."""
        rect_vals = rect.getRect()

        box_w = rect_vals[2] / img_w
        box_h = rect_vals[3] / img_h

        x_center = (rect_vals[0] / img_w) + (box_w / 2)
        y_center = (rect_vals[1] / img_h) + (box_h / 2)

        return cls(class_id, x_center, y_center, box_w, box_h, img_w,
                   img_h, list_widget)

    def convert_to_string(self, img_w, img_h) -> str:
        """Converts bounding box to string to be able to be saved to YOLO annotation file."""
        return_statement = f"{self.class_id} {self.x_center} {self.y_center} {self.box_w} {self.box_h}"
        return return_statement
