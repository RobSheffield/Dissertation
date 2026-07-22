from data_classes.ScrollableQGraphicsView import ScrollableQGraphicsView
from data_classes.bounding_box_item import BoundingBoxItem


class ScrollableViewResults(ScrollableQGraphicsView):
    """A child class of ScrollableQGraphicsView. Adds confidence score to image annotations."""
    def __init__(self, image_path, ai_result, annotation_list):
        self.ai_result = ai_result
        super().__init__(image_path, "", annotation_list)

    def load_annotations(self):
        """ Load YOLO bounding boxes and convert to pixel coordinates """
        for r in self.ai_result:
            boxes = r.boxes
            for box in boxes:
                # Extract values
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x_center, y_center, box_w, box_h = box.xywhn[0]

                box_item = BoundingBoxItem(
                    class_id,
                    x_center,
                    y_center,
                    box_w,
                    box_h,
                    self.img_w,
                    self.img_h,
                    self.annotation_list,
                    str(confidence)
                )

                self.scene.addItem(box_item)
                self.bounding_boxes.append(box_item)
