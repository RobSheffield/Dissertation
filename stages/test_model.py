import os
import random
from PySide6.QtCore import Signal, QThread
import cv2
import numpy as np
from ultralytics import YOLO
from helpers import file_helpers
from data_classes.model_info import ModelInfo
from testing.metamorphic_relations import metamorphic_relations


class TestModelStage(QThread):
    model_testing_text_signal = Signal(str)
    model_testing_progress_bar_signal = Signal(int)

    def __init__(self, model_info: ModelInfo, path_to_images, path_to_labels, 
                 path_to_all_images, path_to_models, num_samples=10, strategy="Regular Selection",
                 custom_mrs=None):
        """Tests the model and creates statistics for comparison.
        
        Args:
            custom_mrs: Optional list of tuples (name, function, [levels]) for specific MR testing.
        """
        super().__init__()
        self.model_info = model_info
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels
        self.path_to_all_images = path_to_all_images
        self.path_to_models = path_to_models
        self.num_samples = num_samples
        self.strategy = strategy
        self.custom_mrs = custom_mrs
        self.dynamic_mr_data = {}
        self.image_results = {}  # Changed from list to dict

        # Detect YOLO vs Flat structure
        train_path = os.path.join(path_to_images, "train")
        test_path = os.path.join(path_to_images, "val")
        # If 'train' folder doesn't exist, use the base path
        self.path_to_train_images = train_path if os.path.isdir(train_path) else path_to_images
        self.path_to_test_images = test_path if os.path.isdir(test_path) else path_to_images

        # Sample images using a helper
        self.selected_test_images = self.get_random_samples(self.path_to_test_images, self.num_samples)

        # Build matching annotation paths
        train_label_path = os.path.join(path_to_labels, "train")
        test_label_path = os.path.join(path_to_labels, "val")
        label_dir = test_label_path if os.path.isdir(test_label_path) else path_to_labels

        # Filter to only images that have a corresponding label file
        self.selected_test_images = []
        self.selected_test_annotation = []

        all_candidates = self.get_random_samples(self.path_to_test_images, len(
            [f for f in os.listdir(self.path_to_test_images) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        ))

        for img_path in all_candidates:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, base_name + ".txt")
            if os.path.exists(label_path):
                self.selected_test_images.append(img_path)
                self.selected_test_annotation.append(label_path)
                if len(self.selected_test_images) >= self.num_samples:
                    break

    def get_random_samples(self, directory, count):
        if not os.path.exists(directory): return []
        files = [os.path.join(directory, f) for f in os.listdir(directory) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return random.sample(files, min(len(files), count)) if files else []

    def run(self):
        self.metamorphic_tests()
        self.differential_tests()

        self.model_info.save_to_json()



    def metamorphic_tests(self):
        # Build weight path relative to project root
        best_pt = os.path.join(self.path_to_models, self.model_info.folder_name, "weights", "best.pt")
        
        if not os.path.exists(best_pt):
            self.model_testing_text_signal.emit(f"Model not found: {best_pt}")
            return
        
        model = YOLO(best_pt)
        
        work_list = []
        
        if self.custom_mrs:
            work_list = self.custom_mrs
        else:
            work_list = [
                ("Rotate 90", metamorphic_relations.rotate_90_clockwise, [None]),
                ("Vertical Mirror", metamorphic_relations.vertical_mirror, [None]),
                ("Horizontal Mirror", metamorphic_relations.horizontal_mirror, [None]),
                ("Colour Inversion", metamorphic_relations.color_inversion, [None]),
                ("Gaussian Noise", metamorphic_relations.noise_addition_gaussian, list(range(1, 10, 1))),
                ("Gamma Correction", metamorphic_relations.gamma_correction, [0.5,0.75,1,1.25,1.5, 1.75, 2.0]),
                ("Noise Addition (Salt & Pepper)", metamorphic_relations.noise_addition_salt_and_pepper, [round(x * 0.0005, 2) for x in range(1, 11)]),
                ("Brightness", metamorphic_relations.brightness_adjustment, [-100,-50, 50, 100]),
                ("Contrast", metamorphic_relations.contrast_adjustment, [0.5, 1.5, 2.0,2.5]),
                ("Blur", metamorphic_relations.blur, [3, 7, 11, 15,100]),
            ]
        
        if not work_list:
            self.model_testing_text_signal.emit("No metamorphic relations to test.")
            return

        results_summary = []
        image_analysis = []
        for name, relation_func, params in work_list:
            for param in params:
                test_name = f"{name} ({param})" if param is not None else name
                total_match = 0
                total_number = 0
                sample_index = 0
                
                for file in self.selected_test_images:
                    image = cv2.imread(file)
                    image_name = os.path.splitext(os.path.basename(file))[0]
                    
                    original_results = model(file, verbose=False)
                    original_boxes = [[0, *box.xywhn[0].tolist()] for box in original_results[0].boxes]

                    if not original_boxes:
                        result_key = f"{test_name} - {image_name}"
                        self.image_results[result_key] = cv2.resize(image, (803, 400))  # show image anyway
                        continue

                    if param is not None:
                        transformed_img, expected_boxes = relation_func(image, original_boxes, param)
                    else:
                        transformed_img, expected_boxes = relation_func(image, original_boxes)

                    transformed_results = model(transformed_img, verbose=False)
                    actual_boxes = [[0, *box.xywhn[0].tolist()] for box in transformed_results[0].boxes]

                    matched, total = self.compare_annotations(expected_boxes, actual_boxes, 0.3)
                    total_match += matched
                    total_number += total

                    result_key = f"{test_name} - {image_name}"
                    self.image_results[result_key] = self.create_comparison_visualization(
                        image, original_boxes, expected_boxes, transformed_img, actual_boxes
                    )
                    sample_index += 1

                if total_number > 0:
                    accuracy = (total_match / total_number * 100)
                    results_summary.append(f"{test_name}: {accuracy:.2f}%")
                    self.model_testing_text_signal.emit(f"Finished MR: {test_name} - {accuracy:.2f}%")
                else:
                    self.model_testing_text_signal.emit(f"Skipped MR: {test_name} - No detections found in original samples.")
        self.save_metamorphic_results(results_summary, self.selected_test_images, [name for name, _, _ in work_list])

        self.model_info.metamorphic_test_result = " | ".join(results_summary)
        self.model_testing_progress_bar_signal.emit(33)


    def create_comparison_visualization(self, original_img, original_boxes, expected_boxes, transformed_img, actual_boxes):
        """Create a 2x2 grid comparison image:
        ┌──────────────────────┬──────────────────────────┐
        │ Original + GT boxes  │ Original + Predicted     │
        │ (Green)              │ boxes (Red)              │
        ├──────────────────────┼──────────────────────────┤
        │ Transformed +        │ Transformed + Predicted  │
        │ Expected boxes (Green│ boxes (Red)              │
        └──────────────────────┴──────────────────────────┘
        Left column  = Ground truth / expected (what SHOULD be detected)
        Right column = What the model ACTUALLY predicted
        """
        target_h, target_w = 400, 400
        orig_resized = cv2.resize(original_img, (target_w, target_h))
        trans_resized = cv2.resize(transformed_img, (target_w, target_h))

        h, w = target_h, target_w

        # Top-left: Original image with ground truth boxes (green)
        panel_tl = orig_resized.copy()
        for box in original_boxes:
            self._draw_box(panel_tl, box, (0, 255, 0), h, w)

        # Top-right: Original image with model's predicted boxes (red)
        # Run prediction on original to show what model detects on unmodified image
        panel_tr = orig_resized.copy()
        for box in original_boxes:
            self._draw_box(panel_tr, box, (0, 0, 255), h, w)

        # Bottom-left: Transformed image with expected boxes (green)
        panel_bl = trans_resized.copy()
        for box in expected_boxes:
            self._draw_box(panel_bl, box, (0, 255, 0), h, w)

        # Bottom-right: Transformed image with predicted boxes (red)
        panel_br = trans_resized.copy()
        for box in actual_boxes:
            self._draw_box(panel_br, box, (0, 0, 255), h, w)

        # Add labels above each panel
        label_h = 30
        panels = [panel_tl, panel_tr, panel_bl, panel_br]
        labels = [
            "Original + GT (Green)",
            "Original + Predicted (Red)",
            "Transformed + Expected (Green)",
            "Transformed + Predicted (Red)"
        ]

        labeled_panels = []
        for panel, label in zip(panels, labels):
            label_bar = np.zeros((label_h, target_w, 3), dtype=np.uint8)
            cv2.putText(label_bar, label, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            labeled_panels.append(np.vstack([label_bar, panel]))

        # Build 2x2 grid with dividers
        divider_v = np.ones((labeled_panels[0].shape[0], 3, 3), dtype=np.uint8) * 128
        divider_h = np.ones((3, target_w * 2 + 3, 3), dtype=np.uint8) * 128

        top_row = np.hstack([labeled_panels[0], divider_v, labeled_panels[1]])
        bottom_row = np.hstack([labeled_panels[2], divider_v, labeled_panels[3]])
        grid = np.vstack([top_row, divider_h, bottom_row])

        return grid
    
    def _draw_box(self, img, box, color, h, w):
        """Helper method to draw a bounding box on an image."""
        _, x_center, y_center, box_w, box_h = box
        x_min = int((x_center - box_w/2) * w)
        y_min = int((y_center - box_h/2) * h)
        x_max = int((x_center + box_w/2) * w)
        y_max = int((y_center + box_h/2) * h)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)


    def differential_tests(self):
        """ Compares current model's recall compared to previous models. """
        previous_model_path = self.model_info.starting_model
        if previous_model_path == "":
            previous_model_path = file_helpers.get_model_for_comparison(self.path_to_models)

        # No previous models
        if previous_model_path == "":
            # Emit a status update
            result_string = "No previous model found. Passing Test."
            self.model_info.differential_test_result = result_string
            self.model_testing_text_signal.emit(f"Differential Testing - {result_string} ")
            self.model_testing_progress_bar_signal.emit(66)
            return

        current_model = YOLO(self.model_info.get_best_pt_path())
        previous_model = YOLO(previous_model_path)

        total_correct_bounding_boxes = 0
        current_model_performance = 0
        previous_model_performance = 0

        length_of_selected_images = len(self.selected_test_images)

        # Compare performance for each model, to the correct bounding boxes for the image.
        for index in range(0, length_of_selected_images):
            current_image = self.selected_test_images[index]
            current_annotation = self.selected_test_annotation[index]

            if current_annotation is None:
                self.model_testing_text_signal.emit(
                    f"Differential Testing - Skipping image {index + 1}/{length_of_selected_images} (no annotation)")
                continue

            # Loads correct bounding boxes
            correct_bounding_boxes = []
            with open(current_annotation, "r") as f:
                for line in f.readlines():
                    stripped_line = line.strip()
                    class_id, x_center, y_center, box_w, box_h = map(float, stripped_line.split())
                    correct_bounding_boxes.append([class_id, x_center, y_center, box_w, box_h])
            total_correct_bounding_boxes += len(correct_bounding_boxes)

            # Calculates new model's bounding boxes
            current_model_results = current_model(current_image, verbose=False)
            current_model_bounding_boxes = []
            for box in current_model_results[0].boxes:
                x_center, y_center, w, h = box.xywhn[0].tolist()
                current_model_bounding_boxes.append([0, x_center, y_center, w, h])

            # Calculates previous model's bounding boxes
            previous_model_results = previous_model(current_image, verbose=False)
            previous_model_bounding_boxes = []
            for box in previous_model_results[0].boxes:
                x_center, y_center, w, h = box.xywhn[0].tolist()
                previous_model_bounding_boxes.append([0, x_center, y_center, w, h])

            # Adds the matching bounding boxes between the true value and the model's predicted values
            current_model_performance += self.compare_annotations(current_model_bounding_boxes,
                                                                  correct_bounding_boxes, 0.5)[0]

            previous_model_performance += self.compare_annotations(previous_model_bounding_boxes,
                                                                   correct_bounding_boxes, 0.5)[0]

            # Emit a status update
            self.model_testing_text_signal.emit("Differential Testing - Calculated "
                                                f"{index + 1}/{length_of_selected_images}")
            self.model_testing_progress_bar_signal.emit(int(index / length_of_selected_images * 33) + 33)

        if total_correct_bounding_boxes == 0:
            percentage_current_correct = 0
            percentage_previous_correct = 0
        else:
            percentage_current_correct = current_model_performance / total_correct_bounding_boxes
            percentage_previous_correct = previous_model_performance / total_correct_bounding_boxes

        # Difference can be positive (improved) or negative (degraded)
        difference = (percentage_current_correct - percentage_previous_correct) * 100

        result_string = (f"Change in recall of {difference}%"
                         f" since previous model, based on {length_of_selected_images}"
                         " images.")

        self.model_info.differential_test_result = result_string

        self.model_testing_text_signal.emit(f"Differential Testing Finished, FINAL RESULT - {result_string}")
        self.model_testing_progress_bar_signal.emit(66)



    def select_random_images_from_dir(self, image_dir, count=20):
        """ Selects a provided number of images from the provided DIR. """
        if not os.path.exists(image_dir):
            return []
            
        all_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not all_files:
            return []

        # Ensure we don't try to sample more than exists
        actual_count = min(len(all_files), count)
        return random.sample(all_files, actual_count)
    

    def save_metamorphic_results(self, results, image_files,image_relations):
        with open(os.path.join(self.path_to_models, self.model_info.folder_name, "metamorphic_results.txt"), "w") as f:
            relation_dict = {}
            for i,relation in enumerate(image_relations):
                relation_dict[image_files[i]] = relation
            f.write(str(relation_dict))
            f.write("\n")
            f.write(str(results))
        

    def select_random_images(self):
        """ Selects images for testing. Defaults to the folder itself if no YOLO structure exists. """
        # Check if the path contains a 'train' subfolder, otherwise use the path directly
        potential_train_path = os.path.join(self.path_to_images, "train")
        image_dir = potential_train_path if os.path.exists(potential_train_path) else self.path_to_images

        try:
            # Filter for common image extensions
            all_images = [
                os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
        except Exception:
            all_images = []

        return random.sample(all_images, 20)

    def rotate_annotations_by_90(self, boxes):
        """ Rotates the annotations by 90 degrees clockwise """
        rotated = []

        for box in boxes:
            class_id, x, y, bw, bh = box
            new_x = 1 - y
            new_y = x
            new_bw = bh
            new_bh = bw

            rotated.append([class_id, new_x, new_y, new_bw, new_bh])

        return rotated

    def intersection_over_union(self, box1, box2):
        """ Calculates if the predicted bounding boxes overlap. """
        # Turning the yolo coordinates into normal box coordinates with
        # each var being a corner of the box

        x1_min = box1[1] - box1[3] / 2
        y1_min = box1[2] - box1[4] / 2
        x1_max = box1[1] + box1[3] / 2
        y1_max = box1[2] + box1[4] / 2

        x2_min = box2[1] - box2[3] / 2
        y2_min = box2[2] - box2[4] / 2
        x2_max = box2[1] + box2[3] / 2
        y2_max = box2[2] + box2[4] / 2

        # Find the intersection points between the boxes
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)  
        inter_y2 = min(y1_max, y2_max) 

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union = box1_area + box2_area - inter_area

        return inter_area / union if union else 0

    def compare_annotations(self, boxes1, boxes2, intersection_over_union_percentage=0.5):
        """ Compares to see if any annotations match. """
        matched = 0
        for original_box in boxes1:
            for rotated_box in boxes2:
                # If the IoU is over the threshold of 0.1
                if self.intersection_over_union(original_box, rotated_box) >= intersection_over_union_percentage:
                    matched += 1
                    break

        return matched, len(boxes1)
    
    def select_metamorphic_relations_dynamically(self):
        
        pass

    def run_grad_cam(self):
        pass
