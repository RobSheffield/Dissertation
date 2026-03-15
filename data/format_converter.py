# As the provided dataset has image annotations in a different format.
# This file converts them to a format that YOLO understands.

import os
import numpy as np
from PIL import Image

def convert_gt_to_yolo(gt_file, images_dir, output_dir, class_id=0):
    """
    Converts a ground truth file to YOLOv5 format using the original working logic.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        gt_data = np.loadtxt(gt_file)
        # Handle case where there is only one annotation in the file
        if gt_data.ndim == 1:
            gt_data = gt_data.reshape(1, -1)
    except Exception as e:
        print(f"Error reading GT file {gt_file}: {e}")
        return []

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    processed_images = set()
    converted_stems = []

    for image_file in image_files:
        try:
            # Extract image ID from the file name (e.g. C0001_0015.png -> 15)
            image_id = int(image_file.split('_')[1].split('.')[0]) 
        except (IndexError, ValueError):
            print(f"Skipping {image_file}, irregular naming format.")
            continue
            
        stem = os.path.splitext(image_file)[0]
        converted_stems.append(stem)
        image_path = os.path.join(images_dir, image_file)

        # Get image dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # Find ground truth annotations for the current image
        if len(gt_data) > 0 and image_id in gt_data[:, 0]:
            image_annotations = gt_data[gt_data[:, 0] == image_id][:, 1:]
            yolo_lines = []

            # Convert ground truth annotations to YOLO format
            for bbox in image_annotations:
                x_min, x_max, y_min, y_max = bbox  # Using your original coordinate unpacking
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                norm_width = (x_max - x_min) / img_width
                norm_height = (y_max - y_min) / img_height

                yolo_lines.append(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}")

            # Save YOLO annotations
            yolo_file_path = os.path.join(output_dir, stem + '.txt')
            with open(yolo_file_path, 'w') as yolo_file:
                yolo_file.write('\n'.join(yolo_lines))
        else:
            # Create an empty label file for defect-free images (Negatives)
            yolo_file_path = os.path.join(output_dir, stem + '.txt')
            open(yolo_file_path, 'w').close()

        processed_images.add(image_id)

    return converted_stems

if __name__ == "__main__":
    ground_truth_file = "FULL_PATH_TO_GROUND_TRUTH_TXT_FOLDER/ground_truth.txt"
    images_directory = "FULL_PATH_TO_IMAGES_DIR"
    output_directory = "FULL_PATH_TO_OUTPUT_DIR"
    convert_gt_to_yolo(ground_truth_file, images_directory, output_directory, class_id=0)
