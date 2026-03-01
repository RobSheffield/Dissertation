# As the provided dataset has image annotations in a different format.
# This file converts them to a format that YOLO understands.

import os
import numpy as np
from PIL import Image


def convert_gt_to_yolo(gt_file, images_dir, output_dir, class_id=0):
    """
    Converts a ground truth file to YOLOv5 format.

    Args:
        gt_file (str): Path to the `ground_truth.txt` file.
        images_dir (str): Directory containing the images.
        output_dir (str): Directory to save YOLOv5 annotations.
        class_id (int): YOLO class ID to assign to all annotations (default is 0).
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load ground truth data
    gt_data = np.loadtxt(gt_file)
    if gt_data.ndim == 1:
        gt_data = gt_data.reshape(1, -1)  # Handle single annotation row

    # Iterate over all images in the directory
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_id = int(image_file.split('_')[1].split('.')[0])  # Extract image ID from the file name
        image_path = os.path.join(images_dir, image_file)

        # Get image dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # Find ground truth annotations for the current image
        matches = gt_data[gt_data[:, 0] == image_id]

        if len(matches) == 0:
            # No annotation - skip entirely, do NOT create empty label
            print(f"No ground truth for {image_file}, skipping.")
            continue

        yolo_lines = []
        for bbox in matches[:, 1:]:
            x_min, x_max, y_min, y_max = bbox
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            norm_width = (x_max - x_min) / img_width
            norm_height = (y_max - y_min) / img_height
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

        if yolo_lines:
            label_file = os.path.join(output_dir, os.path.splitext(image_file)[0] + ".txt")
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
            print(f"Saved {len(yolo_lines)} annotations for {image_file}.")
        else:
            print(f"WARNING: Empty annotations for {image_file}, skipping.")


if __name__ == "__main__":
    ground_truth_file = "FULL_PATH_TO_GROUND_TRUTH_TXT_FOLDER/ground_truth.txt"
    images_directory = "FULL_PATH_TO_IMAGES_DIR"
    output_directory = "FULL_PATH_TO_OUTPUT_DIR"
    convert_gt_to_yolo(ground_truth_file, images_directory, output_directory, class_id=0)
