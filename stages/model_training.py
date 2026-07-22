import os
import yaml


def create_yaml(data_dir, output_path, class_names):
    """
    Create a dataset YAML file for YOLOv5 training.

    Args:
        data_dir (str): Path to the dataset directory containing 'images' and 'labels' subdirectories.
        output_path (str): Path to save the YAML file.
        class_names (list): List of class names.
    """
    yaml_data = {
        "train": os.path.join(data_dir, "images/train"),
        "val": os.path.join(data_dir, "images/val"),
        "nc": len(class_names),
        "names": class_names
    }

    with open(output_path, 'w') as file:
        yaml.dump(yaml_data, file)

    print(f"YAML file saved at: {output_path}")
