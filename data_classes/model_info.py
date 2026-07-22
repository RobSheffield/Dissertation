import json
import os


class ModelInfo:
    """Class to represent an AI Model configuration object."""
    def __init__(self, name, model, date_time_trained, total_training_time, number_of_images, path="", epoch="",
                 box_loss="", cls_loss="", mAP_50="", mAP_50_95="", precision="", recall="",
                 dataset_config="", starting_model="", folder_name="", metamorphic_test_result="",
                 differential_test_result="", fuzzing_test_result=""):
        self.name = name
        self.model = model
        self.date_time_trained = date_time_trained
        self.total_training_time = total_training_time
        self.number_of_images = number_of_images
        self.path = path
        self.epoch = epoch
        self.box_loss = box_loss
        self.cls_loss = cls_loss
        self.mAP_50 = mAP_50
        self.mAP_50_95 = mAP_50_95
        self.precision = precision
        self.recall = recall
        self.dataset_config = dataset_config
        self.starting_model = starting_model
        self.folder_name = folder_name
        self.metamorphic_test_result = metamorphic_test_result
        self.differential_test_result = differential_test_result
        self.fuzzing_test_result = fuzzing_test_result

    @classmethod
    def fromPath(cls, file_path):
        """Alternative constructor: Load config from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Info file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            data["path"] = os.path.dirname(file_path)
            return cls(**data)  # Create a Config instance from JSON data

    @classmethod
    def from_json(cls, json_str):
        """Creates an instance from a JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def to_dict(self):
        """Converts self into a JSON string"""
        return {
            "name": self.name,
            "model": self.model,
            "date_time_trained": self.date_time_trained,
            "total_training_time": self.total_training_time,
            "number_of_images": self.number_of_images,
            "path": self.path,
            "epoch": self.epoch,
            "box_loss": self.box_loss,
            "cls_loss": self.cls_loss,
            "mAP_50": self.mAP_50,
            "mAP_50_95": self.mAP_50_95,
            "precision": self.precision,
            "recall": self.recall,
            "dataset_config": self.dataset_config,
            "starting_model": self.starting_model,
            "folder_name": self.folder_name,
            "metamorphic_test_result": self.metamorphic_test_result,
            "differential_test_result": self.differential_test_result,
            "fuzzing_test_result": self.fuzzing_test_result
        }

    def to_json(self):
        """Returns an instance of the object as a Json string."""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=4, ensure_ascii=False)
        return json_str

    def save_to_json(self):
        """Saves the configuration to a JSON file at the given folder path."""
        os.makedirs(self.path, exist_ok=True)  # Ensure the folder exists
        file_path = os.path.join(self.path, "info.json")

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.to_dict(), file, indent=4)  # Save with indentation

        print(f"Configuration saved to {file_path}")

    def get_results_png(self):
        """Gets the results png"""
        from PySide6.QtGui import QPixmap
        return QPixmap(self.path + "/results.png")

    def get_best_pt_path(self) -> str:
        """Gets the location of the best trained weights for the model"""
        valid_path = os.path.join(self.path, "weights", "best.pt")
        return valid_path
