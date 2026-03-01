import argparse
import os
import sys
import pandas as pd
import torch
import shutil
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_classes.model_info import ModelInfo
from helpers import file_helpers


def train_yolo(data_yaml, model_info, training_start, model_dir,
               weights="yolov5m.pt", img_size="640", batch_size="16", epochs="50", flips = False):
    """
    Train a YOLOv5 model with the specified parameters.

    Args:
        data_yaml (str): Path to the dataset YAML file.
        weights (str): Pretrained weights to use (e.g., 'yolov5m.pt').
        img_size (int): Image size for training (e.g., 640).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        output_root (str): Root directory where trained models should be saved

    :return: Path to created model
    :rtype: str
    """
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go to the parent directory
    parent_dir = os.path.dirname(current_dir)

    # Build the path to the target file inside the parent directory
    runs_dir = os.path.join(parent_dir, "runs", "detect")

    # Check if CUDA (GPU) is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device.upper()}...")

    model = YOLO(weights)

    # Train the model
    print("data_yaml:", data_yaml+" | model_dir: "+model_dir+" | weights: "+weights+" | img_size: "+img_size+" | batch_size: "+batch_size+" | epochs: "+epochs)

    if not flips:
        results = model.train(
            data=data_yaml,
            imgsz=int(img_size),
            batch=int(batch_size),
            epochs=int(epochs),
            cache=True,
            device=device)
    else:
        results = model.train(
            data=data_yaml,
            imgsz = int(img_size),
            batch = int(batch_size),
            epochs = int(epochs),
            cache = True,
            device = device,
            fliplr = 0.5,
            flipud=0.5)

    save_dir = results.save_dir

    print(f"Saving model artifacts in: {model_dir}")

    # Move contents to new location
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    shutil.copytree(save_dir,model_dir)
    shutil.rmtree(save_dir)
    results_df = (pd.read_csv(os.path.join(model_dir, "results.csv"))).iloc[-1].to_dict()

    minutes_training = round(float(results_df["time"]) / 60, 3)
    hours_training = round(float(results_df["time"]) / 3600, 3)
    total_train_time_string = f"{minutes_training} Minutes ({hours_training} Hours)"

    model_info_object = ModelInfo.from_json(model_info)

    model_info_object.path = model_dir
    model_info_object.date_time_trained = training_start
    model_info_object.total_training_time = total_train_time_string
    model_info_object.recall = results_df["metrics/recall(B)"]
    model_info_object.precision = results_df["metrics/precision(B)"]
    model_info_object.epoch = results_df["epoch"]
    model_info_object.box_loss = results_df["train/box_loss"]
    model_info_object.cls_loss = results_df["train/cls_loss"]
    model_info_object.mAP_50 = results_df["metrics/mAP50(B)"]
    model_info_object.mAP_50_95 = results_df["metrics/mAP50-95(B)"]
    model_info_object.folder_name = file_helpers.get_folder_name_from_path(model_dir)

    model_info_object.save_to_json()

    print(f"Model saved at: {model_dir}")

    return model_dir  # Return the directory where the model was saved


if __name__ == "__main__":
    """ This file is designed to be ran from the command line or by the program creating a multiprocess to run it.
        The main file handles the arguments provided when it is called, and allow the program to train a valid model
        based on the information provided."""

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Train a YOLOv5 model with the specified parameters.")

    # Arguments
    parser.add_argument("--data_yaml", type=str)
    parser.add_argument("--model_info", type=str)
    parser.add_argument("--training_start", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--weights", type=str, default="yolov5m.pt")
    parser.add_argument("--img_size", type=str, default="640")
    parser.add_argument("--batch_size", type=str, default="16")
    parser.add_argument("--epochs", type=str, default="50")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    train_yolo(
        data_yaml=args.data_yaml,
        model_info=args.model_info,
        training_start=args.training_start,
        model_dir=args.model_dir,
        weights=args.weights,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs
    )