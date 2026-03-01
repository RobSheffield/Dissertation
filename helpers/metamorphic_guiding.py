import argparse
from datetime import datetime
import os
import sys
import pandas as pd
import torch
import shutil
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_classes.model_info import ModelInfo
from helpers import file_helpers

if __name__ == "__main__":
    pass



def train_yolo_guided(data_yaml, model_info, training_start, model_dir,
               weights="yolov5m.pt", img_size="640", batch_size="16"):
    # train the model on failed metamorphic tests
    # Load the model info
    model_info_object = ModelInfo.from_json(model_info)

    # Get failed test cases from metamorphic testing
    failed_cases = model_info_object.metamorphic_test_result

    if not failed_cases:
        print("No failed metamorphic tests found. Skipping guided training.")
        return

    print(f"Training guided model on failed metamorphic test cases...")

    # Check if CUDA (GPU) is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device.upper()}...")

    # Load base model
    model = YOLO(weights)

    # Train with augmentation focused on failed transformations
    results = model.train(
        data=data_yaml,
        imgsz=int(img_size),
        batch=int(batch_size),
        epochs=50,
        cache=True,
        device=device,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        augment=True
    )

    # Save trained model
    save_dir = results.save_dir
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    shutil.copytree(save_dir, model_dir)
    shutil.rmtree(save_dir)

    # Update model info
    results_df = pd.read_csv(os.path.join(model_dir, "results.csv")).iloc[-1].to_dict()
    model_info_object.path = model_dir
    model_info_object.date_time_trained = training_start
    model_info_object.total_training_time = f"{round(float(results_df['time']) / 60, 3)} Minutes"
    model_info_object.mAP_50 = results_df.get("metrics/mAP50(B)", 0)
    model_info_object.mAP_50_95 = results_df.get("metrics/mAP50-95(B)", 0)
    model_info_object.save_to_json()

    print(f"Guided model saved at: {model_dir}")
    pass