import argparse
from datetime import datetime
import os
import sys
import pandas as pd
import torch
import shutil
from ultralytics import YOLO
from stages import test_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_classes.model_info import ModelInfo
from helpers import file_helpers

if __name__ == "__main__":
    pass



def train_yolo_guided(data_yaml, model_info, training_start, model_dir,
               weights="yolov5m.pt", img_size="640", batch_size="16"):
    # train the model on failed metamorphic tests
    tester = test_model.TestModelStage(model_info=model_info, model_dir=model_dir)
    pass