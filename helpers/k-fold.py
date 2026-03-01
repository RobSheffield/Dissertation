import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
import os
import numpy as np
import shutil
from stages.train_model import train_yolo
import datetime


def run_k_fold(image_path, defects_by_folder = False, k=5):
    '''K-fold across dataset - detefects_by_folder defines whether folders seperate groups of images of the same defect. (required to avoid training images leaking into test set)'''
    if defects_by_folder:
        defect_folders = [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]
        random.shuffle(defect_folders)
        folds = [defect_folders[i::k] for i in range(k)]
        for i, fold_folders in enumerate(folds):
            fold_name = f"fold_{i + 1}"
    
            fold_dir = os.path.join("Folds", fold_name)
            if os.path.exists(fold_dir):
                shutil.rmtree(fold_dir)
            os.makedirs(fold_dir, exist_ok=True)

            print(f"Building {fold_name} ({len(fold_folders)} defect folders)...")

            for folder in fold_folders:
                folder_path = os.path.join(image_path, folder)

                for file in os.listdir(folder_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        src = os.path.join(folder_path, file)
                        dst = os.path.join(fold_dir, f"{folder}_{file}")
                        shutil.copy(src, dst)
                        
                        annotation_file = os.path.splitext(src)[0] + '.txt'
                        if os.path.exists(annotation_file):
                            dst_annotation = os.path.splitext(dst)[0] + '.txt'
                            shutil.copy(annotation_file, dst_annotation)

def train_k_fold(folds_path="Folds"):
    all_folds = sorted([d for d in os.listdir(folds_path) if d.startswith("fold_")])

    for fold in all_folds:
        fold_path = os.path.join(folds_path, fold)
        
        # All other folds are training data
        train_dirs = [os.path.join(folds_path, d) for d in all_folds if d != fold]
        val_dir = os.path.abspath(fold_path)

        yaml_content = {
            'path': os.path.abspath(folds_path),
            'train': [os.path.abspath(d) for d in train_dirs],  # list of paths
            'val': val_dir,
            'nc': 1,
            'names': ['defect']
        }

        yaml_path = os.path.join(fold_path, "data.yaml")
        with open(yaml_path, 'w') as f:
            f.write(f"path: {yaml_content['path']}\n")
            f.write(f"train: {yaml_content['train']}\n")
            f.write(f"val: {yaml_content['val']}\n")
            f.write(f"nc: {yaml_content['nc']}\n")
            f.write(f"names: {yaml_content['names']}\n")

        train_yolo(
            data_yaml=yaml_path,
            model_info="model_info.json",
            training_start=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            model_dir=os.path.join("models", fold),
            weights="yolov5m.pt",
            img_size="640",
            batch_size="16",
            epochs="50"
        )


run_k_fold("Castings", defects_by_folder=True, k=8)
train_k_fold("Folds")
