import sys
import os

# Add the parent directory (Dissertation) to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.format_converter import convert_gt_to_yolo
from stages.train_model import train_yolo

def run_k_fold(image_path, output_path, k=5, seed=42):
    '''K-fold across dataset - detefects_by_folder defines whether folders seperate groups of images of the same defect. (required to avoid training images leaking into test set)'''
    defect_folders = [f for f in os.listdir(image_path) 
                      if os.path.isdir(os.path.join(image_path, f))]
    
    random.seed(seed)
    random.shuffle(defect_folders)

    folds = [defect_folders[i::k] for i in range(k)]

    for fold_idx, fold_folders in enumerate(folds):
        fold_name = f"fold_{fold_idx + 1}"
        fold_dir = os.path.join(output_path, fold_name)

        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)
            print(f"Cleared existing {fold_name}")

        img_dir = os.path.join(fold_dir, "images")
        lbl_dir = os.path.join(fold_dir, "labels")
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)

        print(f"Building {fold_name} ({len(fold_folders)} defect folders)...")

        for folder in fold_folders:
            folder_path = os.path.join(image_path, folder)
            gt_folder = os.path.join(folder_path, "gt")

            # Find gt file
            gt_file = None
            if os.path.isdir(gt_folder):
                gt_files = [f for f in os.listdir(gt_folder) if f.endswith('.txt')]
                if gt_files:
                    gt_file = os.path.join(gt_folder, gt_files[0])

            # Copy images with folder prefix
            for file in os.listdir(folder_path):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                src = os.path.join(folder_path, file)
                dst = os.path.join(img_dir, f"{folder}_{file}")
                shutil.copy(src, dst)

            # Convert gt to YOLO format labels into a temp dir, then rename with prefix
            if gt_file:
                temp_lbl_dir = os.path.join(fold_dir, "labels_temp")
                os.makedirs(temp_lbl_dir, exist_ok=True)

                convert_gt_to_yolo(gt_file, folder_path, temp_lbl_dir, class_id=0)

                # Rename converted labels to include folder prefix
                for lbl_file in os.listdir(temp_lbl_dir):
                    src_lbl = os.path.join(temp_lbl_dir, lbl_file)
                    dst_lbl = os.path.join(lbl_dir, f"{folder}_{lbl_file}")
                    shutil.move(src_lbl, dst_lbl)

                shutil.rmtree(temp_lbl_dir)
            else:
                # No gt file - create empty labels for all images
                for file in os.listdir(folder_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        lbl_name = f"{folder}_{os.path.splitext(file)[0]}.txt"
                        open(os.path.join(lbl_dir, lbl_name), 'w').close()

        print(f"  -> {len(os.listdir(img_dir))} images, {len(os.listdir(lbl_dir))} labels in {fold_name}")

def train_k_fold(folds_path="Folds"):
    all_folds = sorted([d for d in os.listdir(folds_path) if d.startswith("fold_")])

    for fold in all_folds:
        fold_path = os.path.join(folds_path, fold)
        
        # All other folds are training data
        train_dirs = [os.path.join(folds_path, d) for d in all_folds if d != fold]
        val_dir = os.path.abspath(fold_path)

        yaml_content = {
            'path': os.path.abspath(folds_path),
            'train': [os.path.join(os.path.abspath(d), "images") for d in train_dirs],
            'val': os.path.join(os.path.abspath(fold_path), "images"),
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


run_k_fold("Castings", output_path="Folds", k=8)
train_k_fold("Folds")
