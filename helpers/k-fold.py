import sys
import os
import random
import shutil
import datetime
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stages.train_model import train_yolo
from data.format_converter import convert_gt_to_yolo

def run_k_fold(image_path, output_path, k=5):
    '''K-fold across dataset - detefects_by_folder defines whether folders seperate groups of images of the same defect. (required to avoid training images leaking into test set)'''
    defect_folders = [f for f in os.listdir(image_path) 
                      if os.path.isdir(os.path.join(image_path, f))]
    
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
            # Find gt file directly in folder (e.g. ground_truth.txt)
            gt_file = None
            direct_gt = os.path.join(folder_path, "ground_truth.txt")
            if os.path.isfile(direct_gt):
                gt_file = direct_gt


            if gt_file:
                # Convert labels first
                temp_lbl_dir = os.path.join(fold_dir, "labels_temp")
                os.makedirs(temp_lbl_dir, exist_ok=True)
                convert_gt_to_yolo(gt_file, folder_path, temp_lbl_dir, class_id=0)

                # Track which stems have labels
                converted_stems = set()
                for lbl_file in os.listdir(temp_lbl_dir):
                    src_lbl = os.path.join(temp_lbl_dir, lbl_file)
                    dst_lbl = os.path.join(lbl_dir, f"{folder}_{lbl_file}")
                    shutil.move(src_lbl, dst_lbl)
                    converted_stems.add(os.path.splitext(lbl_file)[0])

                shutil.rmtree(temp_lbl_dir)

                for file in os.listdir(folder_path):
                    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    stem = os.path.splitext(file)[0]
                    if stem in converted_stems:
                        src = os.path.join(folder_path, file)
                        dst = os.path.join(img_dir, f"{folder}_{file}")
                        shutil.copy(src, dst)
                    else:
                        print(f"  Skipped (no label): {file}")
            else:
                # No gt file - treat all images as negative cases with empty labels
                images_in_folder = [f for f in os.listdir(folder_path)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if images_in_folder:
                    print(f"  No gt file for {folder} - treating {len(images_in_folder)} images as negative cases")
                    for file in images_in_folder:
                        src = os.path.join(folder_path, file)
                        dst = os.path.join(img_dir, f"{folder}_{file}")
                        shutil.copy(src, dst)
                        # Empty label = no defects
                        lbl_dst = os.path.join(lbl_dir, f"{folder}_{os.path.splitext(file)[0]}.txt")
                        open(lbl_dst, 'w').close()
                else:
                    print(f"WARNING: {folder} has no gt file and no images, skipping.")

        print(f"  -> {len(os.listdir(img_dir))} images, {len(os.listdir(lbl_dir))} labels in {fold_name}")


def run_k_fold_temp(image_path, output_path, k=5):
    '''Creates fold directories, then for each fold iteration creates a single merged train 
    directory, trains, then deletes it before moving to the next fold - to save file quota'''
    
    # First build the base folds
    run_k_fold(image_path, output_path, k)

    all_folds = sorted([
        d for d in os.listdir(output_path)
        if d.startswith("fold_") and os.path.isdir(os.path.join(output_path, d))
    ])

    model_info_json = '{"name":"k_fold","model":"YOLOv5","number_of_images":"","date_time_trained":"","total_training_time":"","path":"","epoch":"","box_loss":"","cls_loss":"","mAP_50":"","mAP_50_95":"","precision":"","recall":"","dataset_config":"K-Fold","starting_model":"","folder_name":"","metamorphic_test_result":"","differential_test_result":"","fuzzing_test_result":""}'

    for fold in all_folds:
        fold_path = os.path.join(output_path, fold)

        # Create merged train dir just for this fold
        temp_dir = os.path.join(fold_path, "train_merged")
        temp_img_dir = os.path.join(temp_dir, "images")
        temp_lbl_dir = os.path.join(temp_dir, "labels")
        os.makedirs(temp_img_dir)
        os.makedirs(temp_lbl_dir)

        # Merge k-1 folds into temp dir
        for other_fold in all_folds:
            if other_fold == fold:
                continue

            src_img_dir = os.path.join(output_path, other_fold, "images")
            src_lbl_dir = os.path.join(output_path, other_fold, "labels")

            for file in os.listdir(src_img_dir):
                shutil.copy(os.path.join(src_img_dir, file), os.path.join(temp_img_dir, file))
            for file in os.listdir(src_lbl_dir):
                shutil.copy(os.path.join(src_lbl_dir, file), os.path.join(temp_lbl_dir, file))

        print(f"{fold} merged train: {len(os.listdir(temp_img_dir))} images, {len(os.listdir(temp_lbl_dir))} labels")

        # Write yaml pointing to merged train and this fold's val
        val_dir = os.path.join(os.path.abspath(fold_path), "images")
        yaml_content = {
            'path': os.path.abspath(output_path),
            'train': os.path.abspath(temp_img_dir),
            'val': val_dir,
            'nc': 1,
            'names': ['defect']
        }
        yaml_path = os.path.join(fold_path, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        print(f"Training fold {fold} ({all_folds.index(fold)+1}/{len(all_folds)})...")

        train_yolo(
            data_yaml=yaml_path,
            model_info=model_info_json,
            training_start=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            model_dir=os.path.join("models", fold),
            weights="yolov5m.pt",
            img_size="768",
            batch_size="16",
            epochs="50"
        )

        # Delete merged dir immediately after training to save file quota
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp dir for {fold}")
        print(f"Finished fold {fold}")

    print("All folds complete!")

if __name__ == '__main__':
    run_k_fold_temp("Castings", output_path="Folds_full_temp_paths_2", k=8)
