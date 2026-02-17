import shutil
import os
from format_converter import convert_gt_to_yolo

# Use absolute paths
base_path = r"c:\Users\Rob\Documents\Dissertation\take2\X-Ray_Image_Analysis"

# Destination paths
dest_images = f"{base_path}/stored_training_images/images/raw/"
dest_labels = f"{base_path}/stored_training_images/labels/raw/"

# Create destination directories if they don't exist
os.makedirs(dest_images, exist_ok=True)
os.makedirs(dest_labels, exist_ok=True)

# Loop through all 87 casting folders (C0001 to C0087)
for i in range(1, 88):
    folder_name = f"C{i:04d}"  # Formats as C0001, C0002, ..., C0087
    source_folder = f"{base_path}/Castings/{folder_name}/"
    
    if not os.path.exists(source_folder):
        print(f"Skipping {folder_name} - folder not found")
        continue
    
    print(f"Processing {folder_name}...")
    
    # Convert labels
    gt_file = f"{source_folder}/ground_truth.txt"
    if os.path.exists(gt_file):
        convert_gt_to_yolo(
            gt_file=gt_file,
            images_dir=source_folder,
            output_dir=dest_labels,
            class_id=0
        )
    else:
        print(f"  No ground_truth.txt found in {folder_name}")
    
    # Copy images
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(source_folder, filename)
            dst = os.path.join(dest_images, filename)
            shutil.copy2(src, dst)
            print(f"  Copied: {filename}")

print("Done! All images and labels imported.")