python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r hpc_requirements.txt


#source /users/acb22re/CopiedDissertation/take2/X-Ray_Image_Analysis/.venv/bin/activate

python -m helpers.train_terminal \
  --data_yaml "data/dataset_yaml/dataset.yaml" \
  --model_info '{"name":"hpc_model_flips","model":"YOLOv5","number_of_images":"3000","date_time_trained":"","total_training_time":"","path":"","epoch":"100","box_loss":"","cls_loss":"","mAP_50":"","mAP_50_95":"","precision":"","recall":"","dataset_config":"All Images","starting_model":"","folder_name":"","metamorphic_test_result":"","differential_test_result":"","fuzzing_test_result":""}' \
  --model_dir "trained_models/hpc_model_flips" \
  --weights yolov5mu.pt \
  --img_size 1280 \
  --batch_size 12 \
  --epochs 50 \
  --flips True


echo "Training completed!"